"""Stage 3 - Edge Proposal. Only request steps are broadcast. Every robot, in parallel, judges
every request with its own (private) knowledge:

    target robot  -> accept | reject
    other robots  -> volunteer | ignore
    any robot     -> already_doing (+ covered_by: an existing own step) for ASK_HELP

already_doing adds NO step: the existing step is linked to the request
(a candidate, PROPOSED; targeted if the robot is the target). This stops a robot from adding a second
copy of work it already does.

Accept / volunteer inserts ONE provider step (HELP or PASS) into the robot's OWN plan. The step is
born with `answers=<request id>`, so pairing is by id, never by text matching.
Every answer is only a CANDIDATE (PROPOSED edge). Nothing is confirmed here:
    accept    -> PROPOSED, targeted=True   (the robot the requester named agreed)
    volunteer -> PROPOSED, targeted=False  (self-nominated)
Stage 4 confirms: one candidate -> by rule, several -> the Graph LLM chooses.

Code checks each answer like Stage 2: HELP must cite a valid can_do index and respect mobility,
PASS must hand over an item the robot actually has and has not already promised.
"""
from __future__ import annotations

import asyncio
from collections import Counter

from .llm import BaseLLM
from .log import EventLog
from .prompts import PROPOSE_USER, _j, indexed, propose_system
from .schemas import (CONFIRMED, HELP, HELP_EDGE, PASS, PROPOSED, PROVIDER_FOR,
                      REQUEST_TYPES, TRANSFER, Edge, Node, Offer, TaskConfig, agent_num, norm)


def _plan_view(nodes: list[Node]) -> list[dict]:
    return [{"id": n.id, "type": n.type, "text": n.text(), "target": n.target} for n in nodes]


async def _judge(cfg: TaskConfig, agent: str, plans: dict[str, list[Node]], offers: dict[str, Offer],
                 requests: list[Node], passes: list[Node], llm: BaseLLM, log: EventLog
                 ) -> tuple[list[dict], list[dict]]:
    visible = [r for r in requests if r.agent != agent]
    to_me = [p for p in passes if p.target == agent]
    if not visible and not to_me:
        log.log("propose", agent, "skip_no_requests")
        return [], []
    user = PROPOSE_USER.format(task=cfg.task, own_offer=_j(offers[agent].to_dict()),
                               can_do_indexed=indexed(offers[agent].can_do),
                               own_plan=_j(_plan_view(plans[agent])),
                               requests=_j([r.brief() | {"location": r.location} for r in visible]) if visible
                               else "(none)",
                               offers=_j([p.brief() | {"for": p.serves} for p in to_me]) if to_me else "(none)")
    d = await llm.complete(propose_system(agent), user, key=f"propose:{agent}")
    js, os_ = d.get("judgments", []) or [], d.get("offers", []) or []
    log.log("propose", agent, "judged", n=len(js), offers=len(os_))
    return js, os_


def _insert(order: list[Node], node: Node, after: str | None) -> None:
    if not after or str(after).upper() == "START":
        order.insert(0, node)
        return
    for i, n in enumerate(order):
        if n.id == after:
            order.insert(i + 1, node)
            return
    order.append(node)       # unknown anchor -> end of own plan


async def edge_proposal(cfg: TaskConfig, plans: dict[str, list[Node]], offers: dict[str, Offer],
                        llm: BaseLLM, log: EventLog) -> tuple[list[Edge], list[dict]]:
    """Mutates `plans` (inserts provider steps). Returns collaboration edges and a judgment record."""
    requests = [n for a in cfg.ids for n in plans[a] if n.type in REQUEST_TYPES and n.status == "active"]
    passes = [n for a in cfg.ids for n in plans[a] if n.type == PASS and n.origin == "offer" and n.status == "active"]
    by_id = {r.id: r for r in requests}
    by_pass = {p.id: p for p in passes}
    raw = await asyncio.gather(*[_judge(cfg, a, plans, offers, requests, passes, llm, log) for a in cfg.ids])

    edges: list[Edge] = []
    record: list[dict] = []
    counters = {a: max([int(n.id.split("_s")[1]) for n in plans[a]] or [0]) for a in cfg.ids}
    stock = {a: Counter(norm(x) for x in offers[a].has_items) for a in cfg.ids}
    for p in passes:                                   # offered items are already promised
        stock[p.agent][norm(p.item)] -= 1

    def linked(pid: str) -> bool:
        return any(e.src == pid for e in edges)

    for agent, (judgments, _) in zip(cfg.ids, raw):
        me = cfg.agent(agent)
        seen: set[str] = set()
        # accepts first, so promised items go to the robot that asked us directly
        order = {"already_doing": 0, "accept": 1}
        judgments = sorted(judgments, key=lambda j: order.get(str(j.get("decision", "")).lower(), 2))
        for j in judgments:
            rid = str(j.get("request", ""))
            req = by_id.get(rid)
            dec = str(j.get("decision", "ignore")).lower()
            rec = {"agent": agent, "request": rid, "decision": dec, "reason": j.get("reason", "")}
            if req is None or req.agent == agent or rid in seen:
                rec["result"] = "invalid_request_id"
                record.append(rec)
                continue
            seen.add(rid)
            is_target = req.target == agent
            if dec == "already_doing":
                cov = str(j.get("covered_by", ""))
                own = {n.id: n for n in plans[agent] if n.status == "active"}
                problem = None
                if req.type != "ASK_HELP":
                    problem = "already_doing is only for ASK_HELP"
                elif cov not in own or own[cov].type not in ("LOCAL", "HELP"):
                    problem = f"covered_by '{cov}' is not an active LOCAL/HELP step of {agent}"
                elif any(e.src == cov and e.dst == rid for e in edges):
                    problem = "already linked"
                if problem:
                    rec["result"] = f"rejected_by_check: {problem}"
                    record.append(rec)
                    log.log("propose", agent, "answer_rejected_by_check", request=rid)
                    continue
                edges.append(Edge(cov, rid, HELP_EDGE, PROPOSED, "stage3", targeted=is_target))
                rec.update(result="CANDIDATE", provider=cov, covered=True, targeted=is_target)
                record.append(rec)
                log.log("propose", agent, "already_doing", request=rid, provider=cov, edge="CANDIDATE")
                continue
            if is_target and dec == "volunteer":
                dec = "accept"
            if not is_target and dec == "accept":
                dec = "volunteer"
            if not is_target and dec == "reject":
                dec = "ignore"
            rec["decision"] = dec
            if dec not in ("accept", "volunteer"):
                rec["result"] = dec
                record.append(rec)
                continue

            ptype = PROVIDER_FOR[req.type]
            problem = None
            uses, item, action = None, None, str(j.get("action", "") or "")
            if ptype == HELP:
                uses = j.get("uses")
                if not (isinstance(uses, int) and 0 <= uses < len(offers[agent].can_do)):
                    problem = f"uses={uses} is not a valid can_do index"
                elif not me.profile.mobile and norm(req.location) != norm(me.profile.room):
                    problem = f"cannot leave {me.profile.room} to help in {req.location}"
                action = action or req.action
                location = req.location
            else:
                item = str(j.get("item") or req.item or "")
                mine = next((p for p in passes if p.agent == agent and p.target == req.agent
                             and norm(p.item) == norm(item) and not linked(p.id)), None)
                if mine is not None:                  # I already offered exactly this: link, no copy
                    edges.append(Edge(mine.id, rid, TRANSFER, PROPOSED, "stage3", targeted=dec == "accept"))
                    mine.answers = rid
                    rec.update(result="CANDIDATE", provider=mine.id, linked_offer=True)
                    record.append(rec)
                    log.log("propose", agent, "linked_own_offer", request=rid, provider=mine.id)
                    continue
                if stock[agent][norm(item)] <= 0:
                    problem = f"item '{item}' not available (not in has_items or already promised)"
                elif cfg.mover(agent, req.agent) is None:
                    problem = f"{agent} and {req.agent} are both fixed: nobody can move the object"

                action = action or f"pass {item} to {req.agent}"
                location = me.profile.room          # the mobile end of the handoff moves it
            if problem:
                rec["result"] = f"rejected_by_check: {problem}"
                record.append(rec)
                log.log("propose", agent, "answer_rejected_by_check", request=rid)
                continue
            if ptype == PASS:
                stock[agent][norm(item)] -= 1

            counters[agent] += 1
            node = Node(id=f"r{agent_num(agent)}_s{counters[agent]}", agent=agent, type=ptype,
                        action=action, item=item, target=req.agent, uses=uses, location=location,
                        origin="accept" if dec == "accept" else "volunteer", answers=rid)
            _insert(plans[agent], node, j.get("insert_after"))
            edges.append(Edge(node.id, rid, TRANSFER if ptype == PASS else HELP_EDGE, PROPOSED, "stage3",
                              targeted=dec == "accept"))
            rec.update(result="CANDIDATE", provider=node.id, targeted=dec == "accept")
            record.append(rec)
            log.log("propose", agent, f"{dec}", request=rid, provider=node.id, edge="CANDIDATE")

    # ---- offers: the receiver decides. Receiving adds a RECEIVE (born linked to the PASS by id)
    # and the receiver's own LOCAL step that uses the object.
    for agent, (_, offer_js) in zip(cfg.ids, raw):
        for oj in offer_js:
            oid = str(oj.get("offer", ""))
            p = by_pass.get(oid)
            dec = str(oj.get("decision", "decline")).lower()
            rec = {"agent": agent, "offer": oid, "decision": dec, "reason": oj.get("reason", "")}
            if p is None or p.target != agent:
                rec["result"] = "invalid_offer_id"
                record.append(rec)
                continue
            if dec not in ("receive", "accept"):
                rec["result"] = "decline"
                record.append(rec)
                log.log("propose", agent, "offer_declined", offer=oid)
                continue
            if linked(oid):
                rec["result"] = "already_linked (you requested it)"
                record.append(rec)
                continue
            uses = oj.get("uses")
            if not isinstance(uses, int) or not (0 <= uses < len(offers[agent].can_do)):
                rec["result"] = f"rejected_by_check: uses={uses} is not a valid can_do index"
                record.append(rec)
                log.log("propose", agent, "answer_rejected_by_check", offer=oid)
                continue
            if cfg.mover(p.agent, agent) is None:
                rec["result"] = f"rejected_by_check: {p.agent} and {agent} are both fixed"
                record.append(rec)
                log.log("propose", agent, "answer_rejected_by_check", offer=oid)
                continue
            n_ = agent_num(agent)
            counters[agent] += 1
            use = Node(id=f"r{n_}_s{counters[agent] + 1}", agent=agent, type="LOCAL",
                       action=str(oj.get("action") or f"use the {p.item}"), uses=uses,
                       location=cfg.agent(agent).profile.room, origin="accept", serves=f"offered {p.item}")
            rcv = Node(id=f"r{n_}_s{counters[agent]}", agent=agent, type="RECEIVE", item=p.item,
                       action=f"receive {p.item}", target=p.agent, location=cfg.agent(agent).profile.room,
                       origin="accept", enables=use.id, serves=f"offered {p.item}")
            counters[agent] += 1
            _insert(plans[agent], rcv, oj.get("insert_after"))
            plans[agent].insert(plans[agent].index(rcv) + 1, use)
            p.answers = rcv.id
            edges.append(Edge(oid, rcv.id, TRANSFER, PROPOSED, "stage3", targeted=True))
            rec.update(result="CANDIDATE", provider=oid, receive=rcv.id, use=use.id)
            record.append(rec)
            log.log("propose", agent, "offer_received", offer=oid, receive=rcv.id, use=use.id)
    return edges, record
