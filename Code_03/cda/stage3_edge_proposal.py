"""Stage 3 - Edge Proposal. Only request steps are broadcast. Every robot, in parallel, judges
every request with its own (private) knowledge:

    target robot  -> accept | reject
    other robots  -> volunteer | ignore

Accept / volunteer inserts ONE provider step (HELP or PASS) into the robot's OWN plan. The step is
born with `answers=<request id>`, so pairing is by id, never by text matching.
    accept    -> CONFIRMED edge   (requester asked + provider agreed)
    volunteer -> PROPOSED edge    (self-nominated backup; Stage 4 may keep or release it)

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
                      REQUEST_TYPES, TRANSFER, Edge, Node, Offer, TaskConfig, agent_num, norm,
                      snap_duration)


def _plan_view(nodes: list[Node]) -> list[dict]:
    return [{"id": n.id, "type": n.type, "text": n.text(), "target": n.target, "duration": n.duration}
            for n in nodes]


async def _judge(cfg: TaskConfig, agent: str, plans: dict[str, list[Node]], offers: dict[str, Offer],
                 requests: list[Node], llm: BaseLLM, log: EventLog) -> list[dict]:
    visible = [r for r in requests if r.agent != agent]
    if not visible:
        log.log("propose", agent, "skip_no_requests")
        return []
    user = PROPOSE_USER.format(task=cfg.task, own_offer=_j(offers[agent].to_dict()),
                               can_do_indexed=indexed(offers[agent].can_do),
                               own_plan=_j(_plan_view(plans[agent])),
                               requests=_j([r.brief() | {"location": r.location} for r in visible]))
    d = await llm.complete(propose_system(agent), user, key=f"propose:{agent}")
    js = d.get("judgments", []) or []
    log.log("propose", agent, "judged", n=len(js))
    return js


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
    requests = [n for a in cfg.ids for n in plans[a] if n.type in REQUEST_TYPES]
    by_id = {r.id: r for r in requests}
    raw = await asyncio.gather(*[_judge(cfg, a, plans, offers, requests, llm, log) for a in cfg.ids])

    edges: list[Edge] = []
    record: list[dict] = []
    counters = {a: max([int(n.id.split("_s")[1]) for n in plans[a]] or [0]) for a in cfg.ids}
    stock = {a: Counter(norm(x) for x in offers[a].has_items) for a in cfg.ids}

    for agent, judgments in zip(cfg.ids, raw):
        me = cfg.agent(agent)
        seen: set[str] = set()
        # accepts first, so promised items go to the robot that asked us directly
        judgments = sorted(judgments, key=lambda j: 0 if str(j.get("decision", "")).lower() == "accept" else 1)
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
                if not isinstance(uses, int) or not (0 <= uses < len(offers[agent].can_do)):
                    problem = f"uses={uses} is not a valid can_do index"
                elif not me.profile.mobile and norm(req.location) != norm(me.profile.room):
                    problem = f"cannot leave {me.profile.room} to help in {req.location}"
                action = action or req.action
                location = req.location
            else:
                item = str(j.get("item") or req.item or "")
                if stock[agent][norm(item)] <= 0:
                    problem = f"item '{item}' not available (not in has_items or already promised)"
                action = action or f"pass {item} to {req.agent}"
                location = me.profile.room
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
                        duration=snap_duration(j.get("duration", 2)),
                        origin="accept" if dec == "accept" else "volunteer", answers=rid)
            _insert(plans[agent], node, j.get("insert_after"))
            status = CONFIRMED if dec == "accept" else PROPOSED
            edges.append(Edge(node.id, rid, TRANSFER if ptype == PASS else HELP_EDGE, status, "stage3"))
            rec.update(result=status, provider=node.id)
            record.append(rec)
            log.log("propose", agent, f"{dec}", request=rid, provider=node.id, edge=status)
    return edges, record
