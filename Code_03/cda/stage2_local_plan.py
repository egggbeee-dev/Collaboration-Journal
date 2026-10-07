"""Stage 2 - Local Plan. Each robot, in parallel, writes LOCAL / ASK_HELP / RECEIVE steps.

Code then checks every step against the robot's own Offer and profile:
- LOCAL must cite a valid can_do index ("uses"), and an immobile robot must stay in its room.
- ASK_HELP / RECEIVE need a valid target (another robot).
- RECEIVE item must appear in the target's has_items.
- Every request must name in "enables" an own LATER LOCAL step that depends on it.
  (A request is a dependency of the robot's own work, not a way to hand out parts of the task.)
- Every entry of the robot's own "checklist" (objects that must change) needs a step that names it;
  a heavy object marked LOCAL by a light-duty body, or a step that assumes another object was moved
  first with no step moving it, is a violation.
On violations the robot gets up to TWO chances to fix its own plan (still decentralized). Essential
goals / checklist entries still uncovered after that are reported in the Joint Plan.
Remaining violations: requests are withdrawn before broadcast (never seen by others);
LOCAL steps with violations are dropped in Stage 4.
"""
from __future__ import annotations

import re

from .graph import _tokens, object_of
from .llm import BaseLLM
from .log import EventLog
from .prompts import PLAN_FIX, PLAN_USER, _j, indexed, plan_system
from .schemas import (ASK_HELP, LOCAL, PASS, RECEIVE, REQUEST_TYPES, Node, Offer, TaskConfig,
                      agent_num, norm)


# No exploration in this setting, and movement is part of HELP / PASS / RECEIVE: steps that only look,
# check or search for something do not change the world and are not allowed.
_MOVE_ONLY = re.compile(r"^\s*(go|move|walk|head|travel|return|navigate|come)\s+(back\s+)?(to|into)\b", re.I)
_NON_PHYSICAL = re.compile(r"^\s*(mark|coordinate|conduct|ensure|make sure|confirm|supervise|oversee|monitor|"
                          r"designate|assess|evaluate|decide|plan|verify|review)\b", re.I)
_OBSERVE = re.compile(r"\b(check|checks|checking|search|searching|look for|looking for|find|finding|"
                      r"inspect|verify|scan|locate|explore|availability)\b", re.I)


# "roll the rug after the Glass coffee table has been removed" -> "Glass coffee table"
_AFTER = re.compile(r"\b(?:after|once|when|now that)\s+(?:the\s+)?(.+?)\s+(?:has|have|is|are)\s+(?:been\s+)?"
                    r"(?:removed|moved|cleared|relocated|taken away|pushed|put away|out of the way)", re.I)
_LIGHT_BODY = re.compile(r"cannot (?:move|carry|lift) heavy|light[- ]duty|only (?:carry|lift|move) light", re.I)
COVER = 0.6          # share of an object's words a step must mention to count as handling it


def _cover(obj: set, text: str) -> float:
    return len(obj & _tokens(text)) / len(obj) if obj else 0.0


def _handled(st: dict) -> set:
    """Words of the object a step acts on (not of objects it merely mentions, e.g. in 'after ...')."""
    if st["type"] == RECEIVE:
        return _tokens(st.get("item") or "")
    action = _AFTER.sub("", st.get("action", ""))
    return object_of(action) or _tokens(action)


def checklist_advice(steps: list[dict], checklist: list, embodiment: str) -> tuple[list[str], list[str]]:
    """Each checklist entry must be handled by its own step (one step per entry, matched by the
    object's words). Returns (errors, uncovered entries)."""
    items = [c for c in (checklist or []) if isinstance(c, dict) and str(c.get("object", "")).strip()]
    usable = [(k, st) for k, st in enumerate(steps) if st["type"] in (LOCAL, ASK_HELP, RECEIVE)]
    pairs = []
    for ci, c in enumerate(items):
        obj = _tokens(str(c["object"]))
        how = str(c.get("how", "")).upper().strip()
        for k, st in usable:
            sc = len(obj & _handled(st)) / len(obj) if obj else 0.0
            if sc >= COVER:
                pairs.append((sc + (0.01 if st["type"] == how else 0), ci, k))
    used_c, used_s, match = set(), set(), {}
    for sc, ci, k in sorted(pairs, reverse=True):
        if ci in used_c or k in used_s:
            continue
        used_c.add(ci)
        used_s.add(k)
        match[ci] = k
    errors, uncovered = [], []
    for ci, c in enumerate(items):
        obj, how = str(c["object"]), str(c.get("how", "")).upper().strip()
        heavy = str(c.get("weight", "")).lower().strip() == "heavy"
        if ci not in match:
            uncovered.append(f"{obj}: {c.get('change', '')}".strip(": "))
            errors.append(f"checklist entry '{obj}' ({c.get('change', '')}) has no step that names it. "
                          f"Add a {how or 'LOCAL / ASK_HELP / RECEIVE'} step for it")
            continue
        st = steps[match[ci]]
        if st["type"] == LOCAL and (how == ASK_HELP or heavy and _LIGHT_BODY.search(embodiment or "")):
            errors.append(f"'{obj}' is {'heavy' if heavy else 'marked ASK_HELP'} but step '{st['action']}' does it "
                          f"yourself. Your body cannot: write ASK_HELP to a robot whose body can move it")
    return errors, uncovered


def _check_mover(giver: str, receiver: str, cfg: TaskConfig, v: list[str]) -> None:
    """The object is moved by the mobile end of the handoff. Two fixed robots cannot exchange objects."""
    if giver in cfg.ids and receiver in cfg.ids and cfg.mover(giver, receiver) is None:
        v.append(f"{giver} and {receiver} are both fixed: nobody can move an object between their rooms. "
                 f"Remove this handoff")


def validate(raw_steps: list[dict], agent: str, offers: dict[str, Offer], cfg: TaskConfig,
             goals: list | None = None) -> tuple[list[dict], list[str]]:
    """Return cleaned step dicts (each with a 'violations' list) and readable error strings.
    If `goals` is given (the robot's own goal list), every step must cite one via "serves"."""
    me = cfg.agent(agent)
    own = offers[agent]
    cleaned, errors = [], []
    for i, s in enumerate(raw_steps or []):
        t = str(s.get("type", "")).upper().strip()
        v: list[str] = []
        step = {
            "type": t,
            "action": str(s.get("action", "") or "").strip(),
            "item": (str(s["item"]).strip() if s.get("item") else None),
            "target": (str(s["target"]).strip().upper() if s.get("target") else None),
            "uses": s.get("uses"),
            "location": str(s.get("location") or me.profile.room),
        }
        if t not in {LOCAL, PASS} | REQUEST_TYPES:
            errors.append(f"step {i}: type '{t}' is not allowed (use LOCAL, ASK_HELP, RECEIVE, PASS)")
            continue
        if t == PASS:                                   # an OFFER: the receiver decides in Stage 3
            step["uses"] = None
            step["location"] = me.profile.room
            if step["target"] not in cfg.ids or step["target"] == agent:
                v.append(f"PASS needs a target among {[x for x in cfg.ids if x != agent]}, got {step['target']}")
            if not step["item"] or norm(step["item"]) not in {norm(x) for x in own.has_items}:
                v.append(f"PASS item '{step['item']}' must be copied from YOUR has_items")
            step["action"] = step["action"] or f"pass {step['item']} to {step['target']}"
            _check_mover(agent, step["target"], cfg, v)

        if goals is not None:
            sv = s.get("serves")
            if isinstance(sv, str) and sv.strip().isdigit():
                sv = int(sv.strip())
            if not isinstance(sv, int) or not (0 <= sv < len(goals)):
                v.append(f"step must cite in 'serves' the index of one of your goals (got {sv}); "
                         f"drop steps that serve none of the task's goals")
                step["serves"] = None
            else:
                g = goals[sv]
                step["serves"] = str(g.get("goal", g) if isinstance(g, dict) else g)
        own_items = {norm(x) for x in own.has_items}
        others = [x for x in cfg.ids if x != agent]
        if t == ASK_HELP and any(it and it in norm(step["action"]) for it in own_items):
            v.append(f"ASK_HELP '{step['action']}' is about your own object. Handoffs go only one way: "
                     f"the robot that needs it writes RECEIVE; you write nothing about it now")
        if t == LOCAL and any(re.search(rf"\b{o}\b", step["action"], re.I) for o in others):
            v.append(f"LOCAL '{step['action']}' hands something to another robot. Do not plan handoffs: "
                     f"the receiver writes RECEIVE and you answer later with a PASS")
        if t == LOCAL and _NON_PHYSICAL.search(step["action"]):
            v.append(f"LOCAL '{step['action']}' is not a physical action. Write what your body does to an "
                     f"object (move X from A to B, open, switch on, fill). If your body cannot do the real "
                     f"work (e.g. heavy furniture), do NOT drop it: write ASK_HELP to a robot that can")
        m = _AFTER.search(step["action"]) if t in (LOCAL, ASK_HELP) else None
        if m:
            obj = _tokens(m.group(1))
            earlier = [x for x in (raw_steps or [])[:i] if str(x.get("type", "")).upper().strip() in (LOCAL, ASK_HELP)
                       and _cover(obj, str(x.get("action", ""))) >= COVER]
            if obj and not earlier:
                v.append(f"'{step['action']}' assumes '{m.group(1)}' was moved first, but no earlier step of "
                         f"yours moves it. Add that step before it (LOCAL if your body can, otherwise ASK_HELP "
                         f"with 'enables' = this step)")
        if t in (LOCAL, ASK_HELP) and (_OBSERVE.search(step["action"]) or _MOVE_ONLY.search(step["action"])):
            v.append(f"{t} '{step['action']}' only looks/checks/searches. Robots cannot explore and "
                     f"movement is part of HELP/PASS/RECEIVE: write only steps that change the room "
                     f"(what the others can see is in their offers)")
        if t == LOCAL:
            u = step["uses"]
            if not isinstance(u, int) or not (0 <= u < len(own.can_do)):
                v.append(f"LOCAL '{step['action']}' cites uses={u}, not a valid can_do index")
            if norm(step["location"]) != norm(me.profile.room):
                v.append(f"LOCAL '{step['action']}' is in '{step['location']}', but you work only in your "
                         f"own room ('{me.profile.room}'). For objects from another room write RECEIVE; "
                         f"for work your body cannot do, write ASK_HELP")
            step["location"] = me.profile.room
        elif t in REQUEST_TYPES:
            step["uses"] = None
            tgt = step["target"]
            if tgt not in cfg.ids or tgt == agent:
                v.append(f"{t} needs a target among {[x for x in cfg.ids if x != agent]}, got {tgt}")
                step["target"] = None
            if t == ASK_HELP and not step["action"]:
                v.append("ASK_HELP needs an action")
            if t == RECEIVE:
                step["location"] = me.profile.room
                _check_mover(step["target"], agent, cfg, v)
                if not step["item"]:
                    v.append("RECEIVE needs an item")
                elif step["target"] in offers and norm(step["item"]) not in \
                        {norm(x) for x in offers[step["target"]].has_items}:
                    v.append(f"RECEIVE item '{step['item']}' is not in {step['target']}'s has_items "
                             f"{offers[step['target']].has_items}")
                state = str(s.get("state") or "").strip()
                if not step["action"] or state:
                    step["action"] = f"receive {step['item']}" + (f" ({state})" if state else "")
        step["violations"] = v
        step["_raw_index"] = i
        step["enables"] = s.get("enables")
        step["prepared_by"] = s.get("prepared_by") if t == PASS else None
        cleaned.append(step)
    # second pass: "enables" must point to an own later LOCAL step (indices refer to raw_steps)
    raw_types = [str(x.get("type", "")).upper().strip() for x in (raw_steps or [])]
    for step in cleaned:
        i = step["_raw_index"]
        pb = step.get("prepared_by")
        if step["type"] == PASS and pb is not None:
            if isinstance(pb, str) and pb.strip().isdigit():
                pb = int(pb.strip())
            if not isinstance(pb, int) or not (0 <= pb < len(raw_types)) or pb == i or raw_types[pb] != LOCAL:
                pb = None                               # optional link: ignore a bad index, keep the offer
            step["prepared_by"] = pb
        if step["type"] in REQUEST_TYPES:
            e = step["enables"]
            if isinstance(e, str) and e.strip().isdigit():
                e = int(e.strip())
            step["enables"] = e
            if not isinstance(e, int) or not (i < e < len(raw_types)) or raw_types[e] != LOCAL:
                step["violations"].append(
                    f"{step['type']} '{step['action'] or step['item']}' must name in 'enables' the index "
                    f"of one of YOUR OWN later LOCAL steps that needs it (got {e}). If none of your "
                    f"steps needs it, remove this request: that part belongs to whoever does it.")
                step["enables"] = None
        else:
            step["enables"] = None
        errors += [f"step {i}: {x}" for x in step["violations"]]
    # one object per step: several RECEIVEs of different objects feeding ONE step means that step
    # handles several objects at once (soft: asks the robot to split it, does not drop anything)
    fed: dict = {}
    for st in cleaned:
        if st["type"] == RECEIVE and isinstance(st.get("enables"), int) and st.get("item"):
            fed.setdefault(st["enables"], set()).add(norm(st["item"]))
    for idx, items in fed.items():
        if len(items) > 1:
            errors.append(f"step {idx} uses {len(items)} received objects at once ({', '.join(sorted(items))}). "
                          f"A robot holds one object at a time: write one LOCAL step per received object, and "
                          f"let each RECEIVE enable its own step")
    if not cleaned and raw_steps:
        errors.append("plan has no valid steps")
    return cleaned, errors


def coverage_advice(steps: list[dict], goals: list) -> list[str]:
    """Essential goals in the robot's own room that no step serves (soft: only triggers the self-fix)."""
    served = {st.get("serves") for st in steps if st.get("serves")}
    out = []
    for g in goals or []:
        if not isinstance(g, dict):
            continue
        text = str(g.get("goal", ""))
        if str(g.get("priority", "")).lower() == "high" and g.get("in_my_room") is True and text not in served:
            out.append(f"ESSENTIAL goal '{text}' in your room has no step. Do it yourself, or if your body "
                       f"cannot, write ASK_HELP to a robot whose body can. Do not leave essential work out.")
    return out


async def make_local_plan(cfg: TaskConfig, agent: str, offers: dict[str, Offer], llm: BaseLLM,
                          log: EventLog, max_fix: int = 2) -> tuple[list[Node], dict]:
    me = cfg.agent(agent)
    own = offers[agent]
    system = plan_system(agent, me.profile.room)
    user = PLAN_USER.format(
        task=cfg.task, deadline=cfg.deadline_min, own_offer=_j(own.to_dict()),
        can_do_indexed=indexed(own.can_do),
        public_offers=_j([o.public() for o in offers.values()]),
        profiles=_j({a.id: {"room": a.profile.room, "mobile": a.profile.mobile,
                            **({"payload_kg": a.profile.payload_kg} if a.profile.payload_kg else {}),
                            "embodiment": a.profile.embodiment} for a in cfg.agents}),
    )
    def check(d: dict):
        steps, errors = validate(d.get("steps", []), agent, offers, cfg, d.get("goals") or [])
        gaps = coverage_advice(steps, d.get("goals") or [])
        cl_err, cl_unc = checklist_advice(steps, d.get("checklist") or [], me.profile.embodiment)
        if gaps or cl_unc:
            log.log("plan", agent, "uncovered", essential=len(gaps), checklist=len(cl_unc))
        return steps, errors + gaps + cl_err, gaps, cl_unc

    d = await llm.complete(system, user, key=f"plan:{agent}")
    steps, errors, gaps, cl_unc = check(d)
    n_fix = 0
    while errors and n_fix < max_fix:
        n_fix += 1
        log.log("plan", agent, "self_fix", errors=len(errors))
        fix_user = user + "\n\nYOUR PREVIOUS PLAN:\n" + _j(d) + "\n\n" + PLAN_FIX.format(
            errors="\n".join(f"- {e}" for e in errors))
        d = await llm.complete(system, fix_user, key=f"plan:{agent}:fix")
        steps, errors, gaps, cl_unc = check(d)

    n = agent_num(agent)
    nodes = [Node(id=f"r{n}_s{k}", agent=agent, type=s["type"], action=s["action"], item=s["item"],
                  target=s["target"], uses=s["uses"], location=s["location"],
                  origin="offer" if s["type"] == PASS else "local", violations=s["violations"],
                  serves=s.get("serves"))
             for k, s in enumerate(steps, start=1)]
    raw_to_id = {s["_raw_index"]: nodes[k].id for k, s in enumerate(steps)}
    for s, node in zip(steps, nodes):
        if s["enables"] is not None:
            node.enables = raw_to_id.get(s["enables"])
        if s.get("prepared_by") is not None:
            node.prepared_by = raw_to_id.get(s["prepared_by"])
        if node.type == PASS and node.violations:          # invalid offer: withdrawn before broadcast
            node.status = "dropped"
            log.log("plan", agent, "offer_withdrawn", node=node.id)
        # a request that is not a dependency of own work is withdrawn before broadcast
        if node.type in REQUEST_TYPES and node.enables is None:
            node.status = "dropped"
            log.log("plan", agent, "request_withdrawn", node=node.id)
    log.log("plan", agent, "broadcast_requests", steps=len(nodes),
            requests=sum(x.type in REQUEST_TYPES and x.status == "active" for x in nodes),
            violations=sum(bool(x.violations) for x in nodes))
    return nodes, {"reasoning": d.get("reasoning", ""), "goals": d.get("goals") or [],
                   "checklist": d.get("checklist") or [], "fix_rounds": n_fix, "remaining_errors": errors,
                   "uncovered_essential": gaps, "uncovered_checklist": cl_unc}
