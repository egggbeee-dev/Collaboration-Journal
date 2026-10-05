"""Stage 2 - Local Plan. Each robot, in parallel, writes LOCAL / ASK_HELP / RECEIVE steps.

Code then checks every step against the robot's own Offer and profile:
- LOCAL must cite a valid can_do index ("uses"), and an immobile robot must stay in its room.
- ASK_HELP / RECEIVE need a valid target (another robot).
- RECEIVE item must appear in the target's has_items.
- Every request must name in "enables" an own LATER LOCAL step that depends on it.
  (A request is a dependency of the robot's own work, not a way to hand out parts of the task.)
On violations the robot gets ONE chance to fix its own plan (still decentralized).
Remaining violations: requests are withdrawn before broadcast (never seen by others);
LOCAL steps with violations are dropped in Stage 4.
"""
from __future__ import annotations

import re

from .llm import BaseLLM
from .log import EventLog
from .prompts import PLAN_FIX, PLAN_USER, _j, indexed, plan_system
from .schemas import (ASK_HELP, LOCAL, RECEIVE, REQUEST_TYPES, Node, Offer, TaskConfig,
                      agent_num, norm, snap_duration)


# No exploration in this setting, and movement is added by the scheduler: steps that only look,
# check or search for something do not change the world and are not allowed.
_MOVE_ONLY = re.compile(r"^\s*(go|move|walk|head|travel|return|navigate|come)\s+(back\s+)?(to|into)\b", re.I)
_OBSERVE = re.compile(r"\b(check|checks|checking|search|searching|look for|looking for|find|finding|"
                      r"inspect|verify|scan|locate|explore|availability)\b", re.I)


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
            "duration": snap_duration(s.get("duration", 2)),
        }
        if t not in {LOCAL} | REQUEST_TYPES:
            errors.append(f"step {i}: type '{t}' is not allowed (use LOCAL, ASK_HELP, RECEIVE)")
            continue

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
        if t in (LOCAL, ASK_HELP) and (_OBSERVE.search(step["action"]) or _MOVE_ONLY.search(step["action"])):
            v.append(f"{t} '{step['action']}' only looks/checks/searches. Robots cannot explore and "
                     f"movement is added automatically: write only steps that change the room "
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
        else:
            step["uses"] = None
            tgt = step["target"]
            if tgt not in cfg.ids or tgt == agent:
                v.append(f"{t} needs a target among {[x for x in cfg.ids if x != agent]}, got {tgt}")
                step["target"] = None
            if t == ASK_HELP and not step["action"]:
                v.append("ASK_HELP needs an action")
            if t == RECEIVE:
                step["location"] = me.profile.room
                if not step["item"]:
                    v.append("RECEIVE needs an item")
                elif step["target"] in offers and norm(step["item"]) not in \
                        {norm(x) for x in offers[step["target"]].has_items}:
                    v.append(f"RECEIVE item '{step['item']}' is not in {step['target']}'s has_items "
                             f"{offers[step['target']].has_items}")
                if not step["action"]:
                    step["action"] = f"receive {step['item']}"
        step["violations"] = v
        step["_raw_index"] = i
        step["enables"] = s.get("enables")
        cleaned.append(step)
    # second pass: "enables" must point to an own later LOCAL step (indices refer to raw_steps)
    raw_types = [str(x.get("type", "")).upper().strip() for x in (raw_steps or [])]
    for step in cleaned:
        i = step["_raw_index"]
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
    if not cleaned and raw_steps:
        errors.append("plan has no valid steps")
    return cleaned, errors


async def make_local_plan(cfg: TaskConfig, agent: str, offers: dict[str, Offer], llm: BaseLLM,
                          log: EventLog, max_fix: int = 1) -> tuple[list[Node], dict]:
    me = cfg.agent(agent)
    own = offers[agent]
    system = plan_system(agent, me.profile.room)
    user = PLAN_USER.format(
        task=cfg.task, deadline=cfg.deadline_min, own_offer=_j(own.to_dict()),
        can_do_indexed=indexed(own.can_do),
        public_offers=_j([o.public() for o in offers.values()]),
        profiles=_j({a.id: {"room": a.profile.room, "mobile": a.profile.mobile,
                            "payload_kg": a.profile.payload_kg} for a in cfg.agents}),
    )
    d = await llm.complete(system, user, key=f"plan:{agent}")
    steps, errors = validate(d.get("steps", []), agent, offers, cfg, d.get("goals") or [])
    n_fix = 0
    while errors and n_fix < max_fix:
        n_fix += 1
        log.log("plan", agent, "self_fix", errors=len(errors))
        fix_user = user + "\n\nYOUR PREVIOUS PLAN:\n" + _j(d) + "\n\n" + PLAN_FIX.format(
            errors="\n".join(f"- {e}" for e in errors))
        d = await llm.complete(system, fix_user, key=f"plan:{agent}:fix")
        steps, errors = validate(d.get("steps", []), agent, offers, cfg, d.get("goals") or [])

    n = agent_num(agent)
    nodes = [Node(id=f"r{n}_s{k}", agent=agent, type=s["type"], action=s["action"], item=s["item"],
                  target=s["target"], uses=s["uses"], location=s["location"], duration=s["duration"],
                  origin="local", violations=s["violations"], serves=s.get("serves"))
             for k, s in enumerate(steps, start=1)]
    raw_to_id = {s["_raw_index"]: nodes[k].id for k, s in enumerate(steps)}
    for s, node in zip(steps, nodes):
        if s["enables"] is not None:
            node.enables = raw_to_id.get(s["enables"])
        # a request that is not a dependency of own work is withdrawn before broadcast
        if node.type in REQUEST_TYPES and node.enables is None:
            node.status = "dropped"
            log.log("plan", agent, "request_withdrawn", node=node.id)
    log.log("plan", agent, "broadcast_requests", steps=len(nodes),
            requests=sum(x.type in REQUEST_TYPES and x.status == "active" for x in nodes),
            violations=sum(bool(x.violations) for x in nodes))
    return nodes, {"reasoning": d.get("reasoning", ""), "goals": d.get("goals") or [],
                   "fix_rounds": n_fix, "remaining_errors": errors}
