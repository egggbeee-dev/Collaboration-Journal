"""Stage 2 - Local Plan. Each robot, in parallel, writes LOCAL / ASK_HELP / RECEIVE steps.

Code then checks every step against the robot's own Offer and profile:
- LOCAL must cite a valid can_do index ("uses"), and an immobile robot must stay in its room.
- ASK_HELP / RECEIVE need a valid target (another robot).
- RECEIVE item must appear in the target's has_items.
On violations the robot gets ONE chance to fix its own plan (still decentralized).
Remaining violations are kept on the node; LOCAL steps with violations are dropped in Stage 4.
"""
from __future__ import annotations

from .llm import BaseLLM
from .log import EventLog
from .prompts import PLAN_FIX, PLAN_USER, _j, indexed, plan_system
from .schemas import (ASK_HELP, LOCAL, RECEIVE, REQUEST_TYPES, Node, Offer, TaskConfig,
                      agent_num, norm, snap_duration)


def validate(raw_steps: list[dict], agent: str, offers: dict[str, Offer], cfg: TaskConfig
             ) -> tuple[list[dict], list[str]]:
    """Return cleaned step dicts (each with a 'violations' list) and readable error strings."""
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

        if t == LOCAL:
            u = step["uses"]
            if not isinstance(u, int) or not (0 <= u < len(own.can_do)):
                v.append(f"LOCAL '{step['action']}' cites uses={u}, not a valid can_do index")
            if not me.profile.mobile and norm(step["location"]) != norm(me.profile.room):
                v.append(f"LOCAL '{step['action']}' is in '{step['location']}' but you cannot leave "
                         f"'{me.profile.room}'")
        else:
            step["uses"] = None
            tgt = step["target"]
            if tgt not in cfg.ids or tgt == agent:
                v.append(f"{t} needs a target among {[x for x in cfg.ids if x != agent]}, got {tgt}")
                step["target"] = None
            if t == ASK_HELP and not step["action"]:
                v.append("ASK_HELP needs an action")
            if t == RECEIVE:
                if not step["item"]:
                    v.append("RECEIVE needs an item")
                elif step["target"] in offers and norm(step["item"]) not in \
                        {norm(x) for x in offers[step["target"]].has_items}:
                    v.append(f"RECEIVE item '{step['item']}' is not in {step['target']}'s has_items "
                             f"{offers[step['target']].has_items}")
                if not step["action"]:
                    step["action"] = f"receive {step['item']}"
        step["violations"] = v
        errors += [f"step {i}: {x}" for x in v]
        cleaned.append(step)
    if not cleaned:
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
    steps, errors = validate(d.get("steps", []), agent, offers, cfg)
    n_fix = 0
    while errors and n_fix < max_fix:
        n_fix += 1
        log.log("plan", agent, "self_fix", errors=len(errors))
        fix_user = user + "\n\nYOUR PREVIOUS PLAN:\n" + _j(d) + "\n\n" + PLAN_FIX.format(
            errors="\n".join(f"- {e}" for e in errors))
        d = await llm.complete(system, fix_user, key=f"plan:{agent}:fix")
        steps, errors = validate(d.get("steps", []), agent, offers, cfg)

    n = agent_num(agent)
    nodes = [Node(id=f"r{n}_s{k}", agent=agent, type=s["type"], action=s["action"], item=s["item"],
                  target=s["target"], uses=s["uses"], location=s["location"], duration=s["duration"],
                  origin="local", violations=s["violations"])
             for k, s in enumerate(steps, start=1)]
    log.log("plan", agent, "broadcast_requests", steps=len(nodes),
            requests=sum(x.type in REQUEST_TYPES for x in nodes),
            violations=sum(bool(x.violations) for x in nodes))
    return nodes, {"reasoning": d.get("reasoning", ""), "fix_rounds": n_fix, "remaining_errors": errors}
