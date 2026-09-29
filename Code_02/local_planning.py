"""Stage 2 - LOCAL PLANNING (each robot, in parallel, one LLM call).

Each robot writes ITS OWN plan. Nobody assigns tasks.
A robot may volunteer (HELP / PASS) for needs that other robots published in their Offers.

    LOCAL     I do this myself.
    ASK_HELP  my body cannot do a task that my part needs
    HELP      I do a task another robot needs (and cannot do)
    RECEIVE   my part needs an item that is not in my room
    PASS      I give one of my items to another robot

Fields per step: type, action, item (RECEIVE / PASS), target (partner hint, optional).
Steps are the NODES of the joint plan; edges are made later by Graph Reasoning.
"""

from __future__ import annotations

from runtime import Agent
from schemas import AgentInput, LocalPlan, Offer, RawLocalPlan


# =============================================================================
# Prompt
# =============================================================================

_EXAMPLE = """EXAMPLE

TASK: "Prepare the living room for exercise and put a water bottle next to the mat."
YOU ARE agent_3: light-duty mobile robot, cannot move heavy furniture.
Your HIDDEN INFO: "A heavy sofa blocks the only free floor spot in the living room."

Other robots' public offers (short):
- agent_1  has_items: water bottle | need_from_others: N1-1 (task) carry the water bottle to the living room
- agent_2  can_do: move heavy furniture
- agent_4  has_items: bath mat

{
  "reasoning": "1) The living room needs a free floor spot, a mat and a water bottle next to the mat. 2) My part: lay a mat on the free spot and place the water bottle. 3) The heavy sofa blocks the spot and my body cannot move it -> ASK_HELP, agent_2 can move heavy furniture. 4) I have no mat; agent_4 has a bath mat -> RECEIVE. 5) agent_1 needs someone to carry the water bottle (N1-1); I can move between rooms -> RECEIVE the bottle from agent_1 and place it myself.",
  "steps": [
    {"type": "ASK_HELP", "action": "Move the heavy sofa away from the free floor spot", "item": null, "target": "agent_2"},
    {"type": "RECEIVE",  "action": "Receive a mat", "item": "bath mat", "target": "agent_4"},
    {"type": "LOCAL",    "action": "Lay the mat on the free floor spot", "item": null, "target": null},
    {"type": "RECEIVE",  "action": "Receive the water bottle in the kitchen", "item": "water bottle", "target": "agent_1"},
    {"type": "LOCAL",    "action": "Place the water bottle next to the mat", "item": null, "target": null}
  ]
}
"""

LOCAL_PLAN_SYSTEM = f"""You are one robot in a team of heterogeneous robots in separate rooms.
Nobody assigns tasks. You write ONLY YOUR OWN plan.

You know: the TASK, your capability, your images, your HIDDEN INFO, your full OFFER,
and the PUBLIC offers of the other robots (capability, can_do, has_items, need_from_others).

{_EXAMPLE}

THINK FIRST (write it in "reasoning", step by step):
1) What does the TASK need overall?
2) What is MY part, given my room, my body and my HIDDEN INFO?
3) Which parts of MY part can my body not do, or need an item that is not in my room?
4) Looking at other robots' need_from_others: which can I do (HELP) or supply (PASS)?
5) In what order should MY steps happen?

STEP TYPES
- LOCAL:    I do this myself.
- ASK_HELP: a TASK my part needs but my body cannot do. Put it before the step that needs it.
- RECEIVE:  an ITEM my part needs that is not in my room. Put it before the step that uses it. Set "item".
- HELP:     a TASK another robot needs (its need_from_others) that I can do. Write the concrete task.
- PASS:     give one of MY has_items to a robot that needs it. Set "item".

"target" (optional): the robot you expect to work with.
  ASK_HELP -> a robot whose can_do covers it; RECEIVE -> a robot whose has_items lists it;
  HELP / PASS -> the robot you do it for. Use null if unsure. It is only a hint.

DO NOT
- plan things unrelated to the TASK
- request something you can do yourself, or an item that is in your room
- volunteer (HELP / PASS) for something your body cannot do or an item you do not have
- write steps for other robots

Return ONE JSON object: {{"reasoning": "...", "steps": [{{"type": ..., "action": ..., "item": ..., "target": ...}}]}}
An empty "steps" list is fine if you have no part in the task.
"""


# =============================================================================
# Formatting
# =============================================================================

def _own_offer_text(o: Offer) -> str:
    obs = "\n".join(f"- {x.object}" + (f" @ {x.location}" if x.location else "") + (f" ({x.state})" if x.state else "")
                    for x in o.obs_scope) or "- (none)"
    can = "\n".join(f"- {c.action} {c.object or ''}".rstrip() for c in o.can_do) or "- (none)"
    cannot = "\n".join(f"- {c.action} {c.object or ''}".rstrip() + (f": {c.reason}" if c.reason else "")
                       for c in o.cannot_do) or "- (none)"
    items = "\n".join(f"- {h.object}" for h in o.has_items) or "- (none)"
    needs = "\n".join(f"- {n.id} ({n.kind}) {n.what}" for n in o.need_from_others) or "- (none)"
    return (f"OBSERVED (private):\n{obs}\n\nCAN DO:\n{can}\n\nCANNOT DO (private):\n{cannot}\n\n"
            f"HAS ITEMS:\n{items}\n\nNEED FROM OTHERS:\n{needs}")


def _public_offer_text(aid: str, o: Offer) -> str:
    can = ", ".join(f"{c.action} {c.object or ''}".strip() for c in o.can_do) or "(none)"
    items = ", ".join(h.object for h in o.has_items) or "(none)"
    needs = "\n".join(f"    {n.id} ({n.kind}) {n.what}" for n in o.need_from_others) or "    (none)"
    return (f"[{aid}]\n  capability: {o.capability}\n  can_do: {can}\n  has_items: {items}\n"
            f"  need_from_others:\n{needs}")


def build_local_plan_user(inp: AgentInput, own: Offer, others: dict[str, Offer]) -> str:
    hidden = "\n".join(f"- {h}" for h in inp.hidden_info) or "- (none)"
    others_txt = "\n\n".join(_public_offer_text(a, o) for a, o in sorted(others.items())) or "(none)"
    return (
        f"TASK:\n{inp.task}\n\nYOU ARE: {own.agent}\n\nCAPABILITY:\n{inp.capability}\n\n"
        f"HIDDEN INFO:\n{hidden}\n\nYOUR OFFER:\n{_own_offer_text(own)}\n\n"
        f"OTHER ROBOTS' PUBLIC OFFERS:\n{others_txt}\n\nReturn JSON only."
    )


# =============================================================================
# Main
# =============================================================================

async def make_local_plan(agent: Agent, known_agents: set[str]) -> LocalPlan:
    assert agent.offer is not None, "make_offer() must run first"
    agent.receive()   # other robots' public offers

    def parse(raw: dict) -> LocalPlan:
        steps = raw.get("steps", [])
        if not isinstance(steps, list):
            raise ValueError("`steps` must be a list")
        for s in steps:
            s["type"] = str(s.get("type", "")).upper().strip()
            s["action"] = str(s.get("action", "")).strip()
            if s.get("target") in {"", "null", "None"}:
                s["target"] = None
            if s.get("item") in {"", "null", "None"}:
                s["item"] = None
        return LocalPlan.from_raw(agent.id, RawLocalPlan.model_validate(raw), known_agents)

    agent.plan = await agent.ask(
        "plan", LOCAL_PLAN_SYSTEM,
        build_local_plan_user(agent.inp, agent.offer, agent.others_offers),
        parse, banner_label="LOCAL PLAN RAW",
    )

    counts = {t: 0 for t in ("LOCAL", "ASK_HELP", "HELP", "RECEIVE", "PASS")}
    for s in agent.plan.steps:
        counts[s.type] += 1

    if agent.verbose:
        print(f"  [PLAN] {agent.id}: " + " ".join(f"{k}={v}" for k, v in counts.items()))
    agent.log.log("plan", agent.id, "plan_made", n_steps=len(agent.plan.steps), **{f"n_{k.lower()}": v for k, v in counts.items()})

    agent.bus.broadcast(agent.id, "plan", agent.plan.public(), phase="plan")
    return agent.plan
