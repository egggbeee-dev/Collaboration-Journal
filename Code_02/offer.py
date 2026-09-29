"""Stage 1 - OFFER (each robot, in parallel, one LLM call).

Each robot describes itself. Nobody is assigned anything here.

    capability        copied from input
    obs_scope         what I see / know                       (private)
    can_do            what I can do for the task
    cannot_do         task-relevant limits of my body          (private)
    has_items         movable objects I could give away
    need_from_others  what my part needs from other robots

Only the public part (capability, can_do, has_items, need_from_others) is broadcast.
"""

from __future__ import annotations

import re

from runtime import Agent
from schemas import AgentInput, HasItem, Offer, RawOffer, public_offer


# =============================================================================
# Movable-item filter (deterministic)
# =============================================================================

# A place or fixture is not movable when it is the head noun ("kitchen counter"),
# but "floor lamp" or "wall clock" are fine.
_PLACE_HEADS = {"sink", "counter", "shelf", "surface", "floor", "wall", "room", "space", "area", "table top"}
# Words that describe a state, not an object.
_STATE_WORDS = {"cleaned", "wiped", "organized", "tidied", "cleared", "arranged",
                "confirmation", "status", "done", "ready", "complete", "completed"}


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def is_movable(item: HasItem) -> bool:
    words = _words(item.object)
    if not words:
        return False
    if words[-1] in _PLACE_HEADS:
        return False
    return not (set(words) & _STATE_WORDS)


# =============================================================================
# Prompt
# =============================================================================

_EXAMPLE = """EXAMPLE

TASK: "Prepare the living room for exercise and put a water bottle next to the mat."
CAPABILITY: "Fixed-base arm in the kitchen. Cannot leave the kitchen."
HIDDEN INFO: "A water bottle is on the high kitchen shelf; only this arm can reach it."

{
  "reasoning": "1) The task needs free floor space, a mat and a water bottle in the living room. 2) In my kitchen, the water bottle on the high shelf is relevant; knives and bread are not. 3) I can pick the bottle and hand it over. 4) I cannot leave the kitchen, so I cannot bring it to the living room. 5) The bottle is movable, so another robot can receive it. 6) My part needs someone to carry the bottle to the living room.",
  "capability": "Fixed-base arm in the kitchen. Cannot leave the kitchen.",
  "obs_scope": [{"object": "water bottle", "location": "high kitchen shelf"}],
  "can_do": [{"action": "pick and hand over", "object": "water bottle"}],
  "cannot_do": [{"action": "leave the kitchen", "reason": "fixed base"}],
  "has_items": [{"object": "water bottle", "location": "kitchen"}],
  "need_from_others": [{"kind": "task", "what": "carry the water bottle from the kitchen to the living room"}]
}
"""

OFFER_SYSTEM = f"""You are one robot in a team of heterogeneous robots.
Each robot is in a different room and sees only its own images and HIDDEN INFO.
Nobody assigns tasks. You only describe yourself so the team can plan.

{_EXAMPLE}

THINK FIRST (write it in "reasoning", step by step):
1) What does the TASK require overall?
2) Which things in MY room / HIDDEN INFO are relevant to it? (ignore unrelated objects)
3) What can I do for the task with my body?
4) What task-relevant things can my body NOT do?
5) Which of my objects are movable and could be useful to others?
6) What does MY part need from other robots?
   - item: an object my part needs that is NOT in my room
   - task: something my part needs that my body cannot do
   A limitation alone is not a need. Only list what MY part depends on.

FIELDS
- capability: copy the given capability exactly.
- obs_scope: only what is in my images or HIDDEN INFO. Never invent objects.
- can_do: concrete, task-relevant actions I can do myself.
- cannot_do: task-relevant limits only.
- has_items: movable, task-relevant objects in my room (not places, not states).
- need_from_others: may be empty.

Return ONE JSON object with keys:
reasoning, capability, obs_scope, can_do, cannot_do, has_items, need_from_others
"""


def build_offer_user(inp: AgentInput) -> str:
    hidden = "\n".join(f"- {h}" for h in inp.hidden_info) or "- (none)"
    return (
        f"TASK:\n{inp.task}\n\n"
        f"CAPABILITY:\n{inp.capability}\n\n"
        f"HIDDEN INFO:\n{hidden}\n\n"
        "The attached images show your own room. Return JSON only."
    )


# =============================================================================
# Main
# =============================================================================

async def make_offer(agent: Agent) -> Offer:
    raw: RawOffer = await agent.ask(
        "offer", OFFER_SYSTEM, build_offer_user(agent.inp),
        RawOffer.model_validate, banner_label="OFFER RAW",
    )

    # drop non-movable "items"
    kept = [h for h in raw.has_items if is_movable(h)]
    dropped = [h.object for h in raw.has_items if not is_movable(h)]
    if dropped:
        agent.log.log("offer", agent.id, "has_items_filtered", dropped=dropped)
        if agent.verbose:
            print(f"  [OFFER FILTER] {agent.id}: not movable, dropped {dropped}")

    # code-assigned need ids: N<robot>-<k>
    num = agent.id.split("_")[-1]
    needs = [n.model_copy(update={"id": f"N{num}-{k}"}) for k, n in enumerate(raw.need_from_others, start=1)]

    agent.offer = Offer(agent=agent.id, **raw.model_dump(exclude={"has_items", "need_from_others"}),
                        has_items=kept, need_from_others=needs)

    if agent.verbose:
        o = agent.offer
        print(f"  [OFFER] {agent.id}: can_do={len(o.can_do)} cannot_do={len(o.cannot_do)} "
              f"has_items={[h.object for h in o.has_items]} "
              f"need_from_others={[f'{n.id} {n.what}' for n in o.need_from_others]}")

    agent.log.log("offer", agent.id, "offer_made",
                  n_items=len(agent.offer.has_items), n_needs=len(agent.offer.need_from_others))

    # only the public part leaves the robot
    agent.bus.broadcast(agent.id, "offer", public_offer(agent.offer), phase="offer")
    return agent.offer
