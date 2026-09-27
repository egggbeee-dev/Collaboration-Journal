"""Stage 1 - OFFER.

Every robot looks at its own images + capability + hidden info and broadcasts what it can
 do, cannot do, can provide and needs. Nobody assigns anything.

The OFFER is the robot's local negotiation interface:
- can_do: actions the robot can execute itself
- cannot_do: task-relevant actions it cannot execute
- can_provide: tangible resources it can physically hand over
- needs: concrete dependencies required to complete its own portion of the task

Important: a task-relevant NEED is allowed to overlap with the global task. Such overlap does
not make it an invalid "task restatement" when the robot cannot perform that required action
itself. The NEED expresses a capability-induced dependency that can become a P2P coordination
candidate in the Auction stage.
"""
from __future__ import annotations

from runtime import Agent
from schemas import AgentInput, Offer, RawOffer

# Words that make an item non-transportable even if the model calls it an "item".
# This post-hoc check is intentionally limited to can_provide: it prevents abstract states,
# places, or status descriptions from entering the P2P matching pool.
NON_PASSABLE_KW = {
    "sink", "counter", "shelf", "surface", "floor", "wall", "room", "space", "area",
    "cleaned", "wiped", "organized", "tidied", "cleared", "arranged", "set", "setup",
    "confirmation", "confirm", "status", "done", "ready", "complete", "completed",
}


def _keywords(text: str) -> set[str]:
    import re

    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _is_passable(item: str) -> bool:
    """A can_provide entry must name a physical object/resource, not a state or a place."""
    return not bool(_keywords(item) & NON_PASSABLE_KW)


_OFFER_EXAMPLE = """EXAMPLE - bathroom robot, global task "prepare the living room for exercise":
{
  "capability": "Mobile robot with a light-duty arm. Can move between rooms but cannot move heavy furniture.",
  "obs_scope": ["bathtub", "sink", "toilet", "bath mat", "trash bin"],
  "can_do": ["pick up the bath mat", "move the trash bin"],
  "cannot_do": ["move heavy furniture in the living room"],
  "can_provide": ["bath mat"],
  "needs": [
    {"kind": "task", "text": "move heavy furniture in the living room"}
  ]
}

IMPORTANT:
The NEED is intentionally similar to a part of the global task. It is valid because this robot
cannot perform that required action itself and its own completion depends on another robot doing it.
Do NOT remove such a NEED merely because its wording overlaps with the global task."""

OFFER_SYSTEM = f"""You are one robot in a team of robots. Each robot works in a different room and can only see its own room.
Nobody assigns work: there is no task allocator. You independently determine what you can do and what you need from other robots, based strictly on your own CAPABILITY and observations.

You receive: the shared TASK, your CAPABILITY, two IMAGES of your own room, and HIDDEN INFO (facts about places you cannot see or explore). You cannot inspect other rooms while constructing this OFFER.

{_OFFER_EXAMPLE}

Write your OFFER. It will be broadcast to the other robots and later used to generate local plans and P2P coordination candidates.
Return ONE JSON object with exactly these keys:
{{
  "capability": string,
  "obs_scope": [string],
  "can_do": [string],
  "cannot_do": [string],
  "can_provide": [string],
  "needs": [{{"kind": "item" | "task", "text": string}}]
}}

RULES:

1. CAPABILITY
- Copy your CAPABILITY faithfully.
- Do not claim abilities that contradict it.
- Treat mobility, payload/weight limits, manipulation ability, and room access as real constraints.

2. OBS_SCOPE
- List only objects and areas actually visible in your images or explicitly stated in HIDDEN INFO.
- Never invent unseen objects.
- Keep entries concrete and concise.

3. CAN_DO
- List up to 8 concrete actions YOU can physically execute.
- Every action must be grounded in an object or area from OBS_SCOPE.
- Every action must be allowed by your CAPABILITY.
- Prefer actions that directly contribute to the global task.
- Do not include actions that require another robot.
- Do not write vague actions such as "help with the task" or "prepare the room".

4. CANNOT_DO
- List only task-relevant actions that you cannot execute because of your physical capability,
  mobility, object access, or lack of a required object.
- Do not list every imaginable thing the robot cannot do.
- Keep each entry as a concrete action, e.g. "move heavy furniture in the living room".
- Do not put a task in CANNOT_DO merely because it is in another room; include it only when the
  task is relevant and your physical constraints actually prevent you from performing it.

5. CAN_PROVIDE
- List up to 2 tangible objects/resources that are physically available to you and that you could
  carry or hand over to another robot.
- The object must be grounded in OBS_SCOPE or HIDDEN INFO.
- Do NOT list states or results such as "cleaned sink", "cleared room", "organized shelf",
  "confirmation", or "status".
- Ask: "Could I physically bring this resource to another robot at a handoff point?"
  If not, leave it out.
- Only include resources that could plausibly help another robot with the global task.

6. NEEDS
A NEED is a concrete dependency required for YOUR OWN portion of the global task.
There are two kinds:
- kind="item": a tangible item you need to receive from another robot.
- kind="task": a concrete task that another robot needs to perform because YOU cannot perform it.

CRITICAL DISTINCTION:
- A NEED is NOT invalid just because its wording overlaps with the global TASK.
- If the global task requires an action and your capability prevents you from doing that action,
  represent that action as a NEED/task.
- Example:
    CANNOT_DO: "move heavy furniture in the living room"
    NEED: {{"kind":"task", "text":"move heavy furniture in the living room"}}
- This expresses a capability-induced dependency and is exactly the kind of dependency that may
  later be matched with another robot's PASS/task offer.
- Do NOT create a NEED merely because another robot could do something or because another room is
  involved. Your own completion must genuinely depend on it.
- Do NOT create a NEED for something you can already do yourself.
- Do NOT assign a specific robot in the NEED. The Auction decides who, if anyone, matches it.
- Keep each NEED short, concrete, and actionable.

7. COLLABORATION
Think about the boundary between your local plan and P2P coordination:
- What can I execute locally?
- What tangible resource can I hand over?
- What required action can I NOT execute?
- Which missing item or external task genuinely blocks my own completion?

8. CONSISTENCY
Before returning JSON, check:
- Every CAN_DO action is physically feasible.
- Every CAN_PROVIDE item is physically available and transportable.
- Every NEED is necessary for your own portion of the task.
- A NEED does not duplicate an action already in CAN_DO.
- CANNOT_DO reflects an actual limitation rather than a generic statement.
- A task-relevant NEED may overlap semantically with the global TASK; do not delete it for that reason.

9. OUTPUT
- Return ONLY valid JSON inside <JSON> tags.
- Do not include explanations outside the JSON.

<JSON>
{{
  "capability": "...",
  "obs_scope": ["..."],
  "can_do": ["..."],
  "cannot_do": ["..."],
  "can_provide": ["..."],
  "needs": [
    {{"kind": "item", "text": "..."}},
    {{"kind": "task", "text": "..."}}
  ]
}}
</JSON>"""


def build_offer_user(inp: AgentInput) -> str:
    hidden = "\n".join(f"- {h}" for h in inp.hidden_info) or "- (none)"
    return (
        f"TASK: {inp.task}\n\nCAPABILITY: {inp.capability}\n\nHIDDEN INFO:\n{hidden}\n\n"
        "The attached images show your own room. Base OBS_SCOPE, CAN_DO and CAN_PROVIDE only on "
        "what you see here plus HIDDEN INFO, filtered through what your CAPABILITY allows. "
        "For NEEDS, reason about what your own portion of the global task genuinely depends on."
    )


async def make_offer(agent: Agent) -> Offer:
    raw: RawOffer = await agent.ask(
        "offer", OFFER_SYSTEM, build_offer_user(agent.inp), RawOffer.model_validate, banner_label="OFFER RAW"
    )

    # Keep the deterministic physical-passability guard for can_provide.
    # Do NOT filter needs by lexical overlap with the global task: a capability-induced task
    # dependency can legitimately use nearly the same wording as the global task.
    kept = [p for p in raw.can_provide if _is_passable(p)]
    dropped = [p for p in raw.can_provide if not _is_passable(p)]
    if dropped:
        agent.log.log("offer", agent.id, "can_provide_filtered", dropped=dropped)
        if agent.verbose:
            print(f"  [OFFER FILTER] {agent.id}: non-passable can_provide dropped: {dropped}")

    raw = raw.model_copy(update={"can_provide": kept})

    agent.offer = Offer(agent=agent.id, **raw.model_dump())
    if agent.verbose:
        print(
            f"  [OFFER] {agent.id}: obs_scope={len(agent.offer.obs_scope)} can_do={len(agent.offer.can_do)} "
            f"can_provide={agent.offer.can_provide} needs={[n.text for n in agent.offer.needs]}"
        )
    agent.log.log("offer", agent.id, "offer_made", n_needs=len(agent.offer.needs), n_provide=len(agent.offer.can_provide))
    agent.bus.broadcast(agent.id, "offer", agent.offer.model_dump(), phase="offer")
    return agent.offer
