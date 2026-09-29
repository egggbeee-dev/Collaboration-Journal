"""
Stage 1 - OFFER

Each robot independently examines:
    - shared task
    - its own capability
    - its own private observations
    - hidden information

and generates a structured Offer.

The Offer contains:

    capability
    obs_scope
    can_do
    cannot_do
    can_provide
    needs

No robot is assigned a task at this stage.

The Offer is a structured communication interface used by
Local Planning and later Auction / Graph Reasoning stages.
"""

from __future__ import annotations

import re

from runtime import Agent
from schemas import AgentInput, Offer, RawOffer


# ===========================================================================
# Deterministic safety check
# ===========================================================================

NON_PASSABLE_KW = {
    "sink",
    "counter",
    "shelf",
    "surface",
    "floor",
    "wall",
    "room",
    "space",
    "area",
    "cleaned",
    "wiped",
    "organized",
    "tidied",
    "cleared",
    "arranged",
    "set",
    "setup",
    "confirmation",
    "confirm",
    "status",
    "done",
    "ready",
    "complete",
    "completed",
}


def _keywords(text: str) -> set[str]:
    return set(
        re.findall(
            r"[a-z0-9]+",
            text.lower(),
        )
    )


def _is_passable(item: str) -> bool:
    """
    A can_provide entry must refer to a physical,
    transferable object/resource.

    This is intentionally conservative.
    """

    return not bool(
        _keywords(item)
        & NON_PASSABLE_KW
    )


# ===========================================================================
# OFFER PROMPT
# ===========================================================================

_OFFER_EXAMPLE = """
EXAMPLE

Global task:
"Prepare the living room for exercise."

Robot capability:
"Mobile robot with a light-duty arm.
Can move between rooms but cannot move heavy furniture."

Possible structured Offer:

{
  "capability": "Mobile robot with a light-duty arm. Can move between rooms but cannot move heavy furniture.",

  "obs_scope": [
    {
      "object": "bath mat",
      "location": "bathroom",
      "state": "present"
    },
    {
      "object": "trash bin",
      "location": "bathroom",
      "state": "present"
    }
  ],

  "can_do": [
    {
      "action": "pick_up",
      "object": "bath mat",
      "location": "bathroom"
    }
  ],

  "cannot_do": [
    {
      "action": "move",
      "object": "heavy furniture",
      "location": "living room",
      "reason": "insufficient physical capability"
    }
  ],

  "can_provide": [
    {
      "type": "item",
      "object": "bath mat",
      "location": "bathroom"
    }
  ],

  "needs": [
    {
      "kind": "task",
      "action": "move",
      "object": "heavy furniture",
      "location": "living room"
    }
  ]
}

The NEED above is valid because the robot's own capability
prevents it from performing that required task.
"""


OFFER_SYSTEM = f"""
You are one robot in a team of heterogeneous robots.

Each robot operates in a spatially separated environment
and can observe only its own environment.

There is NO centralized task allocator at this stage.

Each robot independently determines:

1. what it can do,
2. what it cannot do,
3. what it can provide,
4. what it needs from other robots.

The purpose of the Offer is to expose local capabilities
and concrete collaboration dependencies.

{_OFFER_EXAMPLE}


============================================================
INPUT
============================================================

You receive:

- GLOBAL TASK
- ROBOT CAPABILITY
- YOUR PRIVATE IMAGES
- HIDDEN INFORMATION

Raw observations are private.

Do not infer facts about another robot's environment
unless they are explicitly contained in HIDDEN INFORMATION.


============================================================
OUTPUT
============================================================

Return ONLY one valid JSON object:

{{
  "capability": "...",
  "obs_scope": [...],
  "can_do": [...],
  "cannot_do": [...],
  "can_provide": [...],
  "needs": [...]
}}


============================================================
1. capability
============================================================

Copy the provided robot capability faithfully.

Do not add capabilities that are not explicitly given.


============================================================
2. obs_scope
============================================================

List concrete objects/facts that are:

- visible in the robot's private observations, OR
- explicitly provided through HIDDEN INFORMATION.

Each entry must have:

{{
  "object": "...",
  "location": "...",
  "state": "..."
}}

Do not invent objects or locations.


============================================================
3. can_do
============================================================

List concrete actions this robot can perform.

Each action must be represented as:

{{
  "action": "...",
  "object": "...",
  "location": "...",
  "target": "..."
}}

Rules:

- Must be physically supported by CAPABILITY.
- Must be grounded in OBS_SCOPE.
- Must directly contribute to GLOBAL TASK.
- Do not include actions requiring another robot.
- Do not invent objects.
- Do not assign work to another robot.


============================================================
4. cannot_do
============================================================

List task-relevant actions that this robot cannot perform
because of physical capability or environment constraints.

Format:

{{
  "action": "...",
  "object": "...",
  "location": "...",
  "reason": "..."
}}

Only include meaningful task-relevant limitations.


============================================================
5. can_provide
============================================================

List concrete things this robot can provide to another robot.

Format:

For an item:

{{
  "type": "item",
  "object": "...",
  "location": "..."
}}

For a task contribution:

{{
  "type": "task",
  "action": "...",
  "object": "...",
  "location": "..."
}}

A can_provide entry must represent something this robot
can actually provide.

Do not include abstract states such as:

- clean room
- cleared space
- confirmation
- completed task

Do not include places as transferable items.


============================================================
6. needs
============================================================

List concrete dependencies required from another robot.

For an item:

{{
  "kind": "item",
  "object": "...",
  "location": "..."
}}

For a task:

{{
  "kind": "task",
  "action": "...",
  "object": "...",
  "location": "..."
}}

Rules:

- A NEED must represent a real dependency.
- Do not assign a specific robot.
- Do not rewrite the entire GLOBAL TASK.
- Do not list something this robot can already do itself.
- A task-relevant NEED may overlap semantically with the GLOBAL TASK
  when it represents a capability-induced dependency.


============================================================
IMPORTANT
============================================================

1. Respect CAPABILITY.
2. Respect private observations.
3. Never invent objects.
4. Never assign final collaboration partners.
5. Never generate a global plan.
6. The Offer only describes local capability,
   possible contributions, and dependencies.
7. Return JSON only.
"""


# ===========================================================================
# USER PROMPT
# ===========================================================================

def build_offer_user(inp: AgentInput) -> str:

    hidden = (
        "\n".join(
            f"- {h}"
            for h in inp.hidden_info
        )
        or "- (none)"
    )

    return (
        f"TASK:\n{inp.task}\n\n"
        f"CAPABILITY:\n{inp.capability}\n\n"
        f"HIDDEN INFO:\n{hidden}\n\n"
        "The attached images show your own environment.\n"
        "Base OBS_SCOPE, CAN_DO and CAN_PROVIDE only on "
        "your own observations and HIDDEN INFO, filtered through "
        "your CAPABILITY.\n\n"
        "Determine concrete NEEDS required to accomplish the "
        "GLOBAL TASK that your robot cannot satisfy itself."
    )


# ===========================================================================
# OFFER GENERATION
# ===========================================================================

async def make_offer(agent: Agent) -> Offer:

    raw: RawOffer = await agent.ask(
        "offer",
        OFFER_SYSTEM,
        build_offer_user(agent.inp),
        RawOffer.model_validate,
        banner_label="OFFER RAW",
    )

    # --------------------------------------------------------
    # Deterministic can_provide safety filter
    # --------------------------------------------------------

    kept_provide = []
    dropped_provide = []

    for provide in raw.can_provide:

        # Only item entries are checked for physical passability.
        if provide.type == "item":

            text = " ".join(
                x
                for x in [
                    provide.object,
                    provide.location,
                ]
                if x
            )

            if not _is_passable(text):
                dropped_provide.append(
                    provide.model_dump()
                )
                continue

        kept_provide.append(provide)

    if dropped_provide:

        agent.log.log(
            "offer",
            agent.id,
            "can_provide_filtered",
            dropped=dropped_provide,
        )

        if agent.verbose:
            print(
                f"  [OFFER FILTER] {agent.id}: "
                f"non-passable can_provide dropped: "
                f"{dropped_provide}"
            )

    raw = raw.model_copy(
        update={
            "can_provide": kept_provide
        }
    )

    # --------------------------------------------------------
    # Create broadcast Offer
    # --------------------------------------------------------

    agent.offer = Offer(
        agent=agent.id,
        **raw.model_dump(),
    )

    if agent.verbose:

        print(
            f"  [OFFER] {agent.id}: "
            f"obs_scope={len(agent.offer.obs_scope)} "
            f"can_do={len(agent.offer.can_do)} "
            f"cannot_do={len(agent.offer.cannot_do)} "
            f"can_provide={len(agent.offer.can_provide)} "
            f"needs={len(agent.offer.needs)}"
        )

        print(
            f"  [OFFER DETAILS] {agent.id}"
        )

        print(
            f"    can_do: "
            f"{[x.model_dump() for x in agent.offer.can_do]}"
        )

        print(
            f"    needs: "
            f"{[x.model_dump() for x in agent.offer.needs]}"
        )

    agent.log.log(
        "offer",
        agent.id,
        "offer_made",
        n_needs=len(agent.offer.needs),
        n_provide=len(agent.offer.can_provide),
    )

    agent.bus.broadcast(
        agent.id,
        "offer",
        agent.offer.model_dump(),
        phase="offer",
    )

    return agent.offer
