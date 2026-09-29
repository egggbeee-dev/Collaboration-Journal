"""Stage 1 - OFFER.

Each robot independently describes:
- what it can do
- what it cannot do
- what it can provide
- what it needs

No robot is assigned a task at this stage.

The Offer is later used by Local Planning to construct:
    LOCAL
    ASK_HELP
    HELP
    RECEIVE
    PASS
"""

from __future__ import annotations

import re

from runtime import Agent
from schemas import (
    AgentInput,
    CanProvide,
    Offer,
    OfferNeed,
    RawOffer,
)


# ---------------------------------------------------------------------------
# physical passability check
# ---------------------------------------------------------------------------

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


def _is_passable(item: CanProvide) -> bool:
    """Check whether a can_provide item is physically transferable."""

    if item.type != "item":
        return True

    if not item.object:
        return False

    return not bool(
        _keywords(item.object) & NON_PASSABLE_KW
    )


# ---------------------------------------------------------------------------
# prompt
# ---------------------------------------------------------------------------

_OFFER_EXAMPLE = """
EXAMPLE

Global task:
"Prepare the living room for exercise."

Robot capability:
"Mobile robot with a light-duty arm. Can move between rooms but cannot move heavy furniture."

Observed room:
- bathtub
- sink
- toilet
- bath mat
- trash bin

Valid OFFER:

{
  "capability": "Mobile robot with a light-duty arm. Can move between rooms but cannot move heavy furniture.",

  "obs_scope": [
    {
      "object": "bath mat",
      "location": "bathroom"
    },
    {
      "object": "trash bin",
      "location": "bathroom"
    }
  ],

  "can_do": [
    {
      "action": "pick up the bath mat",
      "object": "bath mat",
      "location": "bathroom"
    }
  ],

  "cannot_do": [
    {
      "action": "move heavy furniture",
      "location": "living room",
      "reason": "insufficient payload capacity"
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
      "action": "move heavy furniture",
      "location": "living room"
    }
  ]
}
"""


OFFER_SYSTEM = f"""
You are one robot in a team of heterogeneous robots.

Each robot works in a different room and has access only to:
- its own images
- its own capability
- its own hidden information

There is NO central task allocator at this stage.

Your job is to independently construct an OFFER describing your
local capabilities, observations, possible contributions, and dependencies.

{_OFFER_EXAMPLE}

Return exactly ONE JSON object.

JSON format:

{{
  "capability": "...",

  "obs_scope": [
    {{
      "object": "...",
      "location": "...",
      "state": "..."
    }}
  ],

  "can_do": [
    {{
      "action": "...",
      "object": "...",
      "location": "...",
      "target": "..."
    }}
  ],

  "cannot_do": [
    {{
      "action": "...",
      "object": "...",
      "location": "...",
      "reason": "..."
    }}
  ],

  "can_provide": [
    {{
      "type": "item" | "task",
      "object": "...",
      "location": "...",
      "action": "..."
    }}
  ],

  "needs": [
    {{
      "kind": "item" | "task",
      "object": "...",
      "location": "...",
      "action": "..."
    }}
  ]
}}

RULES

1. CAPABILITY
- Copy the provided capability faithfully.
- Never claim abilities that contradict the capability.
- Respect mobility, payload, manipulation, and room-access constraints.

2. OBS_SCOPE
- Include only objects or areas visible in your images or explicitly stated
  in HIDDEN INFO.
- Never invent unseen objects.

3. CAN_DO
- List concrete actions that YOU can physically execute.
- Ground each action in OBS_SCOPE.
- The action must be allowed by your capability.
- Do not include actions that require another robot.
- Avoid vague phrases such as "help with the task".

4. CANNOT_DO
- List only task-relevant actions that your capability prevents you from doing.
- Explain the relevant limitation briefly.
- Do not list every imaginable inability.

5. CAN_PROVIDE

There are two possible forms.

ITEM:
- A physical object/resource that another robot could receive.
- It must actually exist in your OBS_SCOPE or HIDDEN INFO.
- It must be physically transferable.

TASK:
- A task you are physically capable of performing for another robot.
- This does NOT mean you are already assigned to perform it.
- It only describes a capability that can later become a HELP/PASS
  collaboration relation.

Never describe abstract states as transferable items.

6. NEEDS

There are two possible forms.

ITEM:
- A physical object you genuinely need to receive from another robot.

TASK:
- A concrete task that another robot must perform because you cannot
  perform it yourself.

A need must represent a genuine dependency of your own contribution.

Do NOT:
- restate the entire global task
- request help merely because another robot is capable of something
- assign a specific robot
- request something you can already do yourself

7. IMPORTANT

The OFFER does NOT decide collaboration.

It only exposes:
- capabilities
- observations
- possible resources
- dependencies

The actual collaboration relations are constructed later during
Local Planning and Auction.

8. OUTPUT
Return JSON only.
"""


# ---------------------------------------------------------------------------
# user prompt
# ---------------------------------------------------------------------------

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
        "The attached images show your own room.\n"
        "Base OBS_SCOPE, CAN_DO, and CAN_PROVIDE only on "
        "your own observations and HIDDEN INFO.\n"
        "For NEEDS, reason about genuine dependencies of "
        "your own contribution."
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

async def make_offer(agent: Agent) -> Offer:

    raw: RawOffer = await agent.ask(
        "offer",
        OFFER_SYSTEM,
        build_offer_user(agent.inp),
        RawOffer.model_validate,
        banner_label="OFFER RAW",
    )

    # ---------------------------------------------------------------
    # deterministic filtering of non-transferable items
    # ---------------------------------------------------------------

    kept_provide = [
        p
        for p in raw.can_provide
        if _is_passable(p)
    ]

    dropped_provide = [
        p
        for p in raw.can_provide
        if not _is_passable(p)
    ]

    if dropped_provide:
        agent.log.log(
            "offer",
            agent.id,
            "can_provide_filtered",
            dropped=[
                p.model_dump()
                for p in dropped_provide
            ],
        )

        if agent.verbose:
            print(
                f"  [OFFER FILTER] {agent.id}: "
                f"non-passable items dropped: "
                f"{[p.model_dump() for p in dropped_provide]}"
            )

    raw = raw.model_copy(
        update={
            "can_provide": kept_provide,
        }
    )

    # ---------------------------------------------------------------
    # attach agent ID
    # ---------------------------------------------------------------

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
