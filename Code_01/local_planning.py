"""Stage 2 - LOCAL PLANNING.

Each robot independently creates its own local plan.

The Local Plan is based on:
1. The shared global task
2. The robot's own capability
3. The robot's private observation
4. The robot's Offer
5. Other robots' Offers

IMPORTANT SEMANTICS

- LOCAL:
    Work this robot performs by itself.

- ASK_HELP:
    This robot genuinely needs another robot to perform a task
    for its own contribution.

- HELP:
    This robot can perform another robot's requested task.

- RECEIVE:
    This robot needs to receive a physical item from another robot.

- PASS:
    This robot offers/provides a physical item to another robot.

CRITICAL:
CANNOT_DO does NOT imply ASK_HELP.

A robot may be unable to perform many actions that are completely
irrelevant to its own contribution.

Therefore:
    CANNOT_DO != NEED
    CANNOT_DO != ASK_HELP

ASK_HELP / RECEIVE must be grounded in the robot's own OFFER.needs.
"""

from __future__ import annotations

import json
import re

from runtime import Agent
from schemas import (
    AgentInput,
    LocalPlan,
    Offer,
    RawLocalPlan,
)


# ============================================================================
# Example
# ============================================================================

_LOCAL_PLAN_EXAMPLE = """
EXAMPLE

Global task:
"Prepare the living room for exercise."

Robot R3:
- light-duty mobile robot
- located in bedroom
- can carry light objects
- cannot move heavy furniture

R3's OFFER:

{
  "can_do": [
    {
      "action": "pick up",
      "object": "ball",
      "location": "bedroom",
      "target": "move to living room"
    }
  ],
  "can_provide": [
    {
      "type": "item",
      "object": "ball",
      "location": "bedroom",
      "action": "move to living room"
    }
  ],
  "needs": []
}

Correct local plan:

{
  "steps": []
}

WHY?

R3 cannot move heavy furniture, but R3 does not need heavy
furniture to complete its own contribution.

Therefore:

    cannot move heavy furniture
            DOES NOT MEAN
    ask another robot to move heavy furniture

The following is WRONG:

{
  "type": "ASK_HELP",
  "action": "Move heavy furniture in the living room",
  "kind": "task",
  "item": null,
  "target": "agent_2"
}

because R3 has no corresponding NEED.

Also, simply because R3 can move a ball to the living room does
NOT mean R3 should do it.

The shared TASK determines whether moving the ball is actually
required.

If the task is only to clear/prep the living room and does not
require bringing the ball into the room, R3 should not invent
that action.

---

VALID HELP EXAMPLE

Suppose R1's OFFER contains:

{
  "needs": [
    {
      "kind": "task",
      "object": "kitchen items",
      "location": "living room",
      "action": "move"
    }
  ]
}

and R2 can actually move those items.

Then R2 may create:

{
  "type": "HELP",
  "action": "Move the kitchen items to the living room",
  "kind": "task",
  "item": null,
  "target": "agent_1"
}

HELP is valid because another robot explicitly has the
corresponding NEED.

---

VALID ASK_HELP EXAMPLE

Suppose R1's OFFER contains:

{
  "needs": [
    {
      "kind": "task",
      "object": "kitchen items",
      "location": "living room",
      "action": "move"
    }
  ]
}

and R1's own contribution actually requires those items in the
living room.

Then R1 may create:

{
  "type": "ASK_HELP",
  "action": "Move the kitchen items to the living room",
  "kind": "task",
  "item": null,
  "target": "agent_2"
}

because the ASK_HELP is grounded in R1's own OFFER.needs.
"""


# ============================================================================
# System Prompt
# ============================================================================

LOCAL_PLAN_SYSTEM = f"""
You are one robot in a team of heterogeneous robots.

Each robot independently creates a local plan for the shared
GLOBAL TASK.

You receive:

- GLOBAL TASK
- YOUR CAPABILITY
- YOUR PRIVATE ROOM OBSERVATION
- HIDDEN INFO
- YOUR OFFER
- OTHER ROBOTS' OFFERS

Your job is NOT to assign tasks globally.

Your job is to determine:

1. What THIS ROBOT should actually do.
2. Whether THIS ROBOT genuinely needs another robot.
3. Whether THIS ROBOT can help another robot with an explicitly
   declared need.

{_LOCAL_PLAN_EXAMPLE}

====================================================================
OUTPUT
====================================================================

Return exactly ONE JSON object:

{{
  "steps": [
    {{
      "type": "LOCAL" | "ASK_HELP" | "HELP" | "RECEIVE" | "PASS",
      "action": string,
      "kind": "item" | "task" | null,
      "item": string | null,
      "target": string | null
    }}
  ]
}}

====================================================================
STEP SEMANTICS
====================================================================

LOCAL
-----
A concrete action THIS ROBOT performs itself.

ASK_HELP
--------
A concrete task that THIS ROBOT genuinely requires another robot
to perform for THIS ROBOT'S OWN contribution.

HELP
----
A concrete task THIS ROBOT can perform for another robot because
that robot has explicitly declared a corresponding NEED.

RECEIVE
-------
A physical item THIS ROBOT genuinely needs to receive from
another robot for its own contribution.

PASS
----
A physical item THIS ROBOT can provide to another robot that
explicitly needs that item.

====================================================================
MOST IMPORTANT RULE: OFFER.GROUNDED COLLABORATION
====================================================================

ASK_HELP and RECEIVE MUST be grounded in YOUR OWN OFFER.needs.

Your Offer is the source of your collaboration requirements.

If:

    YOUR OFFER.needs = []

then:

    ASK_HELP = forbidden
    RECEIVE = forbidden

unless the required dependency is explicitly represented in
your Offer.

DO NOT invent a new need during Local Planning.

====================================================================
CANNOT_DO IS NOT A NEED
====================================================================

This rule is ABSOLUTE.

The fact that you cannot perform an action does NOT mean that
you should ask another robot to perform it.

For example:

Capability:
"I cannot move heavy furniture."

This alone does NOT justify:

ASK_HELP:
"Move heavy furniture."

Only create ASK_HELP if:

1. YOUR OWN contribution requires something,
2. that contribution is blocked by the missing task/item, AND
3. the corresponding dependency exists in YOUR OFFER.needs.

Therefore:

    CANNOT_DO != ASK_HELP

    CANNOT_DO != NEED

====================================================================
GLOBAL TASK FIRST
====================================================================

Always reason in this order:

STEP 1.
Determine exactly what the GLOBAL TASK requires.

STEP 2.
Determine whether your own room contains anything that can
meaningfully contribute to that task.

STEP 3.
Determine your own contribution.

STEP 4.
Check YOUR OFFER.needs.

STEP 5.
Only if a dependency in YOUR OFFER.needs is required for your
contribution may you create ASK_HELP or RECEIVE.

STEP 6.
Check OTHER ROBOTS' OFFER.needs.

STEP 7.
Only if another robot has an explicit relevant NEED may you
create HELP or PASS.

====================================================================
DO NOT INVENT WORK
====================================================================

Do NOT perform an action merely because:

- the object exists
- the object is movable
- the robot can manipulate it
- the object appears in can_do
- the object appears in can_provide
- another robot might want it
- the object is in another room
- the action would generally be useful
- the robot cannot perform some unrelated action

The GLOBAL TASK must actually require the action.

====================================================================
TARGET ROOM RULE
====================================================================

Do not move objects into the target room simply because they can
be moved there.

For example:

GLOBAL TASK:
"Prepare the living room for exercise."

Do NOT automatically move:

- ball
- cardboard box
- laptop
- bath mat
- random objects

into the living room.

Only move an object into the target room if the GLOBAL TASK
actually requires that object there or the object's movement
directly contributes to a required task.

Preparing/clearing a room does not mean importing unrelated
objects into that room.

====================================================================
LOCAL RULES
====================================================================

A LOCAL step must:

- directly contribute to the GLOBAL TASK
- be physically executable by THIS ROBOT
- use only objects known from THIS ROBOT'S observation
- not require another robot
- not be an unrelated action

If the robot has no meaningful contribution:

    return {{"steps": []}}

Do not invent a contribution merely because the robot has
objects in its room.

====================================================================
ASK_HELP RULES
====================================================================

ASK_HELP is allowed ONLY when ALL conditions are satisfied:

1. YOUR OFFER contains a corresponding NEED.
2. The NEED is relevant to the GLOBAL TASK.
3. The NEED is necessary for YOUR OWN contribution.
4. You cannot perform the required action yourself.
5. Another robot could plausibly perform it.

If any condition fails:

DO NOT create ASK_HELP.

Especially:

If YOUR OFFER.needs is empty:

    DO NOT create ASK_HELP.

====================================================================
RECEIVE RULES
====================================================================

RECEIVE is allowed ONLY when:

1. YOUR OFFER contains a corresponding item NEED.
2. Your own contribution actually requires that item.
3. Another robot's OFFER can plausibly provide it.

Do not create RECEIVE merely because another robot has a useful
item.

====================================================================
HELP RULES
====================================================================

HELP is allowed ONLY when another robot's OFFER explicitly
contains a corresponding NEED.

Do not infer needs from:

- cannot_do
- can_do
- can_provide
- capability

For example:

Robot R3 says:

    cannot move heavy furniture

This does NOT create a NEED for R3.

Therefore another robot must NOT create HELP:

    HELP: move heavy furniture

unless R3 explicitly has that NEED.

====================================================================
PASS RULES
====================================================================

PASS is allowed ONLY when:

1. Another robot explicitly needs the item.
2. This robot actually has that item.
3. The item is relevant to the GLOBAL TASK.
4. The item can physically be transferred.

Do not create PASS merely because another robot might find
the item useful.

====================================================================
OTHER ROBOTS' OFFERS
====================================================================

Other robots' OFFERS are coordination information.

Their:

    needs

are explicit requests.

Their:

    can_provide

are capabilities/resources.

Their:

    cannot_do

are NOT requests.

Never convert another robot's CANNOT_DO into HELP.

====================================================================
PRIVATE OBSERVATION
====================================================================

Only use objects that are:

- visible in your own images, OR
- explicitly stated in HIDDEN INFO, OR
- already explicitly represented in YOUR OFFER.

Never invent objects.

====================================================================
PLAN MINIMALITY
====================================================================

Prefer the smallest plan that correctly solves the robot's
meaningful contribution.

Do not add actions simply to make the plan longer.

If there is no valid action:

    return {{"steps": []}}

====================================================================
FINAL SELF-CHECK
====================================================================

Before returning the plan:

1. Does every LOCAL step directly contribute to GLOBAL TASK?

2. Does every ASK_HELP correspond to YOUR OFFER.needs?

3. If YOUR OFFER.needs is empty, are there ZERO ASK_HELP and
   ZERO RECEIVE steps?

4. Does every HELP correspond to another robot's explicit NEED?

5. Does every PASS correspond to another robot's explicit
   item NEED?

6. Did I accidentally convert CANNOT_DO into ASK_HELP?

7. Did I move an unrelated object into the target room?

8. Did I invent any object or task?

9. Can every action actually be performed with my capability?

If uncertain:

    OMIT the step.

Return JSON only.
"""


# ============================================================================
# User Prompt
# ============================================================================

def build_local_plan_user(
    inp: AgentInput,
    own_offer: Offer,
    others: dict[str, Offer],
) -> str:

    hidden = (
        "\n".join(
            f"- {h}"
            for h in inp.hidden_info
        )
        or "- (none)"
    )

    others_txt = json.dumps(
        {
            agent_id: offer.model_dump()
            for agent_id, offer in sorted(others.items())
        },
        ensure_ascii=False,
        indent=2,
    )

    own_needs = (
        "\n".join(
            f"- ({need.kind}) "
            f"object={need.object} "
            f"location={need.location} "
            f"action={need.action} "
            f"text={need.text}"
            for need in own_offer.needs
        )
        if own_offer.needs
        else "- NONE"
    )

    other_needs = []

    for agent_id, offer in sorted(others.items()):
        for need in offer.needs:
            other_needs.append(
                f"- {agent_id}: "
                f"({need.kind}) "
                f"object={need.object} "
                f"location={need.location} "
                f"action={need.action} "
                f"text={need.text}"
            )

    other_needs_txt = (
        "\n".join(other_needs)
        if other_needs
        else "- NONE"
    )

    return (
        f"GLOBAL TASK:\n"
        f"{inp.task}\n\n"

        f"YOU ARE:\n"
        f"{own_offer.agent}\n\n"

        f"YOUR CAPABILITY:\n"
        f"{inp.capability}\n\n"

        f"HIDDEN INFO:\n"
        f"{hidden}\n\n"

        f"YOUR OFFER:\n"
        f"{json.dumps(own_offer.model_dump(), ensure_ascii=False, indent=2)}\n\n"

        f"YOUR EXPLICIT NEEDS:\n"
        f"{own_needs}\n\n"

        f"OTHER ROBOTS' OFFERS:\n"
        f"{others_txt}\n\n"

        f"OTHER ROBOTS' EXPLICIT NEEDS:\n"
        f"{other_needs_txt}\n\n"

        "The attached images show your own room.\n\n"

        "IMPORTANT:\n"
        "Do not infer a collaboration request from CANNOT_DO.\n"
        "Use YOUR EXPLICIT NEEDS as the only basis for ASK_HELP "
        "and RECEIVE.\n"
        "If YOUR EXPLICIT NEEDS is NONE, do not create ASK_HELP "
        "or RECEIVE.\n"
        "Also do not move unrelated objects into the target room "
        "just because they are movable."
    )


# ============================================================================
# Utility
# ============================================================================

def _normalize_text(text: str) -> str:
    return re.sub(
        r"[^a-z0-9가-힣 ]+",
        " ",
        text.lower(),
    ).strip()


def _token_set(text: str) -> set[str]:
    return set(
        re.findall(
            r"[a-z0-9가-힣]+",
            _normalize_text(text),
        )
    )


def _similarity(a: str, b: str) -> float:
    aa = _token_set(a)
    bb = _token_set(b)

    if not aa or not bb:
        return 0.0

    return len(aa & bb) / max(
        1,
        min(len(aa), len(bb)),
    )


# ============================================================================
# Need matching
# ============================================================================

def _matches_need(
    action: str,
    kind: str | None,
    item: str | None,
    need,
) -> bool:

    action_text = _normalize_text(action)

    need_text_parts = [
        str(need.text or ""),
        str(need.action or ""),
        str(need.object or ""),
        str(need.location or ""),
    ]

    need_text = " ".join(
        x for x in need_text_parts if x
    )

    if kind != need.kind:
        return False

    if kind == "item":
        if item:
            if need.object:
                if _similarity(item, need.object) < 0.20:
                    return False

    score = _similarity(
        action_text,
        need_text,
    )

    return score >= 0.20


# ============================================================================
# Hard validation of collaboration steps
# ============================================================================

def _validate_collaboration_steps(
    agent: Agent,
    plan: LocalPlan,
) -> tuple[LocalPlan, list[str]]:

    """
    Deterministically remove hallucinated collaboration requests.

    IMPORTANT:
    - ASK_HELP must correspond to own OFFER.needs.
    - RECEIVE must correspond to own OFFER.needs.
    - HELP must correspond to another robot's OFFER.needs.
    - PASS must correspond to another robot's item NEED.

    This is intentionally strict.

    The Offer stage defines the robot's declared dependencies.
    Local Planning must not invent new dependencies.
    """

    if agent.offer is None:
        return plan, []

    own_needs = list(agent.offer.needs)

    other_needs = []

    for other_id, other_offer in agent.others_offers.items():
        for need in other_offer.needs:
            other_needs.append(
                (
                    other_id,
                    need,
                )
            )

    kept_steps = []
    removed = []

    for step in plan.steps:

        # ------------------------------------------------------------
        # ASK_HELP
        # ------------------------------------------------------------

        if step.type == "ASK_HELP":

            matched = any(
                _matches_need(
                    step.action,
                    step.kind,
                    step.item,
                    need,
                )
                for need in own_needs
                if need.kind == "task"
            )

            if not matched:
                removed.append(
                    f"{step.id}: ASK_HELP without matching own OFFER.need"
                )
                continue

        # ------------------------------------------------------------
        # RECEIVE
        # ------------------------------------------------------------

        if step.type == "RECEIVE":

            matched = any(
                _matches_need(
                    step.action,
                    step.kind,
                    step.item,
                    need,
                )
                for need in own_needs
                if need.kind == "item"
            )

            if not matched:
                removed.append(
                    f"{step.id}: RECEIVE without matching own OFFER.need"
                )
                continue

        # ------------------------------------------------------------
        # HELP
        # ------------------------------------------------------------

        if step.type == "HELP":

            matched = False

            for other_id, need in other_needs:

                if need.kind != "task":
                    continue

                if step.target is not None:
                    if step.target != other_id:
                        continue

                if _matches_need(
                    step.action,
                    step.kind,
                    step.item,
                    need,
                ):
                    matched = True
                    break

            if not matched:
                removed.append(
                    f"{step.id}: HELP without matching other OFFER.need"
                )
                continue

        # ------------------------------------------------------------
        # PASS
        # ------------------------------------------------------------

        if step.type == "PASS":

            matched = False

            for other_id, need in other_needs:

                if need.kind != "item":
                    continue

                if step.target is not None:
                    if step.target != other_id:
                        continue

                if _matches_need(
                    step.action,
                    step.kind,
                    step.item,
                    need,
                ):
                    matched = True
                    break

            if not matched:
                removed.append(
                    f"{step.id}: PASS without matching other item NEED"
                )
                continue

        kept_steps.append(step)

    plan.steps = kept_steps

    return plan, removed


# ============================================================================
# Main
# ============================================================================

async def make_local_plan(
    agent: Agent,
    known_agents: set[str],
) -> LocalPlan:

    assert (
        agent.offer is not None
    ), "make_offer() must run first"

    # Receive all broadcast offers/plans available at this stage.
    agent.receive()

    user = build_local_plan_user(
        agent.inp,
        agent.offer,
        agent.others_offers,
    )

    def parse(raw: dict) -> LocalPlan:

        # ------------------------------------------------------------
        # Normalize possible HELP/RECEIVE aliases if the model outputs
        # them despite the explicit schema.
        # ------------------------------------------------------------

        for step in raw.get("steps", []):

            step_type = str(
                step.get("type", "")
            ).upper()

            if step_type == "NEED":

                kind = step.get("kind")

                if kind == "task":
                    step["type"] = "ASK_HELP"

                elif kind == "item":
                    step["type"] = "RECEIVE"

            # Keep explicit five-type semantics.
            if step_type == "HELP":
                step["type"] = "HELP"

            if step_type == "PASS":
                step["type"] = "PASS"

            if step_type == "ASK_HELP":
                step["kind"] = "task"
                step["item"] = None

            elif step_type == "RECEIVE":
                step["kind"] = "item"

            elif step_type == "HELP":
                step["kind"] = "task"
                step["item"] = None

        return LocalPlan.from_raw(
            agent.id,
            RawLocalPlan.model_validate(raw),
            known_agents,
        )

    # ------------------------------------------------------------
    # LLM planning
    # ------------------------------------------------------------

    agent.plan = await agent.ask(
        "plan",
        LOCAL_PLAN_SYSTEM,
        user,
        parse,
        banner_label="LOCAL PLAN RAW",
    )

    # ------------------------------------------------------------
    # Hard collaboration validation
    # ------------------------------------------------------------

    agent.plan, removed = _validate_collaboration_steps(
        agent,
        agent.plan,
    )

    if removed:

        agent.log.log(
            "plan",
            agent.id,
            "invalid_collaboration_removed",
            removed=removed,
        )

        if agent.verbose:

            for msg in removed:

                print(
                    f"  [PLAN FILTER] "
                    f"{agent.id}: {msg}"
                )

    # ------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------

    counts = {
        "LOCAL": 0,
        "ASK_HELP": 0,
        "HELP": 0,
        "RECEIVE": 0,
        "PASS": 0,
    }

    for step in agent.plan.steps:

        if step.type in counts:
            counts[step.type] += 1

    if agent.verbose:

        print(
            f"  [PLAN] {agent.id}: "
            f"steps={len(agent.plan.steps)} "
            f"LOCAL={counts['LOCAL']} "
            f"ASK_HELP={counts['ASK_HELP']} "
            f"HELP={counts['HELP']} "
            f"RECEIVE={counts['RECEIVE']} "
            f"PASS={counts['PASS']}"
        )

    # ------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------

    agent.log.log(
        "plan",
        agent.id,
        "plan_made",
        n_steps=len(agent.plan.steps),
        n_collab=len(
            agent.plan.collaboration_steps()
        ),
        counts=counts,
    )

    # ------------------------------------------------------------
    # Broadcast
    # ------------------------------------------------------------

    agent.bus.broadcast(
        agent.id,
        "plan",
        agent.plan.model_dump(),
        phase="plan",
    )

    return agent.plan
