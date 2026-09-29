"""Stage 2 - LOCAL PLANNING.

Each robot independently constructs its own local plan.

Step types:

    LOCAL
        Perform the action locally.

    ASK_HELP
        Ask another robot to perform a task.

    HELP
        Perform a task requested by another robot.

    RECEIVE
        Receive a physical item from another robot.

    PASS
        Pass a physical item to another robot.

The Local Plan does NOT finalize global coordination.
It exposes collaboration intents that are later processed by Auction.
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


# ---------------------------------------------------------------------------
# example
# ---------------------------------------------------------------------------

_LOCAL_PLAN_EXAMPLE = """
EXAMPLE

Global task:
"Prepare the living room for exercise."

Your capability:
"Mobile robot that can move heavy furniture."

Another robot's OFFER contains:

{
  "needs": [
    {
      "kind": "task",
      "action": "move heavy table",
      "location": "living room"
    }
  ]
}

and:

{
  "can_provide": [
    {
      "type": "item",
      "object": "yoga mat",
      "location": "bedroom"
    }
  ]
}

A valid local plan is:

{
  "steps": [
    {
      "type": "LOCAL",
      "action": "Move the chair aside"
    },
    {
      "type": "HELP",
      "action": "Move the heavy table",
      "kind": "task",
      "item": null,
      "target": "agent_1"
    },
    {
      "type": "PASS",
      "action": "Give the yoga mat to agent_1",
      "kind": "item",
      "item": "yoga mat",
      "target": "agent_1"
    }
  ]
}

If YOU cannot move the heavy table yourself:

{
  "type": "ASK_HELP",
  "action": "Move the heavy table",
  "kind": "task",
  "item": null,
  "target": null
}

If YOU need a yoga mat from another robot:

{
  "type": "RECEIVE",
  "action": "Receive the yoga mat",
  "kind": "item",
  "item": "yoga mat",
  "target": null
}
"""


# ---------------------------------------------------------------------------
# system prompt
# ---------------------------------------------------------------------------

LOCAL_PLAN_SYSTEM = f"""
You are one robot in a team of heterogeneous robots.

Each robot works in a different room and has access only to its own
observations.

Nobody assigns tasks to you.

You must construct YOUR OWN local plan based on:
- the shared TASK
- your CAPABILITY
- your own images
- HIDDEN INFO
- your own OFFER
- other robots' OFFERS

{_LOCAL_PLAN_EXAMPLE}

Return exactly ONE JSON object:

{{
  "steps": [
    {{
      "type": "LOCAL" | "ASK_HELP" | "HELP" | "RECEIVE" | "PASS",
      "action": "...",
      "kind": "task" | "item" | null,
      "item": "..." | null,
      "target": "agent_X" | null
    }}
  ]
}}

==================================================
STEP TYPES
==================================================

1. LOCAL

Perform an action yourself.

Use LOCAL when:
- you can physically perform the action
- the action is part of your own contribution
- no collaboration is required

Example:

{{
  "type": "LOCAL",
  "action": "Move the chair away from the exercise area",
  "kind": null,
  "item": null,
  "target": null
}}


2. ASK_HELP

Ask another robot to perform a TASK that you cannot perform.

Use ASK_HELP when:
- the task is necessary for your own contribution
- your capability prevents you from doing it
- another robot may potentially be able to do it

Example:

{{
  "type": "ASK_HELP",
  "action": "Move the heavy table",
  "kind": "task",
  "item": null,
  "target": null
}}

Do NOT assign a specific robot unless there is a strong reason.
The target is only a hint.


3. HELP

Perform a TASK that another robot needs.

Use HELP when:
- another robot's OFFER indicates a task dependency
- you can physically perform that task
- the task is something you can contribute to another robot

Example:

{{
  "type": "HELP",
  "action": "Move the heavy table",
  "kind": "task",
  "item": null,
  "target": "agent_1"
}}

IMPORTANT:

HELP is the opposite side of ASK_HELP.

ASK_HELP:
    "I need someone to do this."

HELP:
    "I will do this for you."


4. RECEIVE

Receive a PHYSICAL ITEM from another robot.

Use RECEIVE when:
- you need an item for your own contribution
- the item is not locally available
- another robot may possess or provide it

Example:

{{
  "type": "RECEIVE",
  "action": "Receive the yoga mat",
  "kind": "item",
  "item": "yoga mat",
  "target": null
}}

RECEIVE is NOT used for task-level assistance.


5. PASS

Pass a PHYSICAL ITEM to another robot.

Use PASS when:
- another robot needs an item
- you can provide that item
- the item is physically available to you

Example:

{{
  "type": "PASS",
  "action": "Give the yoga mat to agent_1",
  "kind": "item",
  "item": "yoga mat",
  "target": "agent_1"
}}

PASS is NOT used for task-level assistance.

==================================================
PAIRING RULES
==================================================

There are exactly two collaboration pairs:

    ASK_HELP ↔ HELP

    RECEIVE ↔ PASS


ASK_HELP and HELP represent TASK collaboration.

RECEIVE and PASS represent ITEM transfer.

Do NOT mix these meanings.

==================================================
IMPORTANT RULE - HELP VISIBILITY
==================================================

Before writing a LOCAL step, inspect the other robots' OFFERS.

If another robot needs a task that you can perform,
represent your contribution as HELP rather than LOCAL.

Why?

A LOCAL step is visible only inside your own local plan.

A HELP step explicitly exposes the collaboration relation
to the Auction.

Example:

Other robot:
    needs = "move heavy table"

You can move heavy furniture.

Correct:

    HELP:
        "Move the heavy table"

Incorrect:

    LOCAL:
        "Move the heavy table"

==================================================
IMPORTANT RULE - PASS VISIBILITY
==================================================

Before writing a LOCAL step involving an item that another robot
needs, check the other robots' needs.

If you are providing that item to another robot,
represent it as PASS rather than LOCAL.

Example:

Other robot needs:
    item = "yoga mat"

You possess:
    yoga mat

Correct:

    PASS:
        "Give the yoga mat to agent_2"

Do NOT represent this simply as LOCAL:
    "Pick up the yoga mat"

The transfer relationship must be explicitly exposed.

==================================================
IMPORTANT RULE - OWN NEEDS
==================================================

If your own OFFER contains:

kind = task

and you cannot perform that task:

    ASK_HELP

If your own OFFER contains:

kind = item

and you need that physical object:

    RECEIVE

Thus:

Offer.need(task)
    ↓
ASK_HELP

Offer.need(item)
    ↓
RECEIVE

==================================================
IMPORTANT RULE - CAPABILITY
==================================================

Only generate HELP when your own capability supports the requested task.

Only generate PASS when the item is physically available to you.

Never volunteer for an action that violates your capability.

==================================================
IMPORTANT RULE - NO GLOBAL REPLANNING
==================================================

You are NOT responsible for assigning the entire global task.

Do not rewrite another robot's plan.

Do not create steps for another robot.

Only create YOUR OWN steps.

==================================================
TARGET
==================================================

target is only a non-binding hint.

For:
    ASK_HELP
    RECEIVE

target can normally be null.

For:
    HELP
    PASS

target may contain the robot you expect to help.

The Auction will determine the actual collaboration edge.

==================================================
OUTPUT
==================================================

Return JSON only.
"""


# ---------------------------------------------------------------------------
# user prompt
# ---------------------------------------------------------------------------

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
            agent: offer.model_dump()
            for agent, offer in sorted(others.items())
        },
        ensure_ascii=False,
        indent=2,
    )

    return (
        f"TASK:\n{inp.task}\n\n"
        f"YOU ARE:\n{own_offer.agent}\n\n"
        f"CAPABILITY:\n{inp.capability}\n\n"
        f"HIDDEN INFO:\n{hidden}\n\n"
        f"YOUR OFFER:\n"
        f"{json.dumps(own_offer.model_dump(), ensure_ascii=False, indent=2)}\n\n"
        f"OTHER ROBOTS' OFFERS:\n"
        f"{others_txt}\n\n"
        "Construct only YOUR OWN local plan."
    )


# ---------------------------------------------------------------------------
# simple text overlap
# ---------------------------------------------------------------------------

def _token_set(text: str) -> set[str]:
    return set(
        re.findall(
            r"[a-z0-9]+",
            text.lower(),
        )
    )


def _overlap(a: str, b: str) -> float:
    aa = _token_set(a)
    bb = _token_set(b)

    if not aa or not bb:
        return 0.0

    return len(aa & bb) / max(
        1,
        min(len(aa), len(bb)),
    )


# ---------------------------------------------------------------------------
# consistency checks
# ---------------------------------------------------------------------------

def _plan_consistency_checks(
    agent: Agent,
    plan: LocalPlan,
) -> list[dict]:

    warnings: list[dict] = []

    own = agent.offer

    if own is None:
        return warnings

    # ---------------------------------------------------------------
    # collect other robots' needs
    # ---------------------------------------------------------------

    other_needs = []

    for aid, offer in agent.others_offers.items():
        for need in offer.needs:
            other_needs.append(
                (
                    aid,
                    need,
                )
            )

    # ---------------------------------------------------------------
    # check HELP
    # ---------------------------------------------------------------

    for step in plan.help_steps():

        candidates = []

        for aid, need in other_needs:

            if need.kind != "task":
                continue

            need_text = " ".join(
                x
                for x in [
                    need.action,
                    need.object,
                    need.location,
                ]
                if x
            )

            score = _overlap(
                step.action,
                need_text,
            )

            if (
                step.target == aid
                and score >= 0.20
            ):
                candidates.append(
                    (
                        aid,
                        need,
                        score,
                    )
                )

        if not candidates:

            all_candidates = []

            for aid, need in other_needs:

                if need.kind != "task":
                    continue

                need_text = " ".join(
                    x
                    for x in [
                        need.action,
                        need.object,
                        need.location,
                    ]
                    if x
                )

                all_candidates.append(
                    (
                        aid,
                        need,
                        _overlap(
                            step.action,
                            need_text,
                        ),
                    )
                )

            if all_candidates:

                aid, need, score = max(
                    all_candidates,
                    key=lambda x: x[2],
                )

                if score >= 0.35:
                    candidates.append(
                        (
                            aid,
                            need,
                            score,
                        )
                    )

        if not candidates:
            warnings.append(
                {
                    "step": step.id,
                    "issue": "help_without_matching_other_need",
                    "action": step.action,
                }
            )

        # Capability check.
        evidence = [
            x.action
            for x in own.can_do
        ]

        evidence.append(
            own.capability
        )

        best = max(
            (
                _overlap(
                    step.action,
                    x,
                )
                for x in evidence
            ),
            default=0.0,
        )

        if best < 0.15:
            warnings.append(
                {
                    "step": step.id,
                    "issue": "help_not_supported_by_declared_capability",
                    "action": step.action,
                }
            )

    # ---------------------------------------------------------------
    # check PASS
    # ---------------------------------------------------------------

    for step in plan.pass_steps():

        if not step.item:
            continue

        matching_receive = False

        for aid, offer in agent.others_offers.items():

            for need in offer.needs:

                if need.kind != "item":
                    continue

                if not need.object:
                    continue

                score = _overlap(
                    step.item,
                    need.object,
                )

                if (
                    score >= 0.50
                    and (
                        step.target is None
                        or step.target == aid
                    )
                ):
                    matching_receive = True
                    break

            if matching_receive:
                break

        if not matching_receive:
            warnings.append(
                {
                    "step": step.id,
                    "issue": "pass_without_matching_item_need",
                    "action": step.action,
                }
            )

        # Check whether the item is actually in own offer.
        own_items = [
            p.object
            for p in own.can_provide
            if p.type == "item"
            and p.object
        ]

        if not any(
            _overlap(step.item, x) >= 0.50
            for x in own_items
        ):
            warnings.append(
                {
                    "step": step.id,
                    "issue": "pass_item_not_in_can_provide",
                    "action": step.action,
                }
            )

    return warnings


# ---------------------------------------------------------------------------
# direction correction
# ---------------------------------------------------------------------------

_REQUEST_VERBS = re.compile(
    r"^(request|ask|need|please)\b",
    re.IGNORECASE,
)


def _fix_reversed_collaboration(
    plan: LocalPlan,
) -> list[str]:

    fixed: list[str] = []

    for step in plan.steps:

        action = step.action.strip()

        # HELP phrased as a request:
        #
        # "Ask agent_2 to move the table"
        #
        # should actually be ASK_HELP.
        if (
            step.type == "HELP"
            and _REQUEST_VERBS.match(action)
        ):
            step.type = "ASK_HELP"
            step.kind = "task"
            step.item = None
            step.target = None

            fixed.append(step.id)

        # PASS phrased as receiving/request:
        #
        # "Ask agent_2 for the cup"
        #
        # should actually be RECEIVE.
        elif (
            step.type == "PASS"
            and _REQUEST_VERBS.match(action)
        ):
            step.type = "RECEIVE"
            step.kind = "item"

            fixed.append(step.id)

    return fixed


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

async def make_local_plan(
    agent: Agent,
    known_agents: set[str],
) -> LocalPlan:

    assert (
        agent.offer is not None
    ), "make_offer() must run first"

    agent.receive()

    user = build_local_plan_user(
        agent.inp,
        agent.offer,
        agent.others_offers,
    )

    def parse(raw: dict) -> LocalPlan:

        return LocalPlan.from_raw(
            agent.id,
            RawLocalPlan.model_validate(raw),
            known_agents,
        )

    agent.plan = await agent.ask(
        "plan",
        LOCAL_PLAN_SYSTEM,
        user,
        parse,
        banner_label="LOCAL PLAN RAW",
    )

    # ---------------------------------------------------------------
    # direction correction
    # ---------------------------------------------------------------

    fixed_ids = _fix_reversed_collaboration(
        agent.plan
    )

    if fixed_ids:

        agent.log.log(
            "plan",
            agent.id,
            "reversed_collaboration_fixed",
            steps=fixed_ids,
        )

        if agent.verbose:
            print(
                f"  [PLAN FIX] {agent.id}: "
                f"reversed collaboration direction "
                f"fixed: {fixed_ids}"
            )

    # ---------------------------------------------------------------
    # consistency checks
    # ---------------------------------------------------------------

    consistency = _plan_consistency_checks(
        agent,
        agent.plan,
    )

    if consistency:

        agent.log.log(
            "plan",
            agent.id,
            "plan_consistency_warning",
            warnings=consistency,
        )

        if agent.verbose:

            for w in consistency:

                print(
                    f"  [PLAN CHECK] "
                    f"{agent.id} "
                    f"{w['issue']}: "
                    f"{w['step']} — "
                    f"{w['action']}"
                )

    # ---------------------------------------------------------------
    # statistics
    # ---------------------------------------------------------------

    n_local = sum(
        1
        for s in agent.plan.steps
        if s.type == "LOCAL"
    )

    n_ask_help = sum(
        1
        for s in agent.plan.steps
        if s.type == "ASK_HELP"
    )

    n_help = sum(
        1
        for s in agent.plan.steps
        if s.type == "HELP"
    )

    n_receive = sum(
        1
        for s in agent.plan.steps
        if s.type == "RECEIVE"
    )

    n_pass = sum(
        1
        for s in agent.plan.steps
        if s.type == "PASS"
    )

    if agent.verbose:

        print(
            f"  [PLAN] {agent.id}: "
            f"steps={len(agent.plan.steps)} "
            f"LOCAL={n_local} "
            f"ASK_HELP={n_ask_help} "
            f"HELP={n_help} "
            f"RECEIVE={n_receive} "
            f"PASS={n_pass}"
        )

    agent.log.log(
        "plan",
        agent.id,
        "plan_made",
        n_steps=len(agent.plan.steps),
        n_collab=len(
            agent.plan.collaboration_steps()
        ),
    )

    # ---------------------------------------------------------------
    # broadcast
    # ---------------------------------------------------------------

    agent.bus.broadcast(
        agent.id,
        "plan",
        agent.plan.model_dump(),
        phase="plan",
    )

    return agent.plan
