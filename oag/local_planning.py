"""Stage 2 - LOCAL PLANNING.

Each robot independently generates its own local plan using:
- the shared TASK
- its private observation (images, HIDDEN INFO, own full OFFER)
- the PUBLIC part of other robots' OFFERS

Local Planning writes only three step types:

    LOCAL     : I do this myself.
    ASK_HELP  : my embodiment cannot do a task that one of MY later
                steps depends on -> I request another robot to do it.
    RECEIVE   : one of MY later steps needs a physical item that is not
                in my room -> I request it from another robot.

HELP / PASS are NOT written here. Local plans are generated in parallel,
so a robot cannot see requests that only appear in other robots' plans.
Responses (HELP / PASS) are proposed afterwards in the Auction stage,
when every local plan has been broadcast.

`target` is only a preferred hint; `purpose` names the own later step
that the request enables.
"""

from __future__ import annotations

import re

from runtime import Agent
from schemas import AgentInput, LocalPlan, Offer, RawLocalPlan


PLAN_STEP_TYPES = {"LOCAL", "ASK_HELP", "RECEIVE"}
REQUEST_TYPES = {"ASK_HELP", "RECEIVE"}


# ---------------------------------------------------------------------------
# Prompt example
# ---------------------------------------------------------------------------

_LOCAL_PLAN_EXAMPLE = """EXAMPLE

Shared TASK:
"Prepare the living room for exercise."

You are agent_3:
"Light-duty mobile robot in the living room. Cannot move heavy furniture."

Your room: a heavy coffee table stands on the only free floor spot.

Other robots' public OFFERS:
- agent_2  CAN PROVIDE: type=task, object=heavy furniture, action=move
- agent_4  CAN PROVIDE: type=item, object=yoga mat, location=bedroom

VALID local plan:

{
  "steps": [
    {
      "type": "LOCAL",
      "action": "Clear the small items from the floor spot",
      "kind": null, "item": null, "target": null, "purpose": null
    },
    {
      "type": "ASK_HELP",
      "action": "Move the heavy coffee table away from the floor spot",
      "kind": "task", "item": null, "target": "agent_2",
      "purpose": "Lay the yoga mat on the floor spot"
    },
    {
      "type": "RECEIVE",
      "action": "Receive the yoga mat",
      "kind": "item", "item": "yoga mat", "target": "agent_4",
      "purpose": "Lay the yoga mat on the floor spot"
    },
    {
      "type": "LOCAL",
      "action": "Lay the yoga mat on the floor spot",
      "kind": null, "item": null, "target": null, "purpose": null
    }
  ]
}

Why this is valid:
- Each request is placed right before the own step it enables.
- `purpose` names that own step.
- The table is requested ONLY because it blocks THIS robot's own step.

INVALID (do NOT do this):

{
  "type": "ASK_HELP",
  "action": "Move heavy furniture in the living room",
  "purpose": null
}
as the last step of the plan.

Reason: nothing of yours depends on it. The global task needing it is NOT
a reason. Another robot that can move furniture will plan it itself.
"""


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

LOCAL_PLAN_SYSTEM = f"""You are one robot in a team of heterogeneous robots.

Each robot independently creates its OWN local plan.
Nobody assigns tasks to you.

You receive:
1. SHARED TASK
2. YOUR CAPABILITY
3. YOUR PRIVATE ROOM IMAGES
4. HIDDEN INFO
5. YOUR OFFER (full, private)
6. OTHER ROBOTS' OFFERS (public part only)

Your images, HIDDEN INFO, observed objects and cannot-do list are PRIVATE.
Never assume objects exist in another robot's room unless that robot's
public OFFER lists them in CAN PROVIDE.

{_LOCAL_PLAN_EXAMPLE}

==================================================
STEP TYPES (only these three)
==================================================

LOCAL      "I do this myself."
ASK_HELP   "I need another robot to do this TASK for me."
RECEIVE    "I need another robot to give me this ITEM."

Do NOT write HELP or PASS.
After all plans are shared, other robots will read your requests and
volunteer to respond. You only state what YOU do and what YOU need.

==================================================
REQUEST RULES (ASK_HELP / RECEIVE)  -- STRICT
==================================================

Write a request ONLY if ALL of these are true:

  [1] A LATER LOCAL step of YOUR OWN plan depends on it.
  [2] You cannot satisfy it yourself:
        ASK_HELP : your capability / embodiment cannot perform the task
                   (see your CANNOT DO and CAPABILITY)
        RECEIVE  : the item is not in your room / HIDDEN INFO
  [3] It is not merely "something the global task needs".

Form:
  - Put the request IMMEDIATELY BEFORE the own step it enables.
  - `purpose` = that own step's action text (required).
  - ASK_HELP : `action` = the concrete task, `kind` = "task", `item` = null
  - RECEIVE  : `item` = the physical object, `kind` = "item"

If nothing of yours is blocked, write NO request.
An empty plan is valid if you have no relevant contribution.

==================================================
LOCAL RULES
==================================================

- Only actions that directly contribute to the SHARED TASK.
- Only actions your capability allows. CANNOT DO entries are hard limits.
- Ground actions in your own observation / HIDDEN INFO.
- Do not create steps for other robots. Do not create WAIT steps.
- Keep the plan concise and execution-oriented.

==================================================
TARGET (hint only)
==================================================

`target` is a PREFERRED robot, never a final assignment.
- ASK_HELP : a robot whose CAPABILITY / CAN DO / CAN PROVIDE shows it can
             do the task, else null.
- RECEIVE  : a robot whose CAN PROVIDE lists the item, else null.
- LOCAL    : null.

==================================================
OUTPUT
==================================================

Return ONE JSON object and nothing else:

{{
  "steps": [
    {{
      "type": "LOCAL" | "ASK_HELP" | "RECEIVE",
      "action": "string",
      "kind": "task" | "item" | null,
      "item": "string" | null,
      "target": "agent_id" | null,
      "purpose": "string" | null
    }}
  ]
}}
"""


# ---------------------------------------------------------------------------
# Offer formatting
# ---------------------------------------------------------------------------

def _format_obs(offer: Offer) -> str:
    if not offer.obs_scope:
        return "- (none)"

    lines = []
    for o in offer.obs_scope:
        parts = [o.object]
        if o.location:
            parts.append(f"@ {o.location}")
        if o.state:
            parts.append(f"({o.state})")
        lines.append("- " + " ".join(parts))
    return "\n".join(lines)


def _format_cannot_do(offer: Offer) -> str:
    if not offer.cannot_do:
        return "- (none)"

    lines = []
    for c in offer.cannot_do:
        text = " ".join(x for x in (c.action, c.object) if x)
        lines.append(f"- {text}: {c.reason}" if c.reason else f"- {text}")
    return "\n".join(lines)


def _format_can_do(offer: Offer) -> str:
    if not offer.can_do:
        return "- (none)"

    return "\n".join(
        "- " + " ".join(x for x in (c.action, c.object) if x)
        for c in offer.can_do
    )


def _format_can_provide(offer: Offer) -> str:
    if not offer.can_provide:
        return "- (none)"

    lines = []
    for c in offer.can_provide:
        parts = [f"type={c.type}"]
        if c.object:
            parts.append(f"object={c.object}")
        if c.location:
            parts.append(f"location={c.location}")
        if c.action:
            parts.append(f"action={c.action}")
        lines.append("- " + ", ".join(parts))
    return "\n".join(lines)


def _format_needs(offer: Offer) -> str:
    return "\n".join(f"- ({n.kind}) {n.text}" for n in offer.needs) or "- (none)"


def build_local_plan_user(
    inp: AgentInput,
    own_offer: Offer,
    others: dict[str, Offer],
) -> str:

    hidden = (
        "\n".join(f"- {h}" for h in inp.hidden_info)
        if inp.hidden_info
        else "- (none)"
    )

    # Other robots: public fields only (obs_scope / cannot_do are not broadcast).
    other_blocks = [
        f"""[{agent_id}]
CAPABILITY:
{offer.capability}

CAN DO:
{_format_can_do(offer)}

CAN PROVIDE:
{_format_can_provide(offer)}

NEEDS:
{_format_needs(offer)}
"""
        for agent_id, offer in sorted(others.items())
    ]
    others_txt = "\n".join(other_blocks) or "- (none)"

    own_offer_txt = f"""CAPABILITY:
{own_offer.capability}

OBSERVED (private):
{_format_obs(own_offer)}

CAN DO:
{_format_can_do(own_offer)}

CANNOT DO (private, hard limits):
{_format_cannot_do(own_offer)}

CAN PROVIDE:
{_format_can_provide(own_offer)}

OWN NEEDS:
{_format_needs(own_offer)}
"""

    return f"""SHARED TASK:
{inp.task}

YOU ARE:
{own_offer.agent}

YOUR CAPABILITY:
{inp.capability}

HIDDEN INFO:
{hidden}

YOUR OFFER:
{own_offer_txt}

OTHER ROBOTS' OFFERS (public part):
{others_txt}

Remember:
- Use only LOCAL / ASK_HELP / RECEIVE.
- A request must be followed by the own LOCAL step it enables,
  and `purpose` must name that step.
- Return JSON only.
"""


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _overlap(a: str, b: str) -> float:
    aa = _token_set(a)
    bb = _token_set(b)

    if not aa or not bb:
        return 0.0

    return len(aa & bb) / max(1, min(len(aa), len(bb)))


# ---------------------------------------------------------------------------
# Structural validation (violations trigger an LLM retry)
# ---------------------------------------------------------------------------

def _validate_plan_structure(steps: list[dict]) -> None:
    """Hard rules that the LLM must satisfy; a ValueError triggers a retry."""

    for i, step in enumerate(steps):

        step_type = step["type"]

        if step_type in {"HELP", "PASS"}:
            raise ValueError(
                f"step {i + 1}: {step_type} is not allowed in Local Planning. "
                "Other robots respond to your requests later. "
                "Use only LOCAL / ASK_HELP / RECEIVE."
            )

        if step_type not in PLAN_STEP_TYPES:
            raise ValueError(
                f"step {i + 1}: invalid type {step_type!r}. "
                f"Allowed: {sorted(PLAN_STEP_TYPES)}"
            )

        if step_type in REQUEST_TYPES:

            if not (step.get("purpose") or "").strip():
                raise ValueError(
                    f"step {i + 1}: {step_type} requires `purpose` "
                    "(the own later step it enables)."
                )

            has_later_local = any(
                s["type"] == "LOCAL" for s in steps[i + 1:]
            )

            if not has_later_local:
                raise ValueError(
                    f"step {i + 1}: {step_type} {step['action']!r} is not "
                    "followed by any own LOCAL step that depends on it. "
                    "Remove it, or place it right before the step it enables."
                )


# ---------------------------------------------------------------------------
# Step normalization / salvage
# ---------------------------------------------------------------------------

def _normalize_step(step: dict) -> None:
    """Normalize one raw step in place; raise ValueError if unusable."""

    if not isinstance(step, dict):
        raise ValueError("each step must be a JSON object")

    step["type"] = str(step.get("type", "")).upper()
    step["action"] = str(step.get("action", "")).strip()

    if not step["action"]:
        raise ValueError(f"{step['type']} step requires an action.")

    if step.get("target") in {"", "null", "None"}:
        step["target"] = None

    if step.get("purpose") in {"", "null", "None"}:
        step["purpose"] = None

    if step["type"] == "ASK_HELP":
        step["kind"] = "task"
        step["item"] = None

    elif step["type"] == "RECEIVE":
        step["kind"] = "item"
        if not step.get("item"):
            raise ValueError("RECEIVE requires an `item` field.")


def _salvage_steps(raw: dict | None) -> tuple[list[dict], list[dict]]:
    """Fallback after the last retry: keep every valid step, drop only the
    ones that break the rules. Returns (kept, dropped-with-reason)."""

    steps = raw.get("steps", []) if isinstance(raw, dict) else []
    if not isinstance(steps, list):
        steps = []

    usable: list[dict] = []
    dropped: list[dict] = []

    for step in steps:
        try:
            _normalize_step(step)
        except ValueError as e:
            dropped.append({"step": str(step)[:200], "why": str(e)})
            continue

        if step["type"] not in PLAN_STEP_TYPES:
            dropped.append({"step": step["action"], "why": f"type {step['type']} not allowed"})
            continue

        usable.append(step)

    kept: list[dict] = []

    for i, step in enumerate(usable):

        if step["type"] in REQUEST_TYPES:

            later_local = next(
                (s for s in usable[i + 1:] if s["type"] == "LOCAL"),
                None,
            )

            if later_local is None:
                dropped.append({"step": step["action"], "why": "request with no own later step"})
                continue

            # missing purpose: use the next own LOCAL step
            if not step.get("purpose"):
                step["purpose"] = later_local["action"]

        kept.append(step)

    return kept, dropped


# ---------------------------------------------------------------------------
# Consistency checks (warnings only)
# ---------------------------------------------------------------------------

def _plan_consistency_checks(
    agent: Agent,
    plan: LocalPlan,
) -> list[dict]:

    warnings: list[dict] = []

    own = agent.offer

    if own is None:
        return warnings

    own_can_do = [
        " ".join(x for x in (c.action, c.object) if x)
        for c in own.can_do
    ]
    own_objects = [o.object for o in own.obs_scope]

    for step in plan.steps:

        # Asking for something the robot declared it can do itself.
        if step.type == "ASK_HELP":
            if any(_overlap(step.action, c) >= 0.8 for c in own_can_do):
                warnings.append({
                    "step": step.id,
                    "issue": "ask_help_for_own_can_do",
                    "action": step.action,
                })

        # Requesting an item that is already in the robot's own room.
        if step.type == "RECEIVE" and step.item:
            if any(_overlap(step.item, o) >= 0.8 for o in own_objects):
                warnings.append({
                    "step": step.id,
                    "issue": "receive_item_already_observed",
                    "action": step.action,
                })

    return warnings


# ---------------------------------------------------------------------------
# Main Local Planning
# ---------------------------------------------------------------------------

async def make_local_plan(
    agent: Agent,
    known_agents: set[str],
) -> LocalPlan:

    assert agent.offer is not None, (
        "make_offer() must run first"
    )

    agent.receive()

    user = build_local_plan_user(
        agent.inp,
        agent.offer,
        agent.others_offers,
    )

    # ---------------------------------------------------------------
    # Parse LLM JSON (strict) / salvage (fallback after last retry)
    # ---------------------------------------------------------------

    def parse(raw: dict) -> LocalPlan:

        steps = raw.get("steps", [])

        if not isinstance(steps, list):
            raise ValueError("`steps` must be a list")

        for step in steps:
            _normalize_step(step)

        _validate_plan_structure(steps)

        return LocalPlan.from_raw(
            agent.id,
            RawLocalPlan.model_validate(raw),
            known_agents,
        )

    def fallback(raw: dict | None) -> LocalPlan:
        kept, dropped = _salvage_steps(raw)

        agent.log.log(
            "plan",
            agent.id,
            "fallback_salvaged",
            kept=len(kept),
            dropped=dropped,
        )

        if agent.verbose:
            print(f"  [PLAN FALLBACK] {agent.id}: kept {len(kept)} steps, dropped {dropped}")

        return LocalPlan.from_raw(
            agent.id,
            RawLocalPlan.model_validate({"steps": kept}),
            known_agents,
        )

    # ---------------------------------------------------------------
    # LLM planning
    # ---------------------------------------------------------------

    agent.plan = await agent.ask(
        "plan",
        LOCAL_PLAN_SYSTEM,
        user,
        parse,
        banner_label="LOCAL PLAN RAW",
        fallback=fallback,
    )

    # ---------------------------------------------------------------
    # Consistency checks
    # ---------------------------------------------------------------

    consistency = _plan_consistency_checks(agent, agent.plan)

    if consistency:

        agent.log.log(
            "plan",
            agent.id,
            "plan_consistency_warning",
            warnings=consistency,
        )

        if agent.verbose:
            for warning in consistency:
                print(
                    f"  [PLAN CHECK] {agent.id} {warning['issue']}: "
                    f"{warning['step']} — {warning['action']}"
                )

    # ---------------------------------------------------------------
    # Statistics / logging
    # ---------------------------------------------------------------

    counts = {t: 0 for t in ("LOCAL", "ASK_HELP", "RECEIVE")}

    for step in agent.plan.steps:
        counts[step.type] += 1

    if agent.verbose:
        print(
            f"  [PLAN] {agent.id}: steps={len(agent.plan.steps)} "
            f"LOCAL={counts['LOCAL']} ASK_HELP={counts['ASK_HELP']} "
            f"RECEIVE={counts['RECEIVE']}"
        )

    agent.log.log(
        "plan",
        agent.id,
        "plan_made",
        n_steps=len(agent.plan.steps),
        n_local=counts["LOCAL"],
        n_ask_help=counts["ASK_HELP"],
        n_receive=counts["RECEIVE"],
    )

    # ---------------------------------------------------------------
    # Broadcast
    # ---------------------------------------------------------------

    agent.bus.broadcast(
        agent.id,
        "plan",
        agent.plan.model_dump(),
        phase="plan",
    )

    return agent.plan
