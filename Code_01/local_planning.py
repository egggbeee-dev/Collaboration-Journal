"""Stage 2 - LOCAL PLANNING.

Every robot plans for itself. It reads ALL broadcast Offers, creates LOCAL / HELP / PASS / RECEIVE intents, and may attach preferred targets as non-binding coordination hints.
A robot that can do something another robot asked for volunteers by writing a PASS/task
step into its OWN plan.

Prompt design notes (informed by the KCC 2-agent pilot, p2p_phase.py phase2_local_plan):
- The single most common failure mode observed in practice: a robot does something that
  another robot explicitly asked for (in that other robot's `needs`), but does it as a plain
  LOCAL step instead of a PASS step. A LOCAL step is invisible outside that robot's own plan,
  so the Auction has nothing to match the other robot's NEED against, and that NEED silently
  ends up unmatched even though the work actually got done. The prompt below calls this out
  explicitly and requires a visible PASS/task step whenever this happens - "doing it" is not
  enough, the robot must also *announce* it as PASS so it can be matched.
- Second most common failure mode: a robot volunteers for one need but decomposes its own
  execution into many small PASS steps (one per object it happens to move) instead of one
  PASS step for the need itself. Since Auction matching is 1:1, only one of those PASS steps
  can ever be used - the rest are silently dropped, and work that was actually planned
  disappears from the final joint plan without any error being raised.
- Third failure mode: a PASS step phrased as a request ("Request agent_2 to move the...")
  has the direction backwards - it should be a NEED. Left as PASS it can never validly match
  anything, so the real request silently disappears. `_fix_reversed_handoffs` catches this
  in code after parsing, since prompting alone did not reliably prevent it.
- Every rule is phrased in terms of CAPABILITY, not just "if you can" - because a plan that
  ignores capability limits produces steps the Graph's rule checks (or physical execution)
  would reject anyway.
"""
from __future__ import annotations

import json
import re

from runtime import Agent
from schemas import AgentInput, LocalPlan, Offer, RawLocalPlan

_LOCAL_PLAN_EXAMPLE = """EXAMPLE - bedroom robot. Other robots broadcast these needs:
- agent_1 needs (task): clear space in the living room
- agent_3 needs (task): move heavy furniture

Your capability allows moving furniture. A good local plan may be:
{
  "steps": [
    {"type": "LOCAL", "action": "Move the dining table and chairs to the side"},
    {"type": "PASS", "action": "Clear space in the living room by moving heavy furniture", "kind": "task", "item": null, "target": "agent_1"}
  ]
}

Here, target is a PREFERRED TARGET inferred from the broadcast offers. It is not a final assignment.
Auction may confirm this match, change it, or leave it unresolved."""

LOCAL_PLAN_SYSTEM = f"""You are one robot in a team of heterogeneous robots. Each robot plans for itself.

You receive the shared TASK, your own CAPABILITY, two IMAGES of your own room, HIDDEN INFO, your own OFFER, and the OFFERS broadcast by ALL other robots. Raw observations remain private: use only your own images and hidden information for physical facts about your room.

{_LOCAL_PLAN_EXAMPLE}

Generate YOUR OWN ordered local plan. Return ONE JSON object:
{{
  "steps": [
    {{
      "type": "LOCAL" | "NEED" | "PASS",
      "action": string,
      "kind": "item" | "task" | null,
      "item": string | null,
      "target": string | null
    }}
  ]
}}

SEMANTICS
- LOCAL: a task you perform yourself using your own capability and private observation.
- NEED + task: a collaboration request for a task you cannot perform. In the final rendering this is [HELP].
- NEED + item: you need to receive a physical item from another robot. In the final rendering this is [RECEIVE].
- PASS + task: you volunteer to perform a task explicitly needed by another robot, when your capability supports it.
- PASS + item: you offer to provide an item that another robot explicitly needs and that you can provide.

OFFER-AWARE PLANNING — follow these three questions:
1. What does the global TASK require from MY robot/room specifically?
2. What can I provide to OTHER robots? Check every other robot's OFFER `needs` against my capability, `can_do`, and `can_provide`. If I can satisfy a concrete need, create a visible PASS step rather than silently doing it as LOCAL.
3. What do I need from OTHER robots? Use my OFFER `needs` and the global task. If I cannot perform a required task, create NEED/task ([HELP]). If I need a physical object, create NEED/item ([RECEIVE]).

COLLABORATION RULES
1. A PASS may be created ONLY when there is a concrete matching need in another robot's broadcast OFFER. Do not invent a need.
2. A PASS must be supported by your declared CAPABILITY / `can_do` / `can_provide`. Never volunteer for a task your capability cannot perform.
3. One PASS represents one collaboration request/need, not one PASS per object or sub-action. If satisfying "clear space" requires moving a table, chairs, and sofa, keep these actions summarized inside ONE PASS/task rather than creating several competing PASSes for the same need.
4. A target is a PREFERRED TARGET, not a final allocation. You MAY set `target` to the robot whose OFFER explicitly contains the need you are addressing. If several robots have compatible needs, choose the most semantically compatible one as your preferred target; Auction will make the final assignment.
5. For HELP/NEED, you MAY set `target` to a preferred helper when another robot's OFFER clearly provides the required capability. This is only a preference/hint; Auction can change or reject it. Do not phrase the action as “Request agent_X...” or “Ask agent_X...”.
6. For RECEIVE/NEED-item, you MAY set a preferred provider when another robot's OFFER explicitly lists the needed item in `can_provide`. Again, this is not binding.
7. Do not choose a target merely because that robot exists. The preferred target must be justified by the broadcast OFFER compatibility.
8. If no suitable target is identifiable from the broadcast Offers, use `target: null`.
9. Do not write steps for another robot. PASS describes what YOU will provide; NEED describes what YOU require.
10. Use only objects visible in your own images or stated in your HIDDEN INFO. Never invent room objects.
11. Keep the plan concise and ordered by execution. Respect any time limit in the TASK.
12. Return JSON only."""

def build_local_plan_user(inp: AgentInput, own_offer: Offer, others: dict[str, Offer]) -> str:
    hidden = "\n".join(f"- {h}" for h in inp.hidden_info) or "- (none)"
    others_txt = json.dumps({a: o.model_dump() for a, o in sorted(others.items())}, ensure_ascii=False, indent=2)
    other_needs = [(a, n.kind, n.text) for a, o in sorted(others.items()) for n in o.needs]
    needs_reminder = (
        "\n".join(f"- {a} needs ({kind}): {text}" for a, kind, text in other_needs)
        if other_needs else "- (nobody has an outstanding need)"
    )
    return (
        f"TASK: {inp.task}\n\nYOU ARE: {own_offer.agent}\nCAPABILITY: {inp.capability}\n\n"
        f"HIDDEN INFO:\n{hidden}\n\nYOUR OFFER:\n{json.dumps(own_offer.model_dump(), ensure_ascii=False, indent=2)}\n\n"
        f"OTHER ROBOTS' OFFERS:\n{others_txt}\n\n"
        f"OTHER ROBOTS' OUTSTANDING NEEDS (check before writing any LOCAL step):\n{needs_reminder}\n\n"
        "The attached images show your own room."
    )



def _token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _overlap(a: str, b: str) -> float:
    aa, bb = _token_set(a), _token_set(b)
    if not aa or not bb:
        return 0.0
    return len(aa & bb) / max(1, min(len(aa), len(bb)))


def _plan_consistency_checks(agent: Agent, plan: LocalPlan) -> list[dict]:
    """Non-destructive checks between the broadcast Offer and Local Plan.

    These are warnings rather than hard filters: natural-language capability descriptions
    are too open-ended for a lexical rule to prove physical feasibility. The purpose is to
    expose LLM drift in experiments instead of silently changing the plan.
    """
    warnings: list[dict] = []
    own = agent.offer
    if own is None:
        return warnings

    other_needs = [(aid, n) for aid, o in agent.others_offers.items() for n in o.needs]
    for step in plan.steps:
        if step.type != "PASS" or step.kind != "task":
            continue

        matched_need = None
        for aid, need in other_needs:
            if step.target == aid and _overlap(step.action, need.text) >= 0.20:
                matched_need = (aid, need)
                break
        if matched_need is None:
            candidates = [(aid, n, _overlap(step.action, n.text)) for aid, n in other_needs]
            if candidates:
                aid, need, score = max(candidates, key=lambda x: x[2])
                if score >= 0.35:
                    matched_need = (aid, need)
        if matched_need is None and other_needs:
            warnings.append({"step": step.id, "issue": "pass_without_matching_other_need", "action": step.action})

        evidence = list(own.can_do) + [own.capability]
        best = max((_overlap(step.action, x) for x in evidence), default=0.0)
        if best < 0.15:
            warnings.append({"step": step.id, "issue": "pass_not_supported_by_declared_capability", "action": step.action})
    return warnings

_REQUEST_VERBS = re.compile(r"^(request|ask)\b", re.IGNORECASE)


def _fix_reversed_handoffs(plan: LocalPlan) -> list[str]:
    """A PASS step means 'I will do this for you'. If its action instead reads as a request
    ('Request agent_2 to move the...', 'Ask agent_3 to...'), the model has the direction
    backwards - it is actually a NEED, and left as PASS it can never validly match anything
    (nobody asked the sender to do it), so the real request silently disappears from the plan.
    Flip it in place; `kind`/`item` stay valid since a task-need needs exactly the same fields."""
    fixed = []
    for s in plan.steps:
        if s.type == "PASS" and _REQUEST_VERBS.match(s.action.strip()):
            s.type = "NEED"
            if not s.action.lstrip().upper().startswith("[HELP]"):
                s.action = "[HELP] " + s.action.strip()
            s.target = None
            fixed.append(s.id)
    return fixed


async def make_local_plan(agent: Agent, known_agents: set[str]) -> LocalPlan:
    assert agent.offer is not None, "make_offer() must run first"
    agent.receive()
    user = build_local_plan_user(agent.inp, agent.offer, agent.others_offers)

    def parse(raw: dict) -> LocalPlan:
        # HELP is a semantic tag over the existing NEED/task schema so this file remains
        # compatible with the current schemas.py. The downstream Auction therefore still
        # sees a normal task-NEED, while the plan text preserves the explicit [HELP] signal.
        for step in raw.get("steps", []):
            if str(step.get("type", "")).upper() == "HELP":
                step["type"] = "NEED"
                step["kind"] = "task"
                action = str(step.get("action", "")).strip()
                step["action"] = action if action.upper().startswith("[HELP]") else "[HELP] " + action
                step["target"] = None
        plan = LocalPlan.from_raw(agent.id, RawLocalPlan.model_validate(raw), known_agents)
        return plan

    agent.plan = await agent.ask("plan", LOCAL_PLAN_SYSTEM, user, parse, banner_label="LOCAL PLAN RAW")

    reversed_ids = _fix_reversed_handoffs(agent.plan)
    if reversed_ids:
        agent.log.log("plan", agent.id, "reversed_pass_fixed", steps=reversed_ids)
        if agent.verbose:
            print(f"  [PLAN FIX] {agent.id}: PASS phrased as a request, changed to NEED: {reversed_ids}")

    consistency = _plan_consistency_checks(agent, agent.plan)
    if consistency:
        agent.log.log("plan", agent.id, "plan_consistency_warning", warnings=consistency)
        if agent.verbose:
            for w in consistency:
                print(f"  [PLAN CHECK] {agent.id} {w['issue']}: {w['step']} — {w['action']}")

    n_pass = sum(1 for s in agent.plan.steps if s.type == "PASS")
    n_need = sum(1 for s in agent.plan.steps if s.type == "NEED")
    if agent.verbose:
        print(f"  [PLAN] {agent.id}: steps={len(agent.plan.steps)} LOCAL={len(agent.plan.steps) - n_pass - n_need} NEED={n_need} PASS={n_pass}")
    agent.log.log("plan", agent.id, "plan_made", n_steps=len(agent.plan.steps), n_collab=len(agent.plan.collaboration_steps()))
    agent.bus.broadcast(agent.id, "plan", agent.plan.model_dump(), phase="plan")
    return agent.plan
