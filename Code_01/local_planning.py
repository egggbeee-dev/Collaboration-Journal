"""Stage 2 - LOCAL PLANNING.

Every robot plans for itself (no task allocation). It reads the other robots' broadcast
Offers, marks the steps that need collaboration (NEED / HELP / PASS) and broadcasts its plan.
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

_LOCAL_PLAN_EXAMPLE = """EXAMPLE - bedroom robot (mobile, heavy-duty), other robot's OFFER has needs=[{"kind":"task","text":"Have the sofa moved aside"}]:
{
  "steps": [
    {"type": "LOCAL", "action": "Take the yoga mat and dumbbells from the closet"},
    {"type": "PASS", "action": "Go to the living room and move the sofa aside", "kind": "task", "item": null, "target": "agent_1"},
    {"type": "PASS", "action": "Hand the yoga mat over", "kind": "item", "item": "yoga mat", "target": "agent_1"}
  ]
}
Notice: moving the sofa is written as PASS/task with target set to whoever asked for it -
NOT as a LOCAL step - because another robot's OFFER explicitly needed it done."""

LOCAL_PLAN_SYSTEM = f"""You are one robot in a team of robots. Each robot works in a different room and can only see its own room. Nobody assigns work: you plan for yourself, strictly within what your own CAPABILITY allows.

You receive: the shared TASK, your own CAPABILITY, two IMAGES of your own room, HIDDEN INFO, your own OFFER, and the OFFERS that all other robots broadcast.

{_LOCAL_PLAN_EXAMPLE}

Write YOUR OWN plan as an ordered list of steps, in the order you would execute them. Return ONE JSON object:
{{
  "steps": [
    {{
      "type": "LOCAL" | "NEED" | "PASS",
      "action": string,                 // one natural-language sentence
      "kind": "item" | "task" | null,   // required for NEED and PASS, null for LOCAL
      "item": string | null,            // required when kind is "item"
      "target": string | null           // optional hint: id of another robot, e.g. "agent_3"
    }}
  ]
}}

Step types:
- LOCAL: something you do alone, for your own room, that nobody else asked for.
- NEED + item: you must receive an item you do not have.
- NEED + item: you must receive an item you do not have.
- HELP + task (encoded as NEED/task with a [HELP] tag in the action): something you cannot do (see your CAPABILITY) and want another robot to do for you.
- PASS + item: you hand an item you have to another robot who NEEDS it (see the other offers).
- PASS + task: you volunteer to do a task that another robot's OFFER lists in its `needs` and that your CAPABILITY allows. PASS is the provider-side response to another robot's HELP/NEED.

CRITICAL RULE - visibility of help you give:
  Before writing a LOCAL step, check every other robot's OFFER `needs` list. If what you are about
  to do matches one of THEIR needs (same item, or same task), you must write it as PASS instead of
  LOCAL, with `target` set to that robot. If YOU cannot perform a required task because of your
  CAPABILITY, write it as NEED/task with action starting `[HELP]` and leave `target` null; the Auction
  will determine who can help you. A LOCAL step is only visible inside your own plan - if you
  quietly do someone else's requested task as LOCAL, nobody else can see it happened, and their NEED
  is left looking unmatched even though the work was done. Doing the work is not enough; you must
  also announce it as PASS so it can be matched to their NEED.
  Only volunteer (PASS) for something your CAPABILITY actually allows - do not offer to move heavy
  furniture if your capability says you cannot lift heavy objects, for example.

CRITICAL RULE - one PASS per need, not one per object:
  If one robot's need is a single task (e.g. "clear space", "move heavy furniture") and satisfying
  it takes several of your own actions (moving a table, then chairs, then a sofa), write ONE PASS/task
  step for that need - not one PASS step per object. Put the individual actions in your own LOCAL
  steps (or in the one PASS step's `action` text, summarized), and let the single PASS step be what
  answers that robot's need, with `target` set to them. Matching is one-to-one: a need can only be
  answered by one PASS step, so splitting your work into many small PASS steps for the same need
  means only one of them can ever match and the rest are silently dropped - the work looks like it
  never happened even though you planned to do it.

Other rules:
- Plan only what you can do with your own capability. If the task needs something you cannot do because of your CAPABILITY, add a NEED/task step whose
action starts with `[HELP]`. Do not write `Request agent_X...` or `Ask agent_X...`, and do not choose
the helper in Local Planning. The helper is selected later by Auction.
- Use only objects visible in your images or stated in your HIDDEN INFO. Never invent objects.
- A NEED must name a specific item or task, not the shared TASK restated wholesale ("yoga mat" is fine; "set up the living room" is not - that is the task itself, not a specific need).
- For `[HELP]` steps, `target` MUST be null because Local Planning must not allocate the task.
- For PASS steps, `target` may identify the robot whose need you are satisfying; this is a coordination hint,
  not a new allocation mechanism.
- Respect any time limit in the TASK: keep the plan short and put NEED steps late enough that the other robot can prepare, and PASS steps early enough.
- Do not write steps for other robots. Return JSON only."""


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
        for step in plan.steps:
            if step.type == "NEED" and step.kind == "task" and step.action.strip().upper().startswith("[HELP]"):
                step.target = None
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
