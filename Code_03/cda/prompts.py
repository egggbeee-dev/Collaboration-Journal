"""All prompts in one place. Every prompt asks for exactly one JSON object."""
from __future__ import annotations

import json

from .schemas import DURATIONS


def _j(x) -> str:
    return json.dumps(x, ensure_ascii=False, indent=1)


# ================================================================ Stage 1: Offer
OFFER_SYSTEM = """You are robot {agent}, one of several heterogeneous household robots.
Each robot sees only its own room. Nobody, including a central system, sees everything.
You cannot explore: reason only from your images, the text about areas you cannot see, and your body.

Write an OFFER describing what YOU can contribute and what YOU need.
Be strict about your embodiment: list in can_do only actions your body can really perform.

Return JSON:
{{
 "reasoning": "short reasoning about the task, your room, and your body",
 "capability": "one sentence about your embodiment",
 "can_do": ["concrete action you can perform", ...],
 "cannot_do": ["task-relevant action you cannot perform, and why", ...],
 "has_items": ["object in your room that is relevant to the task and that you can hand over", ...],
 "need_from_others": ["what you need from other robots for this task", ...],
 "obs_scope": "what you can and cannot observe"
}}"""

OFFER_USER = """TASK: {task}
DEADLINE: {deadline} minutes

YOUR BODY:
- room: {room}
- mobile (can leave its room): {mobile}
- payload: {payload} kg
- embodiment: {embodiment}

ABOUT AREAS YOU CANNOT SEE:
{instruction}

Your camera images are attached ({n_img} image(s))."""


# ================================================================ Stage 2: Local Plan
PLAN_SYSTEM = """You are robot {agent}. Make YOUR OWN local plan for the shared task.
Nobody assigns you work. Decide what you will do in your own room, and where you depend on others.

Step types you may use (only these three):
- LOCAL:    something you do yourself. Must cite "uses": the index of the can_do entry it relies on.
            "location" is where it happens ({room} unless you are mobile).
- ASK_HELP: you need another robot to DO something you cannot do. Requires "target" (a robot id)
            and "action" (what they should do). Choose the target from the public offers.
- RECEIVE:  you need an OBJECT from another robot. Requires "target" and "item".
            "item" MUST be copied exactly from the target's has_items.

Do NOT write PASS or HELP steps. Others will answer your requests themselves.
Only ask for help when your own can_do really does not cover it.
"duration" is minutes, one of {durations}.
Order the steps in the order you would execute them.

Return JSON:
{{
 "reasoning": "think step by step: what the task needs, what you can do, what you must request",
 "steps": [
  {{"type": "LOCAL", "action": "...", "uses": 0, "location": "{room}", "duration": 3}},
  {{"type": "ASK_HELP", "action": "...", "target": "R3", "location": "{room}", "duration": 5}},
  {{"type": "RECEIVE", "item": "...", "target": "R2", "location": "{room}", "duration": 1}}
 ]
}}"""

PLAN_USER = """TASK: {task}
DEADLINE: {deadline} minutes

YOUR OFFER (private parts included):
{own_offer}

YOUR can_do WITH INDICES (use these for "uses"):
{can_do_indexed}

PUBLIC OFFERS OF ALL ROBOTS:
{public_offers}

ROBOT PROFILES:
{profiles}"""

PLAN_FIX = """Your plan violated these rules:
{errors}
Fix ONLY these problems and return the full corrected JSON in the same format."""


# ================================================================ Stage 3: Edge Proposal
PROPOSE_SYSTEM = """You are robot {agent}. Other robots have published requests.
Judge each request yourself, using everything you know (including your private observations).

Decisions:
- If the request's target is you: "accept" or "reject".
- If the request's target is another robot (or none): "volunteer" (offer to do it as a backup) or "ignore".
Only accept or volunteer if your can_do really covers it. Do not volunteer for everything.

If you accept or volunteer, you add ONE step to your own plan:
- for ASK_HELP you will add a HELP step: give "action" and "uses" (index into your can_do).
- for RECEIVE you will add a PASS step: give "item", copied exactly from YOUR has_items.
- "insert_after": the id of the step in your plan after which you do it, or "START".
  Place it where it makes sense in your own schedule.
- "duration": minutes, one of {durations}.

Return JSON:
{{
 "judgments": [
  {{"request": "r1_s2", "decision": "accept", "reason": "...", "action": "...", "uses": 2,
    "insert_after": "r3_s1", "duration": 5}},
  {{"request": "r4_s3", "decision": "reject", "reason": "..."}},
  {{"request": "r2_s1", "decision": "volunteer", "reason": "...", "item": "...",
    "insert_after": "START", "duration": 1}}
 ]
}}"""

PROPOSE_USER = """TASK: {task}

YOUR OFFER (private parts included):
{own_offer}

YOUR can_do WITH INDICES:
{can_do_indexed}

YOUR CURRENT PLAN (ids you can use for insert_after):
{own_plan}

REQUESTS FROM OTHER ROBOTS:
{requests}"""


# ================================================================ Stage 4: Graph Reasoning
GRAPH_SYSTEM = """You are the graph integrator for a team of robots. You are NOT a planner and NOT an allocator.
You see only the structure of the joint plan and the issues found by code. You cannot see the robots'
cameras or private information. Every step was written by the robot that will execute it.

You may only use these operations, and only with ids that appear in the issue list:
- {{"op": "connect", "src": PROVIDER_ID, "dst": REQUEST_ID, "why": "..."}}
     Choose one candidate for a request. Only providers listed in that issue's "candidates".
     Choose only if exactly one candidate clearly fits the meaning; otherwise use "unresolved".
- {{"op": "disconnect", "src": PROVIDER_ID, "dst": REQUEST_ID, "why": "..."}}
     Remove a PROPOSED edge that is wrong (e.g. the item or action does not fit). Never a CONFIRMED edge.
- {{"op": "move", "node": PROVIDER_ID, "after": NODE_ID_OF_SAME_ROBOT_or_START, "why": "..."}}
     Change WHEN a robot performs a HELP/PASS step it already committed to (only to break a cycle).
- {{"op": "drop", "node": NODE_ID, "why": "..."}}
     Remove a step that makes the plan inconsistent. Never a request step, never a step with a CONFIRMED edge.
- {{"op": "unresolved", "issue": ISSUE_ID, "why": "..."}}
     Leave the issue open when the choice is ambiguous.

You can never create steps or decide new collaborations. Prefer the smallest change.
Return JSON: {{"ops": [ ... ]}}"""

GRAPH_USER = """TASK: {task}

ISSUES FOUND BY CODE:
{issues}"""


def plan_system(agent: str, room: str) -> str:
    return PLAN_SYSTEM.format(agent=agent, room=room, durations=list(DURATIONS))


def propose_system(agent: str) -> str:
    return PROPOSE_SYSTEM.format(agent=agent, durations=list(DURATIONS))


def indexed(xs: list[str]) -> str:
    return "\n".join(f"[{i}] {x}" for i, x in enumerate(xs)) or "(none)"


__all__ = ["_j", "OFFER_SYSTEM", "OFFER_USER", "PLAN_USER", "PLAN_FIX", "PROPOSE_USER",
           "GRAPH_SYSTEM", "GRAPH_USER", "plan_system", "propose_system", "indexed"]
