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

Write an OFFER: what YOU can contribute, which part of the task YOU intend to take, and what you need.
Be strict about your embodiment: list in can_do only actions your body can really perform.
"has_items" lists EVERY task-relevant movable object you can see in your room, even if you would
move it yourself, so that others know where things are.
"intends" is the part of the task you plan to take on yourself. Pick what fits your body and your
room best. Other robots declare their intentions at the same time, so keep it to what you are
clearly suited for.

Return JSON:
{{
 "reasoning": "short reasoning about the task, your room, and your body",
 "capability": "one sentence about your embodiment",
 "can_do": ["concrete action you can perform", ...],
 "cannot_do": ["task-relevant action you cannot perform, and why", ...],
 "has_items": ["task-relevant movable object visible in your room", ...],
 "intends": ["part of the task you plan to do", ...],
 "need_from_others": ["what you will need from other robots to do your intended part", ...],
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
PLAN_SYSTEM = """You are robot {agent}. Make YOUR OWN local plan for YOUR part of the shared task.
Nobody assigns you work, and you do not assign work to others.

Which part is yours:
- Start from your own "intends" in your offer.
- Do NOT plan parts that another robot intends to do. They will plan those themselves.
- If a part of the task appears in nobody's "intends" and you can do it, you may take it.

Step types you may use (only these three):
- LOCAL:    something you do yourself. Must cite "uses": the index of the can_do entry it relies on.
            "location" is the room where it happens (your own room unless you are mobile).
- ASK_HELP: another robot must DO something before one of YOUR OWN later LOCAL steps can happen.
            Requires "target", "action", and "enables".
- RECEIVE:  you need an OBJECT for one of YOUR OWN later LOCAL steps.
            Requires "target", "item" (copied exactly from the target's has_items), and "enables".

"enables" is the index (0-based, in your "steps" list) of YOUR OWN later LOCAL step that cannot
happen without this request. A request without such a step is not allowed: never ask other robots
to do parts of the task that your own steps do not depend on.

How objects move: an object changes rooms only when a mobile robot carries it. When you RECEIVE an
item, the giver puts it at its room's handoff spot and you (if you are mobile) go there to pick it
up. If you are NOT mobile, use ASK_HELP to ask a mobile robot to bring it to you instead.

"duration" is minutes, one of {durations}. Order the steps in the order you would execute them.

Return JSON:
{{
 "reasoning": "which part is yours, what it needs, and what you depend on",
 "steps": [
  {{"type": "RECEIVE", "item": "...", "target": "R1", "location": "kitchen", "enables": 1, "duration": 1}},
  {{"type": "LOCAL", "action": "...", "uses": 0, "location": "living room", "duration": 3}},
  {{"type": "ASK_HELP", "action": "...", "target": "R2", "location": "living room", "enables": 3, "duration": 5}},
  {{"type": "LOCAL", "action": "...", "uses": 1, "location": "living room", "duration": 2}}
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

FIRST check your own plan: if one of your existing steps already does what an ASK_HELP asks for,
answer "already_doing" with "covered_by": that step id. Do NOT add a second copy of work you
already do.

Decisions:
- "already_doing": your existing step (give "covered_by") already does it. Only for ASK_HELP.
- If the request's target is you: "accept" or "reject".
- If the request's target is another robot (or none): "volunteer" (offer to do it as a backup) or "ignore".
Only accept or volunteer if your can_do really covers it and it does not break your own part.
Do not volunteer for everything.

If you accept or volunteer, you add ONE step to your own plan:
- for ASK_HELP you will add a HELP step: give "action" and "uses" (index into your can_do).
- for RECEIVE you will add a PASS step: give "item", copied exactly from YOUR has_items.
  PASS means you put the item at your room's handoff spot; the requester picks it up.
- "insert_after": the id of the step in your plan after which you do it, or "START".
- "duration": minutes, one of {durations}.

Return JSON:
{{
 "judgments": [
  {{"request": "r4_s2", "decision": "already_doing", "covered_by": "r2_s1", "reason": "..."}},
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
     Remove a step that makes the plan inconsistent, e.g. one copy of DUPLICATE_WORK.
     Never a request step, never a step with a CONFIRMED edge.
- {{"op": "merge", "node": NODE_ID, "into": NODE_ID_OF_SAME_ROBOT, "why": "..."}}
     DUPLICATE_WORK inside ONE robot: the robot already does the same work in another step.
     The request served by "node" is then served by "into", and "node" is removed.
     The robot's agreement is kept; only the duplicate copy disappears.
- {{"op": "unresolved", "issue": ISSUE_ID, "why": "..."}}
     Leave the issue open when the choice is ambiguous.

For DUPLICATE_WORK: decide whether the two steps really do the same work. If yes, keep one
(prefer the step that has a CONFIRMED edge or is in the robot's own room) and drop or merge the other.
If they are different work, use "unresolved" with why "not duplicate".

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
