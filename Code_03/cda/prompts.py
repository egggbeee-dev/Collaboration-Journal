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

Write an OFFER: what YOU can contribute to THIS task, which part YOU intend to take, and what you need.

Think in this order and write it in "reasoning":
 1. GOAL STATE: when this task is done, what must be true in the house?
 2. PRIORITY: which of those matter most (the task's stated priorities, the people it is for, the deadline)?
 3. MY CONTRIBUTION: which of the important ones happen in MY room?
 4. MY OBJECTS: which objects in my room could help the task in another room (e.g. where the task
    mainly happens)? You cannot bring them yourself, but you can suggest them.
Only then fill the fields.

- "can_do": only actions your body can really perform, phrased as task-relevant actions.
- "has_items": EVERY movable object you can see in your room. Do NOT filter by relevance: other robots
  decide what they need. This list is the only way they learn what exists in your room.
- "suggests": objects from your room that would make the task's result better, each with what it would
  be used for and where (e.g. "Towel: wiping sweat after the workout in the living room"). Think
  generously: what would a thoughtful person bring from your room?
- "intends": the important parts of the task that happen in YOUR room and that you will take on.
  It may be EMPTY if nothing important happens in your room. Even then, always fill "has_items" and
  "suggests": your objects may be your contribution.

TEAM GOAL: the robots are a team. A well-prepared result is better than a minimal one. Do not invent
work unrelated to the task, but DO use other rooms' objects and other robots' help whenever they
make the result better or faster (e.g. water and a towel for a workout).

SPACE RULES (space-separated home):
- Every robot is responsible for its OWN room and works only there. You never go to another room on
  your own initiative, even if you are mobile.
- Objects from another room: write RECEIVE to that room's robot. Objects are handed over at the room
  doors; how they travel is not your concern.
- Work in your room that your body cannot do (out of reach, too heavy, you are fixed in place): write
  ASK_HELP to a robot that can, and say exactly what it should do. A mobile robot leaves its room ONLY
  when it accepts such a request.

Return JSON:
{{
 "reasoning": "1. goal state ... 2. priority ... 3. my contribution ... 4. my objects ...",
 "capability": "one sentence about your embodiment",
 "can_do": ["concrete action you can perform", ...],
 "cannot_do": ["task-relevant action you cannot perform, and why", ...],
 "has_items": ["every movable object visible in your room", ...],
 "suggests": ["Object: what it could be used for in this task", ...],
 "intends": ["important part of the task you will do", ...],
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

TEAM GOAL: the robots are a team. A well-prepared result is better than a minimal one. Do not invent
work unrelated to the task, but DO use other rooms' objects and other robots' help whenever they
make the result better or faster (e.g. water and a towel for a workout).

STEP 1 - GOALS. Write "goals": what must be true when the task is done, each with a priority
(high / medium / low) taken from the task itself. Only goals of the TASK, not general housekeeping.

STEP 2 - YOUR PART. Your part is the work in YOUR room (start from your "intends").

STEP 3 - FROM OTHER ROOMS. Go through EVERY other robot's "suggests" one by one. For each object that
would make the result in your room better, write a RECEIVE followed by your own LOCAL step that uses it
(e.g. RECEIVE Towel, then "move the Towel to the edge of the workout area"). Do not skip a useful
object just because you could finish without it.

STEP 4 - YOUR OBJECTS FOR OTHERS. For each of YOUR "suggests" that the room where the task mainly
happens would use, write a PASS offer to that room's robot.

STEP 5 - TIME. Add up your steps. If your own room's work takes more than about 60% of the deadline,
ask a mobile robot to do specific parts of it in parallel (e.g. "move the two light chairs to the
wall"): write ASK_HELP with "basis": "time" (no "enables" needed) and do NOT also plan that part
yourself. If nobody helps, you will do it yourself.

If nothing needs you in your room, you need nothing, and you have nothing to offer, return an EMPTY
"steps" list.

STEP 6 - STEPS. Every step must serve one of YOUR goals: "serves" is the index of that goal.
Do high-priority goals first. Use only these step types:
- LOCAL:    something you do yourself, in your own room. "uses": index of the can_do entry it relies on.
- ASK_HELP: another robot must DO something in your room. Either "basis": "dependency" (it must happen
            before one of YOUR later LOCAL steps; requires "enables") or "basis": "time" (parallel
            help with part of your work, see STEP 5). Requires "target", "action".
- RECEIVE:  you need an OBJECT for one of YOUR OWN later LOCAL steps.
            Requires "target", "item" (copied exactly from the target's has_items), "enables".
"enables" is the index (0-based, in your "steps" list) of YOUR OWN later LOCAL step that is
physically impossible before the request is done (e.g. you cannot put the Mug on the table before
you receive the Mug). Never request parts of the task your own steps do not depend on.

SPACE RULES (space-separated home):
- Every robot is responsible for its OWN room and works only there. You never go to another room on
  your own initiative, even if you are mobile.
- Objects from another room: write RECEIVE to that room's robot. Objects are handed over at the room
  doors; how they travel is not your concern.
- Work in your room that your body cannot do (out of reach, too heavy, you are fixed in place): write
  ASK_HELP to a robot that can, and say exactly what it should do. A mobile robot leaves its room ONLY
  when it accepts such a request.

OBJECT HANDOFFS (two ways, both decided by the other side):
- You NEED an object from another room: write RECEIVE (+ "enables": your step that uses it).
- You HAVE an object that another room's work needs: write PASS with "target" (that robot) and
  "item" (copied from YOUR has_items). This is an OFFER; the target decides whether to take it.
  PASS already includes picking the object up and handing it over at your door. Never write steps
  like "put it on the counter for collection" or "leave it by the door": write the PASS instead.
- If you prepare an object for another room (e.g. fill a glass with water), follow it with the PASS
  and give "prepared_by": the index of the preparing step (if the offer is not taken, the
  preparation is cancelled automatically).

PHYSICAL RULES (simulator): a robot holds at most ONE object at a time. Write moving an object as
ONE step with its destination, e.g. "move the Mug from the counter to the CoffeeTable". Fixed
furniture (beds, counters, sinks, bathtubs) cannot be moved. Do NOT write steps that only move,
look, check, search or verify: robots cannot explore, travel is added automatically, and what other
robots see is already in their offers.

"duration" is minutes, one of {durations}. Order the steps in the order you would execute them.

Return JSON:
{{
 "reasoning": "which goals matter, which part is mine, what I depend on",
 "goals": [{{"goal": "...", "priority": "high"}}, {{"goal": "...", "priority": "low"}}],
 "steps": [
  {{"type": "RECEIVE", "item": "...", "target": "R1", "enables": 1, "serves": 0, "duration": 1}},
  {{"type": "LOCAL", "action": "move ... from ... to ...", "uses": 0, "serves": 0, "duration": 3}},
  {{"type": "ASK_HELP", "basis": "dependency", "action": "move the heavy ... to ...", "target": "R2", "enables": 3, "serves": 1, "duration": 5}},
  {{"type": "ASK_HELP", "basis": "time", "action": "move the two light chairs to the wall", "target": "R3", "serves": 0, "duration": 3}},
  {{"type": "LOCAL", "action": "...", "uses": 1, "serves": 1, "duration": 2}},
  {{"type": "LOCAL", "action": "fill the Cup with water at the sink", "uses": 2, "serves": 1, "duration": 1}},
  {{"type": "PASS", "item": "Cup", "target": "R2", "prepared_by": 4, "serves": 1, "duration": 1}}
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
Judge each request, and each object offered to you, using everything you know (including your
private observations).

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
  You hold one object at a time; write object moves as "move X from A to B".
- for RECEIVE you will add a PASS step: give "item", copied exactly from YOUR has_items.
  If you already OFFERED exactly this item to this robot (a PASS in your plan), just "accept":
  your offer is linked, no second PASS is added.
  PASS means you hand the item over at your room's door; it includes picking it up.
  Do not add other steps for the handoff.
- Accepting an ASK_HELP for another room is the ONLY reason to leave your room. Accept only if your
  body can do it, and only if it does not break your own room's work.
- "insert_after": the id of the step in your plan after which you do it, or "START".
- "duration": minutes, one of {durations}.

OFFERS TO YOU: other robots offer objects (PASS). "receive" it if it helps ANY goal of the task (even a
low-priority one) and still fits the deadline; a handoff costs only a few minutes. Say what you will do
with it. "decline" only with a concrete reason. Receiving adds to your plan a RECEIVE and ONE step of
yours that uses the object ("action", "uses", "duration", "insert_after").

Some ASK_HELP requests have "basis": "time": the robot is short of time and asks for parallel help in
its room. If you are mobile, have little to do, and your body can do it, accept.

Return JSON:
{{
 "offers": [
  {{"offer": "r1_s3", "decision": "receive", "action": "move the Mug to the CoffeeTable", "uses": 0,
    "insert_after": "r2_s4", "duration": 1}},
  {{"offer": "r4_s2", "decision": "decline", "reason": "..."}}
 ],
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
{requests}

OFFERS TO YOU:
{offers}"""


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

For DUPLICATE_WORK: decide whether the two steps produce the same result in the world (the same
objects end up in the same place), even if worded differently. If yes, keep one and drop or merge
the other. Keep the copy that can run: "can_run" is false when one of its "inputs" (requests) is
served by nobody, e.g. because the item was already promised to the other copy. Dropping a step also
withdraws the requests that only existed for it. Otherwise prefer the copy with a CONFIRMED edge.
Use "unresolved" with why "not duplicate" only if the two steps clearly produce different results.

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
