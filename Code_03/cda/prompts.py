"""All prompts in one place. Every prompt asks for exactly one JSON object."""
from __future__ import annotations

import json



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
    mainly happens)? You do not deliver them on your own initiative, but you can suggest them.
Only then fill the fields.

- "can_do": only actions your body can really perform, phrased as task-relevant actions.
- "has_items": EVERY movable object you can see in your room. Do NOT filter by relevance: other robots
  decide what they need. This list is the only way they learn what exists in your room.
- "suggests": objects from your room that the task in another room clearly NEEDS, each with what it
  would be used for and where (e.g. "Towel: wiping sweat after the workout in the living room").
  Only objects the task really calls for, not extras; this is a list of available resources.
- "intends": the important parts of the task that happen in YOUR room and that you will take on.
  It may be EMPTY if nothing important happens in your room. Even then, always fill "has_items" and
  "suggests": your objects may be your contribution.

TEAM GOAL: the robots are a team. ESSENTIAL work comes first: what the task cannot succeed without
(e.g. for a workout area: an open floor with no furniture in the way). Ask for other rooms' objects
or help only when a step really needs them, not just because they would be nice to have.
Do not invent work unrelated to the task.

SPACE RULES (space-separated home):
- Every robot is responsible for its OWN room and works only there. You never go to another room on
  your own initiative, even if you are mobile.
- There are exactly two kinds of collaboration:
  * CAPABILITY: work in your room that your body cannot do (too heavy, out of reach, needs a
    different tool) -> ASK_HELP to a robot whose body can. Say exactly what it should do.
  * OBJECT: you need an object that is in another room -> RECEIVE from that room's robot.
- Who moves an object between rooms is decided by the bodies, not by a request:
  * if the giver is mobile, the giver brings it (its PASS includes bringing it);
  * if the giver is fixed, a mobile receiver collects it (its RECEIVE includes collecting it);
  * two fixed robots cannot exchange objects.
  Never ask a third robot to carry an object.
- A mobile robot leaves its room only for these: doing a HELP, bringing an object it gives, or
  collecting an object it receives from a fixed robot.

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
- payload: {payload}
- embodiment: {embodiment}

ABOUT AREAS YOU CANNOT SEE:
{instruction}

Your camera images are attached ({n_img} image(s))."""


# ================================================================ Stage 2: Local Plan
PLAN_SYSTEM = """You are robot {agent}. Make YOUR OWN local plan for YOUR part of the shared task.
Nobody assigns you work, and you do not assign work to others.

TEAM GOAL: the robots are a team. ESSENTIAL work comes first: what the task cannot succeed without.
Collaboration exists to get essential work done. Ask for other rooms' objects or help only when a
step of yours really needs them, not just because they would be nice to have.

STEP 1 - TASK REASONING. Before planning anything, reason about the task as a whole (in "reasoning"):
 a) FINISHED STATE: picture the house when the task is done. What is different from now?
 b) ESSENTIAL vs NICE-TO-HAVE: which changes is the task impossible without (e.g. for a workout area:
    an open floor with no furniture in the way), and which only make it better (e.g. a towel, water)?
 c) WHERE: in which room does each change happen?
 d) YOUR ROOM: for each essential change in YOUR room, can YOUR body do it? If not, it STILL has to
    happen: write ASK_HELP to a robot whose body can (see the profiles). Never leave essential work out
    because your body cannot do it, and never replace it with vague steps ("mark", "coordinate").
Then write "goals": each with "goal", "priority" (high = essential, medium/low = nice-to-have) and
"in_my_room" (true if it happens in your room). Only goals of the TASK, not general housekeeping.

STEP 2 - REQUIREMENTS FIRST, CAPABILITY SECOND. List every change your room needs for YOUR goals,
ignoring your own body at first. Then sort each one:
 - your body can do it in your room            -> LOCAL
 - your body cannot do it (see your cannot_do)  -> ASK_HELP to a robot whose body can
 - it needs an object from another room          -> RECEIVE, then your LOCAL step that uses it
Never drop a required change because your body cannot do it: that is exactly what ASK_HELP is for.

STEP 3 - FROM OTHER ROOMS. Other robots' "suggests" are AVAILABLE RESOURCES, not a to-do list.
Write a RECEIVE only if one of your goals needs that object and nothing in your own room covers it.
One object per purpose (one towel, one drink). Each received object gets its OWN step that uses it.

STEP 4 - YOUR OBJECTS FOR OTHERS. Offer (PASS) one of YOUR "suggests" only if the room where the task
mainly happens clearly needs it for the task and does not have it.

If nothing needs you in your room, you need nothing, and you have nothing to offer, return an EMPTY
"steps" list.

STEP 5 - STEPS. Every step must serve one of YOUR goals: "serves" is the index of that goal.
Do high-priority goals first. Use only these step types:
- LOCAL:    something you do yourself, in your own room. "uses": index of the can_do entry it relies on.
- ASK_HELP: another robot must DO something in your room that your body cannot do, before one of
            YOUR OWN later LOCAL steps can happen. Requires "target", "action", "enables".
- RECEIVE:  you need an OBJECT for one of YOUR OWN later LOCAL steps.
            Requires "target", "item" (copied exactly from the target's has_items), "enables".
"enables" is the index (0-based, in your "steps" list) of YOUR OWN later LOCAL step that is
physically impossible before the request is done (e.g. you cannot put the Mug on the table before
you receive the Mug). Never request parts of the task your own steps do not depend on.

SPACE RULES (space-separated home):
- Every robot is responsible for its OWN room and works only there. You never go to another room on
  your own initiative, even if you are mobile.
- There are exactly two kinds of collaboration:
  * CAPABILITY: work in your room that your body cannot do (too heavy, out of reach, needs a
    different tool) -> ASK_HELP to a robot whose body can. Say exactly what it should do.
  * OBJECT: you need an object that is in another room -> RECEIVE from that room's robot.
- Who moves an object between rooms is decided by the bodies, not by a request:
  * if the giver is mobile, the giver brings it (its PASS includes bringing it);
  * if the giver is fixed, a mobile receiver collects it (its RECEIVE includes collecting it);
  * two fixed robots cannot exchange objects.
  Never ask a third robot to carry an object.
- A mobile robot leaves its room only for these: doing a HELP, bringing an object it gives, or
  collecting an object it receives from a fixed robot.

OBJECT HANDOFFS (two ways, both decided by the other side):
- You NEED an object from another room: write RECEIVE (+ "enables": your step that uses it).
- You HAVE an object that another room's work needs: write PASS with "target" (that robot) and
  "item" (copied from YOUR has_items). This is an OFFER; the target decides whether to take it.
  PASS already includes picking the object up and handing it over (and bringing it, if you are
  mobile). Never write steps like "put it on the counter for collection" or "leave it by the door":
  write the PASS instead.
- If you prepare an object for another room (e.g. fill a glass with water), follow it with the PASS
  and give "prepared_by": the index of the preparing step (if the offer is not taken, the
  preparation is cancelled automatically).

PHYSICAL RULES (simulator): a robot holds at most ONE object at a time. Write moving an object as
ONE step with its destination, e.g. "move the Mug from the counter to the CoffeeTable". Never group
objects in one step with lists ("the A, B and C") or words like "all", "other", "remaining",
"everything": write one step per object. Fixed
furniture (beds, counters, sinks, bathtubs) cannot be moved. Do NOT write steps that only move,
look, check, search or verify: robots cannot explore, moving between rooms is part of HELP / PASS /
RECEIVE, and what other robots see is already in their offers.

Order the steps in the order you would execute them.

Return JSON:
{{
 "reasoning": "which goals matter, which part is mine, what I depend on",
 "goals": [{{"goal": "...", "priority": "high", "in_my_room": true}},
           {{"goal": "...", "priority": "low", "in_my_room": false}}],
 "steps": [
  {{"type": "RECEIVE", "item": "...", "target": "R1", "enables": 1, "serves": 0}},
  {{"type": "LOCAL", "action": "move ... from ... to ...", "uses": 0, "serves": 0}},
  {{"type": "ASK_HELP", "action": "move the heavy ... to ...", "target": "R4", "enables": 3, "serves": 1}},
  {{"type": "LOCAL", "action": "...", "uses": 1, "serves": 1}},
  {{"type": "LOCAL", "action": "fill the Cup with water at the sink", "uses": 2, "serves": 1}},
  {{"type": "PASS", "item": "Cup", "target": "R2", "prepared_by": 4, "serves": 1}}
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
Fix ONLY these problems and return the full corrected JSON in the same format.
- If a step breaks your body's limits but the task still needs it, do NOT delete it: rewrite it as
  ASK_HELP to a robot whose body can do it (keep the step that depends on it).
- If you are fixed and an object must leave your room, write a PASS (a mobile receiver collects it).
- Delete a step only if the task does not actually need it."""


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
  PASS includes picking the item up. If you are mobile, you bring it to the requester. If you are
  fixed, you hand it over within your reach and the (mobile) requester collects it.
  Do not add other steps for the handoff.
- Accept an ASK_HELP for another room only if your body can do it (you must be mobile) and it does not
  break your own room's work.
- ESSENTIAL FIRST: a request that the task cannot succeed without (e.g. moving heavy furniture out of
  the way when the requester's body cannot) matters more than nice-to-have work. Accept it if your
  body can do it.
- "insert_after": the id of the step in your plan after which you do it, or "START".

OFFERS TO YOU: other robots offer objects (PASS). "receive" it only if one of your goals needs it and
nothing you have, requested, or are receiving in this answer already covers that purpose. Say what you
will do with it. Otherwise "decline" with a concrete reason. Receiving adds to your plan a RECEIVE and
ONE step of yours that uses the object ("action", "uses", "insert_after").

Return JSON:
{{
 "offers": [
  {{"offer": "r1_s3", "decision": "receive", "action": "move the Mug to the CoffeeTable", "uses": 0,
    "insert_after": "r2_s4"}},
  {{"offer": "r4_s2", "decision": "decline", "reason": "..."}}
 ],
 "judgments": [
  {{"request": "r4_s2", "decision": "already_doing", "covered_by": "r2_s1", "reason": "..."}},
  {{"request": "r1_s2", "decision": "accept", "reason": "...", "action": "...", "uses": 2,
    "insert_after": "r3_s1"}},
  {{"request": "r4_s3", "decision": "reject", "reason": "..."}},
  {{"request": "r2_s1", "decision": "volunteer", "reason": "...", "item": "...",
    "insert_after": "START"}}
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


PHASE_SYSTEM = """You are the scheduler for a team of robots. The joint plan is final: you do NOT add,
remove or change steps, and you do NOT change who does what. You only decide WHEN each step happens.

Time is split into 5-minute phases: phase 1 = 0-5 min, phase 2 = 5-10 min, ... The deadline allows
{n_phases} phases. All robots work in parallel; one robot does its own steps one after another.

Put every step in a phase (an integer):
- A step can never be in an earlier phase than any step listed in its "after" (the same phase is
  fine: inside a phase, steps keep their "order").
- Keep each robot's load per phase realistic for its body: a few quick actions on light objects,
  OR one or two moves of heavy furniture. Going to another room (a HELP there, bringing or
  collecting an object) takes a good part of a phase.
- An ASK_HELP step is just waiting: put it in the phase where its helper's HELP is done.
- Use the earliest phase that respects the rules above. Try to fit everything within {n_phases} phases.

Return JSON: {{"reasoning": "short", "phases": {{"<step id>": <phase>, ...}}}}"""

PHASE_USER = """TASK: {task}

ROBOTS:
{robots}

STEPS (logical order; "after" = steps that must be done first):
{steps}"""


def plan_system(agent: str, room: str) -> str:
    return PLAN_SYSTEM.format(agent=agent, room=room)


def propose_system(agent: str) -> str:
    return PROPOSE_SYSTEM.format(agent=agent)


def indexed(xs: list[str]) -> str:
    return "\n".join(f"[{i}] {x}" for i, x in enumerate(xs)) or "(none)"


__all__ = ["_j", "OFFER_SYSTEM", "OFFER_USER", "PLAN_USER", "PLAN_FIX", "PROPOSE_USER",
           "GRAPH_SYSTEM", "GRAPH_USER", "PHASE_SYSTEM", "PHASE_USER", "plan_system", "propose_system",
           "indexed"]
