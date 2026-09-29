"""Scripted LLM answers for `python pipeline.py --task examples/home_training.json --mock`.

Keys are "<phase>:<agent>". The plans leave gaps on purpose so the graph has work to do:
- R4 has a bath mat but did not volunteer a PASS        -> complete_transfer adds it
- R1 offers the water bottle, R3 uses it but has no RECEIVE -> complete_transfer adds it
- R3 moves a cardboard box (unrelated)                  -> drop
- R2 keeps moving furniture after helping                -> add_order (same-room conflict)
- one wrong op (RECEIVE served by HELP)                  -> rejected by the rules
Stage 3: R2 and R3 both propose 2-2 -> 3-1 (mutual, connected before the graph LLM);
R3's one-sided 1-2 -> 3-2 is wrong and the graph does not use it.
"""


def _offer(cap, *, can_do=(), cannot=(), items=(), needs=()):
    return {"reasoning": "(scripted)", "capability": cap, "obs_scope": [{"object": "room"}],
            "can_do": list(can_do), "cannot_do": list(cannot),
            "has_items": list(items), "need_from_others": list(needs)}


def _s(t, action, item=None, target=None):
    return {"type": t, "action": action, "item": item, "target": target}


DEMO_SCRIPT = {
    "offer:agent_1": _offer(
        "Fixed-base robot arm mounted in the kitchen. Cannot leave the kitchen. Can pick up and hand over items within reach.",
        can_do=[{"action": "pick and hand over", "object": "water bottle"}],
        cannot=[{"action": "leave the kitchen", "reason": "fixed base"}],
        items=[{"object": "water bottle", "location": "kitchen"}],
        needs=[{"kind": "task", "what": "carry the water bottle to the living room"}]),
    "offer:agent_2": _offer(
        "Heavy-duty mobile robot in the living room. Can move heavy furniture.",
        can_do=[{"action": "move", "object": "heavy furniture"}]),
    "offer:agent_3": _offer(
        "Light-duty mobile robot in the bedroom. Can move between rooms and carry light objects. Cannot move heavy furniture.",
        can_do=[{"action": "carry", "object": "light objects"}],
        cannot=[{"action": "move", "object": "heavy furniture"}],
        needs=[{"kind": "item", "what": "exercise mat"},
               {"kind": "task", "what": "move the heavy sofa away from the free floor spot"}]),
    "offer:agent_4": _offer(
        "Light-duty mobile robot in the bathroom. Can move between rooms and carry light objects. Cannot move heavy furniture.",
        can_do=[{"action": "carry", "object": "light objects"}],
        items=[{"object": "bath mat", "location": "bathroom"}]),

    "plan:agent_1": {"reasoning": "(scripted)", "steps": [
        _s("LOCAL", "Pick the water bottle from the high shelf"),
        _s("PASS", "Hand over the water bottle", item="water bottle")]},
    "plan:agent_2": {"reasoning": "(scripted)", "steps": [
        _s("LOCAL", "Move the coffee table to the wall"),
        _s("HELP", "Move the heavy sofa away from the free floor spot", target="agent_3"),
        _s("LOCAL", "Move the armchair to the corner of the living room")]},
    "plan:agent_3": {"reasoning": "(scripted)", "steps": [
        _s("ASK_HELP", "Move the heavy sofa away from the free floor spot", target="agent_2"),
        _s("RECEIVE", "Receive an exercise mat", item="exercise mat", target="agent_4"),
        _s("LOCAL", "Lay the mat on the free floor spot"),
        _s("LOCAL", "Move the cardboard box to the living room"),
        _s("LOCAL", "Place the water bottle next to the mat")]},
    "plan:agent_4": {"reasoning": "(scripted)", "steps": [
        _s("LOCAL", "Pick up the bath mat")]},

    # Stage 3 (Edge Proposal). R4 has no collaboration step -> skipped, no LLM call.
    "propose:agent_1": {"reasoning": "(scripted)", "edges": []},
    "propose:agent_2": {"reasoning": "(scripted)", "edges": [
        {"mine": "2-2", "other": "3-1", "why": "I can move heavy furniture; R3 needs the sofa moved"},
        {"mine": "2-1", "other": "3-1", "why": "invalid on purpose: 2-1 is LOCAL"}]},
    "propose:agent_3": {"reasoning": "(scripted)", "edges": [
        {"mine": "3-1", "other": "2-2", "why": "R2 volunteered to move the sofa"},
        {"mine": "3-2", "other": "1-2", "why": "wrong on purpose: a water bottle is not a mat"}]},

    "graph:reasoner": {"reasoning": "(scripted)", "ops": [
        {"op": "connect", "request": "3-1", "provider": "2-2", "reason": "R2 volunteered exactly this"},
        {"op": "connect", "request": "3-2", "provider": "2-2", "reason": "wrong on purpose"},
        {"op": "complete_transfer", "request": "3-2", "robot": "agent_4", "item": "bath mat",
         "reason": "a bath mat can serve as the exercise mat"},
        {"op": "complete_transfer", "provider": "1-2", "robot": "agent_3",
         "reason": "R3 places the water bottle"},
        {"op": "add_order", "before": "2-3", "after": "3-3",
         "reason": "same room: finish moving furniture before laying the mat"},
        {"op": "drop", "step": "3-4", "reason": "unrelated to the task"},
        {"op": "flag_gap", "description": "Nobody checks that the floor spot is clean before laying the mat"},
    ]},
}
