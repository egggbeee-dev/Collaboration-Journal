"""Canned LLM answers for the 4-robot 'home training' example (--mock and tests).
Illustrative only (not model output): lets you check the plumbing without an API key."""

_CAP = {
    "agent_1": "Small wheeled robot with one arm. Can carry light objects (under 2 kg) and tidy small items. Cannot move heavy furniture.",
    "agent_2": "Wheeled robot with one arm. Can open the fridge and cabinets and carry light objects (under 2 kg). Cannot lift heavy objects.",
    "agent_3": "Wheeled robot with a strong arm. Can carry heavy objects (up to 20 kg) and move furniture.",
    "agent_4": "Compact robot. Can carry only very light objects such as towels. Cannot move furniture or heavy items.",
}


def _offer(aid, can_do, cannot_do, provide, needs):
    return {"capability": _CAP[aid], "can_do": can_do, "cannot_do": cannot_do, "can_provide": provide, "needs": needs}


def _step(type_, action, kind=None, item=None, target=None):
    return {"type": type_, "action": action, "kind": kind, "item": item, "target": target}


DEMO_SCRIPT = {
    "offer:agent_1": _offer(
        "agent_1", ["carry light objects", "tidy small items", "arrange items on the floor"], ["move heavy furniture"], [],
        [{"kind": "task", "text": "Move the sofa aside"}, {"kind": "item", "text": "yoga mat"},
         {"kind": "item", "text": "dumbbells"}, {"kind": "item", "text": "water bottle"}, {"kind": "item", "text": "towel"}],
    ),
    "offer:agent_2": _offer("agent_2", ["open fridge and cabinets", "carry light objects"], ["lift heavy objects"], ["water bottle"], []),
    "offer:agent_3": _offer("agent_3", ["carry heavy objects", "move furniture"], [], ["yoga mat", "dumbbells"], []),
    "offer:agent_4": _offer("agent_4", ["carry very light objects"], ["move furniture", "carry heavy objects"], ["towel"], []),
    "plan:agent_1": {"steps": [
        _step("LOCAL", "Clear clutter from the living room floor"),
        _step("NEED", "Have the sofa moved aside", "task", None, "agent_3"),
        _step("NEED", "Receive a yoga mat", "item", "yoga mat", "agent_3"),
        _step("NEED", "Receive dumbbells", "item", "dumbbells", "agent_3"),
        _step("NEED", "Receive a water bottle", "item", "water bottle", "agent_2"),
        _step("NEED", "Receive a towel", "item", "towel", "agent_4"),
        _step("LOCAL", "Lay out the yoga mat and place the dumbbells, water bottle and towel next to it"),
    ]},
    "plan:agent_2": {"steps": [
        _step("LOCAL", "Open the fridge and take out a water bottle"),
        _step("PASS", "Bring the water bottle to the living room and hand it over", "item", "water bottle", "agent_1"),
    ]},
    "plan:agent_3": {"steps": [
        _step("LOCAL", "Take the yoga mat and dumbbells from the closet"),
        _step("PASS", "Go to the living room and move the sofa aside", "task", None, "agent_1"),
        _step("PASS", "Hand the yoga mat over", "item", "yoga mat", "agent_1"),
        _step("PASS", "Hand the dumbbells over", "item", "dumbbells", "agent_1"),
    ]},
    "plan:agent_4": {"steps": [
        _step("LOCAL", "Take a clean towel from the bathroom cabinet"),
        _step("PASS", "Bring the towel to the living room and hand it over", "item", "towel", "agent_1"),
    ]},
    # the central graph LLM: one active fix (a cross-robot ordering the sequences do not imply)
    "graph:reasoner": {"ops": [
        {"op": "add_order", "before": "1-1", "after": "3-2",
         "reason": "The floor must be cleared before the sofa is moved, otherwise clutter gets in the way."},
    ]},
}
