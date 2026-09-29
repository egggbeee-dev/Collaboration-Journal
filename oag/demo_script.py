"""Canned LLM answers for `python pipeline.py --task examples/home_training.json --mock`.

Keys are "<phase>:<agent>". A list value is consumed in order (last one repeats):
agent_3's first Local Plan breaks the request rule on purpose, to show the retry.
"""


def _offer(cap, provide=(), cannot=()):
    return {"capability": cap, "obs_scope": [{"object": "room"}], "can_do": [],
            "cannot_do": list(cannot), "can_provide": list(provide), "needs": []}


def _local(action):
    return {"type": "LOCAL", "action": action}


_HEAVY = {"action": "move", "object": "heavy furniture"}

DEMO_SCRIPT = {
    "offer:agent_1": _offer("Fixed-base arm in the kitchen. Cannot leave the kitchen."),
    "offer:agent_2": _offer("Heavy-duty mobile robot. Can move heavy furniture.",
                            provide=[{"type": "task", **_HEAVY}]),
    "offer:agent_3": _offer("Light-duty mobile robot in the bedroom. Cannot move heavy furniture.",
                            provide=[{"type": "item", "object": "ball"}],
                            cannot=[_HEAVY]),
    "offer:agent_4": _offer("Light-duty mobile robot in the bathroom. Cannot move heavy furniture.",
                            provide=[{"type": "item", "object": "bath mat"},
                                     {"type": "task", "action": "carry", "object": "light objects"}],
                            cannot=[_HEAVY]),

    "plan:agent_1": {"steps": []},
    "plan:agent_2": {"steps": [_local("Move the coffee table to clear space"),
                               _local("Move the desk to clear space")]},
    "plan:agent_3": [
        # invalid: request with no own later step -> retry
        {"steps": [_local("Move the ball to the living room"),
                   {"type": "ASK_HELP", "action": "Move heavy furniture in the living room"}]},
        {"steps": [
            _local("Move the ball to the living room"),
            {"type": "ASK_HELP", "action": "Move the heavy sofa away from the free floor spot",
             "target": "agent_2", "purpose": "Lay the exercise mat on the free floor spot"},
            {"type": "RECEIVE", "action": "Receive an exercise mat", "item": "exercise mat",
             "purpose": "Lay the exercise mat on the free floor spot"},
            _local("Lay the exercise mat on the free floor spot"),
        ]},
    ],
    "plan:agent_4": {"steps": [
        {"type": "ASK_HELP", "action": "Move the heavy cabinet blocking the towel shelf",
         "purpose": "Take a towel from the shelf"},
        _local("Take a towel from the shelf"),
    ]},

    "graph:reasoner": {"ops": []},
}
