"""Stage 2 - LOCAL PLANNING.

Every robot plans for itself (no task allocation). It reads the other robots' broadcast
Offers, marks the steps that need collaboration (NEED / PASS) and broadcasts its plan.
A robot that can do something another robot asked for volunteers by writing a PASS/task
step into its OWN plan.
"""
from __future__ import annotations

import json

from runtime import Agent
from schemas import AgentInput, LocalPlan, Offer, RawLocalPlan

LOCAL_PLAN_SYSTEM = """You are one robot in a team of robots. Each robot works in a different room and can only see its own room. Nobody assigns work: you plan for yourself.

You receive: the shared TASK, your own CAPABILITY, two IMAGES of your own room, HIDDEN INFO, your own OFFER, and the OFFERS that all other robots broadcast.

Write YOUR OWN plan as an ordered list of steps, in the order you would execute them. Return ONE JSON object:
{
  "steps": [
    {
      "type": "LOCAL" | "NEED" | "PASS",
      "action": string,                 // one natural-language sentence
      "kind": "item" | "task" | null,   // required for NEED and PASS, null for LOCAL
      "item": string | null,            // required when kind is "item"
      "target": string | null           // optional hint: id of another robot, e.g. "agent_3"
    }
  ]
}

Step types:
- LOCAL: something you do alone in your own room.
- NEED + item: you must receive an item you do not have.
- NEED + task: something you cannot do (see your CAPABILITY) and want another robot to do for you.
- PASS + item: you hand an item you have to another robot who NEEDS it (see the other offers).
- PASS + task: you volunteer to do a task that another robot cannot do and has asked for. Describe in `action` where you must go and what you do (e.g. "Go to the living room and move the sofa aside"). Volunteer only if your CAPABILITY allows it.

Rules:
- Plan only what you can do with your own capability. If the task needs something you cannot do, add a NEED step instead of pretending.
- Use only objects visible in your images or stated in your HIDDEN INFO. Never invent objects.
- Add a PASS step only for something another robot actually asks for in its OFFER needs.
- `target` is only a hint about who you expect to match with; it is not binding. Use null if unsure.
- Respect any time limit in the TASK: keep the plan short and put NEED steps late enough that the other robot can prepare, and PASS steps early enough.
- Do not write steps for other robots. Return JSON only."""


def build_local_plan_user(inp: AgentInput, own_offer: Offer, others: dict[str, Offer]) -> str:
    hidden = "\n".join(f"- {h}" for h in inp.hidden_info) or "- (none)"
    others_txt = json.dumps({a: o.model_dump() for a, o in sorted(others.items())}, ensure_ascii=False, indent=2)
    return (
        f"TASK: {inp.task}\n\nYOU ARE: {own_offer.agent}\nCAPABILITY: {inp.capability}\n\n"
        f"HIDDEN INFO:\n{hidden}\n\nYOUR OFFER:\n{json.dumps(own_offer.model_dump(), ensure_ascii=False, indent=2)}\n\n"
        f"OTHER ROBOTS' OFFERS:\n{others_txt}\n\nThe attached images show your own room."
    )


async def make_local_plan(agent: Agent, known_agents: set[str]) -> LocalPlan:
    assert agent.offer is not None, "make_offer() must run first"
    agent.receive()
    user = build_local_plan_user(agent.inp, agent.offer, agent.others_offers)

    def parse(raw: dict) -> LocalPlan:
        return LocalPlan.from_raw(agent.id, RawLocalPlan.model_validate(raw), known_agents)

    agent.plan = await agent.ask("plan", LOCAL_PLAN_SYSTEM, user, parse)
    agent.log.log("plan", agent.id, "plan_made", n_steps=len(agent.plan.steps), n_collab=len(agent.plan.collaboration_steps()))
    agent.bus.broadcast(agent.id, "plan", agent.plan.model_dump(), phase="plan")
    return agent.plan
