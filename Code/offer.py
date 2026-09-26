"""Stage 1 - OFFER.

Every robot looks at its own images + capability + hidden info and broadcasts what it can
do, cannot do, can provide and needs. Nobody assigns anything.
"""
from __future__ import annotations

from runtime import Agent
from schemas import AgentInput, Offer, RawOffer

OFFER_SYSTEM = """You are one robot in a team of robots. Each robot works in a different room and can only see its own room.
Nobody assigns work: there is no task allocator. You decide for yourself what you can do and what you would need from others.

You receive: the shared TASK, your CAPABILITY, two IMAGES of your own room, and HIDDEN INFO (facts about places you cannot see or explore, e.g. the inside of a fridge). You cannot see or enter other rooms while planning.

Write your OFFER. It will be broadcast to every other robot. Return ONE JSON object with exactly these keys:
{
  "capability": string,                 // copy your capability text as given; do not embellish
  "can_do": [string],                   // kinds of actions you can perform, derived ONLY from your capability
  "cannot_do": [string],                // task-relevant actions you cannot perform because of your capability
  "can_provide": [string],              // physical items you could hand to another robot
  "needs": [ {"kind": "item" | "task", "text": string} ]
                                        // things you would need from others to help finish the task:
                                        // items you lack, or tasks you cannot do yourself
}

Rules:
- List an item in can_provide ONLY if it is visible in your images or stated in HIDDEN INFO. Never invent objects.
- Do not list a need for something you can do or provide yourself.
- Do not assign work to other robots. A need is a request; others may or may not volunteer.
- Keep every entry short and concrete (a few words). Return JSON only."""


def build_offer_user(inp: AgentInput) -> str:
    hidden = "\n".join(f"- {h}" for h in inp.hidden_info) or "- (none)"
    return f"TASK: {inp.task}\n\nCAPABILITY: {inp.capability}\n\nHIDDEN INFO:\n{hidden}\n\nThe attached images show your own room."


async def make_offer(agent: Agent) -> Offer:
    raw: RawOffer = await agent.ask("offer", OFFER_SYSTEM, build_offer_user(agent.inp), RawOffer.model_validate)
    agent.offer = Offer(agent=agent.id, **raw.model_dump())
    agent.log.log("offer", agent.id, "offer_made", n_needs=len(agent.offer.needs), n_provide=len(agent.offer.can_provide))
    agent.bus.broadcast(agent.id, "offer", agent.offer.model_dump(), phase="offer")
    return agent.offer
