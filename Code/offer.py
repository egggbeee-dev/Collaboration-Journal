"""Stage 1 - OFFER.

Every robot looks at its own images + capability + hidden info and broadcasts what it can
do, cannot do, can provide and needs. Nobody assigns anything.

Prompt design notes (informed by the KCC 2-agent pilot, p2p_phase.py phase1_offer):
- A concrete few-shot example anchors the model on "verb + specific visible object" style
  instead of vague, task-restating phrases ("clear the space", "set up the room").
- `obs_scope` is asked for explicitly (separate from can_do) so a later rule check can verify
  that a robot's needs/can_do only reference things it actually reported seeing.
- can_provide is filtered post-hoc against NON_PASSABLE_KW: an item name built from an
  abstract/state word ("space", "surface", "confirmation", "status", ...) is not something a
  robot can physically hand to another robot, and letting it through just wastes an Auction
  candidate slot that will never validly match.
- The system prompt repeats "based on your CAPABILITY" at each rule, not just once at the top,
  because a single mention at the top of a long prompt is easy for the model to drop by the
  time it reaches can_provide/needs.
"""
from __future__ import annotations

from runtime import Agent
from schemas import AgentInput, Offer, RawOffer

# Words that make an item non-transportable even if the model calls it an "item"
# (KCC's NON_PASSABLE_KW, ported as-is - this list is deliberately conservative).
NON_PASSABLE_KW = {
    "sink", "counter", "shelf", "surface", "floor", "wall", "room", "space", "area",
    "cleaned", "wiped", "organized", "tidied", "cleared", "arranged", "set", "setup",
    "confirmation", "confirm", "status", "done", "ready", "complete", "completed",
}


def _keywords(text: str) -> set[str]:
    import re

    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _is_passable(item: str) -> bool:
    """A can_provide entry must name a physical object, not a state or a place."""
    return not bool(_keywords(item) & NON_PASSABLE_KW)


_OFFER_EXAMPLE = """EXAMPLE - kitchen robot, task "prepare a picnic basket":
{
  "capability": "Wheeled robot with a gripper arm. Can open the fridge and cabinets and carry light objects (under 2 kg).",
  "obs_scope": ["refrigerator", "countertop", "fruit bowl with apples", "bread basket", "cabinet"],
  "can_do": ["take an apple from the fruit bowl", "take bread from the bread basket", "wipe the countertop"],
  "cannot_do": ["move the dining table"],
  "can_provide": ["apple", "bread"],
  "needs": [{"kind": "item", "text": "picnic basket"}]
}
Notice: can_do/can_provide name SPECIFIC objects from obs_scope, not the task itself
("prepare picnic basket" would be wrong - too vague and not physically checkable)."""

OFFER_SYSTEM = f"""You are one robot in a team of robots. Each robot works in a different room and can only see its own room.
Nobody assigns work: there is no task allocator. You decide for yourself what you can do and what you would need from others, based strictly on your own CAPABILITY.

You receive: the shared TASK, your CAPABILITY, two IMAGES of your own room, and HIDDEN INFO (facts about places you cannot see or explore, e.g. the inside of a fridge). You cannot see or enter other rooms while planning.

{_OFFER_EXAMPLE}

Write your OFFER. It will be broadcast to every other robot. Return ONE JSON object with exactly these keys:
{{
  "capability": string,                 // copy your capability text as given; do not embellish
  "obs_scope": [string],                // every object or area you actually see, from images + HIDDEN INFO only
  "can_do": [string],                   // "verb + specific object from obs_scope", filtered by what your CAPABILITY allows
  "cannot_do": [string],                // task-relevant actions your CAPABILITY rules out
  "can_provide": [string],              // physical, hand-sized objects from obs_scope you could carry to another robot
  "needs": [ {{"kind": "item" | "task", "text": string}} ]
                                        // things you would need from others to help finish the task:
                                        // specific items you lack, or specific tasks you cannot do because of your CAPABILITY
}}

Rules (each checked against your CAPABILITY and obs_scope, not the task in the abstract):
- obs_scope: list only what is actually visible in your images or stated in HIDDEN INFO. Never invent objects.
- can_do: every entry must use an object from obs_scope AND be something your CAPABILITY allows. If your capability says you cannot lift heavy things, no can_do entry may involve heavy objects.
- can_provide: only tangible objects small/light enough to hand over (a cup, a tool, food, a document). NEVER things like "clean surface", "cleared space", "confirmation", "the room being set up" - those are states, not objects, and cannot be carried. When in doubt, ask: "could I physically place this in someone's hands?" If no, leave it out.
- needs: name the specific item or specific task, not the task restated wholesale. "yoga mat" is a need; "set up living room" is not a need, it is the task itself and belongs in your own can_do/plan instead.
- Do not list a need for something you can do or provide yourself.
- Do not assign work to other robots. A need is a request; others may or may not volunteer.
- Keep every entry short and concrete (a few words). Return JSON only."""


def build_offer_user(inp: AgentInput) -> str:
    hidden = "\n".join(f"- {h}" for h in inp.hidden_info) or "- (none)"
    return (
        f"TASK: {inp.task}\n\nCAPABILITY: {inp.capability}\n\nHIDDEN INFO:\n{hidden}\n\n"
        "The attached images show your own room. Base obs_scope, can_do and can_provide only on "
        "what you see here plus HIDDEN INFO, filtered through what your CAPABILITY allows."
    )


async def make_offer(agent: Agent) -> Offer:
    raw: RawOffer = await agent.ask(
        "offer", OFFER_SYSTEM, build_offer_user(agent.inp), RawOffer.model_validate, banner_label="OFFER RAW"
    )

    kept = [p for p in raw.can_provide if _is_passable(p)]
    dropped = [p for p in raw.can_provide if not _is_passable(p)]
    if dropped:
        agent.log.log("offer", agent.id, "can_provide_filtered", dropped=dropped)
        if agent.verbose:
            print(f"  [OFFER FILTER] {agent.id}: non-passable can_provide dropped: {dropped}")
    raw = raw.model_copy(update={"can_provide": kept})

    agent.offer = Offer(agent=agent.id, **raw.model_dump())
    if agent.verbose:
        print(
            f"  [OFFER] {agent.id}: obs_scope={len(agent.offer.obs_scope)} can_do={len(agent.offer.can_do)} "
            f"can_provide={agent.offer.can_provide} needs={[n.text for n in agent.offer.needs]}"
        )
    agent.log.log("offer", agent.id, "offer_made", n_needs=len(agent.offer.needs), n_provide=len(agent.offer.can_provide))
    agent.bus.broadcast(agent.id, "offer", agent.offer.model_dump(), phase="offer")
    return agent.offer
