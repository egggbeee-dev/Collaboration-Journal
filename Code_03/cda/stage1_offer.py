"""Stage 1 - Offer. Each robot, in parallel, declares capability / can_do / has_items / needs.
Only the public part is broadcast."""
from __future__ import annotations

from .llm import BaseLLM
from .log import EventLog
from .prompts import OFFER_SYSTEM, OFFER_USER
from .schemas import AgentInput, Offer, TaskConfig


def _suggests(x) -> list[str]:
    out = []
    for v in (x if isinstance(x, list) else [x] if x else []):
        if isinstance(v, dict):
            obj = v.get("object") or v.get("item") or v.get("name") or ""
            use = v.get("use") or v.get("for") or v.get("purpose") or v.get("why") or ""
            v = f"{obj}: {use}".strip(": ")
        if str(v).strip():
            out.append(str(v).strip())
    return out


def _as_list(x) -> list[str]:
    if isinstance(x, str):
        return [x] if x.strip() else []
    return [str(v).strip() for v in (x or []) if str(v).strip()]


async def make_offer(cfg: TaskConfig, a: AgentInput, llm: BaseLLM, log: EventLog) -> Offer:
    user = OFFER_USER.format(task=cfg.task, deadline=cfg.deadline_min, room=a.profile.room,
                             mobile=a.profile.mobile,
                             payload=(f"{a.profile.payload_kg:g} kg" if a.profile.payload_kg
                                      else "not specified (judge from your embodiment)"),
                             embodiment=a.profile.embodiment,
                             instruction=a.instruction or "(none)", n_img=len(a.images))
    d = await llm.complete(OFFER_SYSTEM.format(agent=a.id), user, key=f"offer:{a.id}", images=a.images)
    offer = Offer(
        agent=a.id,
        capability=str(d.get("capability", a.profile.embodiment)),
        can_do=_as_list(d.get("can_do")),
        has_items=_as_list(d.get("has_items")),
        need_from_others=_as_list(d.get("need_from_others")),
        intends=_as_list(d.get("intends")),
        suggests=_suggests(d.get("suggests")),
        obs_scope=str(d.get("obs_scope", "")),
        cannot_do=_as_list(d.get("cannot_do")),
        reasoning=str(d.get("reasoning", "")),
    )
    log.log("offer", a.id, "broadcast", can_do=len(offer.can_do), items=len(offer.has_items),
            needs=len(offer.need_from_others), intends=len(offer.intends),
            suggests=len(offer.suggests))
    return offer
