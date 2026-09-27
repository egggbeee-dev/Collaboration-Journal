"""Shared runtime pieces: event log, broadcast bus, validated LLM call, and the Agent object.

Each robot is an `Agent` with its own inbox and its own LLM calls. Agents never read each
other's private input; they only see what is broadcast on the bus.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from pydantic import ValidationError

from llm import BaseLLM
from schemas import AgentInput, LocalPlan, Offer


class PipelineError(RuntimeError):
    pass


# --------------------------------------------------------------------------- log
class EventLog:
    """Append-only event log; every metric (LLM calls, who sent what when) can be recomputed from it."""

    def __init__(self) -> None:
        self.events: list[dict] = []
        self._t0 = time.time()

    def log(self, phase: str, agent: str, event: str, **data) -> None:
        self.events.append(
            {"seq": len(self.events), "t": round(time.time() - self._t0, 4), "phase": phase, "agent": agent, "event": event, **data}
        )

    def dump(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for e in self.events:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- bus
@dataclass
class Message:
    sender: str
    kind: str  # "offer" | "plan"
    payload: dict[str, Any]
    seq: int


class Bus:
    """Broadcast channel: `broadcast` drops a copy into every other agent's inbox."""

    def __init__(self, log: EventLog) -> None:
        self._inboxes: dict[str, asyncio.Queue[Message]] = {}
        self._log = log
        self._seq = 0

    def register(self, agent_id: str) -> asyncio.Queue:
        q: asyncio.Queue[Message] = asyncio.Queue()
        self._inboxes[agent_id] = q
        return q

    def broadcast(self, sender: str, kind: str, payload: dict, phase: str) -> None:
        self._seq += 1
        msg = Message(sender=sender, kind=kind, payload=payload, seq=self._seq)
        for aid, q in self._inboxes.items():
            if aid != sender:
                q.put_nowait(msg)
        self._log.log(phase, sender, f"broadcast_{kind}", to=[a for a in self._inboxes if a != sender])


# --------------------------------------------------------------------------- console logging
# Print-based, human-readable trace (separate from EventLog, which is for machine metrics).
# Controlled per-agent by Agent.verbose / a global default; turn it off for large batch runs.

def banner(title: str) -> None:
    print("\n" + "=" * 68)
    print(f"  {title}")
    print("=" * 68)


def log_block(label: str, content: str) -> None:
    print(f"\n[{label}]\n{content}")


# --------------------------------------------------------------------------- validated call
async def call_validated(
    llm: BaseLLM,
    log: EventLog,
    *,
    phase: str,
    who: str,
    system: str,
    user: str,
    parse: Callable[[dict], Any],
    images: list[str] | None = None,
    max_retries: int = 2,
    verbose: bool = False,
    banner_label: str | None = None,
):
    """Call the LLM; if the JSON does not fit the schema, tell it what was wrong and retry.
    With verbose=True, prints a banner and the raw response before parsing (and any retry) -
    this is the trace that a one-line metrics summary cannot give you when a result looks off."""
    err: str | None = None
    for attempt in range(1 + max_retries):
        u = user if err is None else f"{user}\n\nYour previous JSON was invalid: {err}\nReturn corrected JSON only."
        if verbose and banner_label:
            banner(f"{banner_label} — {who}" + (f" (retry {attempt})" if attempt else ""))
        try:
            raw = await llm.complete_json(tag=f"{phase}:{who}", system=system, user=u, images=images)
            if verbose and banner_label:
                log_block("RAW RESPONSE", json.dumps(raw, ensure_ascii=False, indent=2))
            return parse(raw)
        except (ValidationError, ValueError) as e:  # JSONDecodeError and pydantic errors are ValueErrors
            err = str(e)[:600]
            log.log(phase, who, "validation_failed", attempt=attempt, error=err)
            if verbose:
                print(f"  [VALIDATION FAILED] attempt {attempt}: {err}")
    raise PipelineError(f"{who}: no valid {phase} JSON after {1 + max_retries} attempts")


# --------------------------------------------------------------------------- agent
class Agent:
    def __init__(self, agent_id: str, inp: AgentInput, llm: BaseLLM, bus: Bus, log: EventLog,
                 max_validation_retries: int = 2, verbose: bool = True) -> None:
        self.id = agent_id
        self.inp = inp
        self.llm = llm
        self.bus = bus
        self.log = log
        self.max_validation_retries = max_validation_retries
        self.verbose = verbose
        self.inbox = bus.register(agent_id)

        self.offer: Offer | None = None
        self.plan: LocalPlan | None = None
        self.others_offers: dict[str, Offer] = {}
        self.others_plans: dict[str, LocalPlan] = {}

    async def ask(self, phase: str, system: str, user: str, parse: Callable[[dict], Any], banner_label: str | None = None):
        """LLM call with this robot's own images attached."""
        return await call_validated(
            self.llm, self.log, phase=phase, who=self.id, system=system, user=user, parse=parse,
            images=self.inp.images, max_retries=self.max_validation_retries,
            verbose=self.verbose, banner_label=banner_label,
        )

    def receive(self) -> None:
        """Drain the inbox (everything broadcast so far)."""
        while not self.inbox.empty():
            msg = self.inbox.get_nowait()
            if msg.kind == "offer":
                self.others_offers[msg.sender] = Offer(**msg.payload)
            elif msg.kind == "plan":
                self.others_plans[msg.sender] = LocalPlan(**msg.payload)
