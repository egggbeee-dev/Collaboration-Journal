"""Core data structures for CDA.

Design rules
- Step IDs ("r2_s3") are created by code only, never by an LLM, and never renumbered.
- Every edge means the same thing: `src` must finish before `dst` can finish.
- The graph stage may drop steps or move a provider step, but it can never create a step
  and never remove a CONFIRMED edge (checked in graph.py).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

# ---------------------------------------------------------------- step / edge vocab
LOCAL, ASK_HELP, HELP, RECEIVE, PASS = "LOCAL", "ASK_HELP", "HELP", "RECEIVE", "PASS"
STEP_TYPES = {LOCAL, ASK_HELP, HELP, RECEIVE, PASS}
REQUEST_TYPES = {ASK_HELP, RECEIVE}        # created in Stage 2
PROVIDER_TYPES = {HELP, PASS}              # created only in Stage 3, as answers to a request
PROVIDER_FOR = {ASK_HELP: HELP, RECEIVE: PASS}

SEQ, TRANSFER, HELP_EDGE = "SEQ", "TRANSFER", "HELP"
CONFIRMED, PROPOSED = "CONFIRMED", "PROPOSED"

DURATIONS = (1, 2, 3, 5, 10)               # minutes; LLM picks one of these


def snap_duration(x) -> int:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 2
    return min(DURATIONS, key=lambda d: abs(d - v))


def agent_num(agent: str) -> int:
    return int(agent.lstrip("Rr"))


def norm(s: str | None) -> str:
    return " ".join((s or "").lower().split())


# ---------------------------------------------------------------- task config
@dataclass
class Profile:
    room: str
    mobile: bool
    payload_kg: float
    embodiment: str            # e.g. "wheeled base, one 6-DoF arm, low reach"


@dataclass
class AgentInput:
    id: str                    # "R1"
    profile: Profile
    images: list[str] = field(default_factory=list)
    instruction: str = ""      # text about areas the robot cannot see (no exploration)


@dataclass
class TaskConfig:
    task_id: str
    task: str
    deadline_min: int
    agents: list[AgentInput]
    travel_min: int = 2        # added when a HELP step happens in another robot's room

    @classmethod
    def load(cls, path: str | Path) -> "TaskConfig":
        path = Path(path)
        d = json.loads(path.read_text(encoding="utf-8"))
        agents = []
        for a in d["agents"]:
            imgs = [str((path.parent / p).resolve()) if not Path(p).is_absolute() else p
                    for p in a.get("images", [])]
            agents.append(AgentInput(a["id"], Profile(**a["profile"]), imgs, a.get("instruction", "")))
        return cls(d["task_id"], d["task"], d["deadline_min"], agents, d.get("travel_min", 2))

    @classmethod
    def from_notebook(cls, task: str, agents: list[dict], deadline_min: int = 20,
                      task_id: str = "task", travel_min: int = 2) -> "TaskConfig":
        """agents: [{"room", "mobile", "capability", "images", "hidden_info", "payload_kg"?}, ...]
        Robots are named R1..Rn in the given order. `hidden_info` (str or list of lines) becomes
        the robot's private text about what it cannot see / knows only itself."""
        out = []
        for i, a in enumerate(agents, start=1):
            hid = a.get("hidden_info", "")
            hid = "\n".join(hid) if isinstance(hid, list) else (hid or "")
            out.append(AgentInput(f"R{i}", Profile(room=a["room"], mobile=bool(a["mobile"]),
                                                   payload_kg=float(a.get("payload_kg", 0) or 0),
                                                   embodiment=a["capability"]),
                                  list(a.get("images", [])), hid))
        return cls(task_id, task, deadline_min, out, travel_min)

    def agent(self, aid: str) -> AgentInput:
        return next(a for a in self.agents if a.id == aid)

    @property
    def ids(self) -> list[str]:
        return [a.id for a in self.agents]


# ---------------------------------------------------------------- stage 1
@dataclass
class Offer:
    agent: str
    capability: str
    can_do: list[str]
    has_items: list[str]
    need_from_others: list[str]
    # private (never broadcast)
    obs_scope: str = ""
    cannot_do: list[str] = field(default_factory=list)
    reasoning: str = ""

    def public(self) -> dict:
        return {"agent": self.agent, "capability": self.capability, "can_do": self.can_do,
                "has_items": self.has_items, "need_from_others": self.need_from_others}

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------- graph elements
@dataclass
class Node:
    id: str
    agent: str
    type: str
    action: str = ""
    item: str | None = None
    target: str | None = None          # request: who is asked; provider: requester agent
    uses: int | None = None            # index into the agent's own can_do
    location: str = ""
    duration: int = 2
    origin: str = "local"              # local | accept | volunteer
    answers: str | None = None         # provider -> id of the request it answers
    status: str = "active"             # active | dropped | blocked
    violations: list[str] = field(default_factory=list)
    travel: int = 0                    # added by the scheduler
    t_start: int | None = None
    t_end: int | None = None

    def brief(self) -> dict:
        d = {"id": self.id, "agent": self.agent, "type": self.type}
        if self.action:
            d["action"] = self.action
        if self.item:
            d["item"] = self.item
        if self.target:
            d["target"] = self.target
        return d

    def text(self) -> str:
        if self.type in (RECEIVE, PASS):
            return f"{self.action} ({self.item})" if self.action else (self.item or "")
        return self.action


@dataclass
class Edge:
    src: str
    dst: str
    kind: str                          # SEQ | TRANSFER | HELP
    status: str = CONFIRMED            # CONFIRMED | PROPOSED  (SEQ is always CONFIRMED)
    source: str = "local"              # local | stage3 | graph

    @property
    def key(self) -> tuple[str, str]:
        return (self.src, self.dst)


def describe_config(cfg: TaskConfig) -> str:
    lines = [f"TASK ({cfg.task_id}, deadline {cfg.deadline_min} min): {cfg.task}", ""]
    for a in cfg.agents:
        p = a.profile
        lines.append(f"{a.id} | room={p.room} | mobile={p.mobile} | {p.embodiment}")
        lines.append(f"     images={a.images}")
        if a.instruction:
            lines.append(f"     hidden_info={a.instruction}")
    return "\n".join(lines)
