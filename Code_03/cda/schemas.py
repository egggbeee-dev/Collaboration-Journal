"""Core data structures for CDA.

Design rules
- Step IDs ("r2_s3") are created by code only, never by an LLM, and never renumbered.
- Every edge means the same thing: `src` must finish before `dst` can finish.
- Stage 3 only produces CANDIDATE collaborations (status PROPOSED). Stage 4 confirms them:
  a request with one candidate is confirmed by rule, one with several by the Graph LLM.
- The graph stage may drop steps, move a provider step and add ORDER edges between existing steps,
  but it can never create a step and never remove a CONFIRMED edge (checked in graph.py).
- Two kinds of collaboration only:
    ASK_HELP <-> HELP     capability: work the requester's body cannot do
    RECEIVE  <-> PASS     object: an object from another room
  There is no separate "carry" request. An object is moved by whichever end of the handoff is
  mobile (mobile giver brings it; otherwise a mobile receiver collects it). Two fixed robots
  cannot exchange objects.
- Time: code first computes a logical order (every step = one unit, no travel time). Stage 4 then
  has each step judged short / medium / long and packs the steps into 5-minute phases (PHASE_MIN)
  with a per-robot capacity, in dependency order (deterministic).
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

SEQ, TRANSFER, HELP_EDGE, ORDER = "SEQ", "TRANSFER", "HELP", "ORDER"   # ORDER: added by Stage 4
CONFIRMED, PROPOSED = "CONFIRMED", "PROPOSED"

PHASE_MIN = 5              # the Joint Plan is scheduled in 5-minute phases (assigned in Stage 4)


def agent_num(agent: str) -> int:
    return int(agent.lstrip("Rr"))


def norm(s: str | None) -> str:
    return " ".join((s or "").lower().split())


# ---------------------------------------------------------------- task config
@dataclass
class Profile:
    room: str
    mobile: bool
    payload_kg: float | None
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

    @classmethod
    def load(cls, path: str | Path) -> "TaskConfig":
        path = Path(path)
        d = json.loads(path.read_text(encoding="utf-8"))
        agents = []
        for a in d["agents"]:
            imgs = [str((path.parent / p).resolve()) if not Path(p).is_absolute() else p
                    for p in a.get("images", [])]
            agents.append(AgentInput(a["id"], Profile(**a["profile"]), imgs, a.get("instruction", "")))
        return cls(d["task_id"], d["task"], d["deadline_min"], agents)

    @classmethod
    def from_notebook(cls, task: str, agents: list[dict], deadline_min: int = 20,
                      task_id: str = "task") -> "TaskConfig":
        """agents: [{"room", "mobile", "capability", "images", "hidden_info", "payload_kg"?}, ...]
        Robots are named R1..Rn in the given order. `hidden_info` (str or list of lines) becomes
        the robot's private text about what it cannot see / knows only itself."""
        out = []
        for i, a in enumerate(agents, start=1):
            hid = a.get("hidden_info", "")
            hid = "\n".join(hid) if isinstance(hid, list) else (hid or "")
            out.append(AgentInput(f"R{i}", Profile(room=a["room"], mobile=bool(a["mobile"]),
                                                   payload_kg=(float(a["payload_kg"]) if a.get("payload_kg") else None),
                                                   embodiment=a["capability"]),
                                  list(a.get("images", [])), hid))
        return cls(task_id, task, deadline_min, out)

    def agent(self, aid: str) -> AgentInput:
        return next(a for a in self.agents if a.id == aid)

    @property
    def ids(self) -> list[str]:
        return [a.id for a in self.agents]

    def mover(self, giver: str, receiver: str) -> str | None:
        """Who physically moves an object from giver's room to receiver's room.
        A mobile giver brings it; otherwise a mobile receiver collects it; two fixed robots: None."""
        if giver in self.ids and self.agent(giver).profile.mobile:
            return giver
        if receiver in self.ids and self.agent(receiver).profile.mobile:
            return receiver
        return None


# ---------------------------------------------------------------- stage 1
@dataclass
class Offer:
    agent: str
    capability: str
    can_do: list[str]
    has_items: list[str]
    need_from_others: list[str]
    intends: list[str] = field(default_factory=list)   # parts of the task I plan to take (public)
    suggests: list[str] = field(default_factory=list)  # "Object: use" from my room for the task (public)
    # private (never broadcast)
    obs_scope: str = ""
    cannot_do: list[str] = field(default_factory=list)
    reasoning: str = ""

    def public(self) -> dict:
        return {"agent": self.agent, "capability": self.capability, "can_do": self.can_do,
                "has_items": self.has_items, "need_from_others": self.need_from_others,
                "intends": self.intends, "suggests": self.suggests}

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
    origin: str = "local"              # local | accept | volunteer
    answers: str | None = None         # provider -> id of the request it answers
    enables: str | None = None         # request -> id of the own later step that needs it
    serves: str | None = None          # the task goal (from the robot's own goal list) this step serves
    prepared_by: str | None = None     # offered PASS -> own earlier LOCAL step that prepared the object
    status: str = "active"             # active | dropped | blocked
    violations: list[str] = field(default_factory=list)
    size: str | None = None            # short | medium | long, judged in Stage 4 (for phase packing)
    phase: int | None = None           # 5-minute phase (1 = 0-5 min), assigned in Stage 4
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
    kind: str                          # SEQ | TRANSFER | HELP | ORDER
    status: str = CONFIRMED            # CONFIRMED | PROPOSED  (SEQ / ORDER are always CONFIRMED)
    source: str = "local"              # local | stage3 | rule | graph
    targeted: bool = False             # candidate from the robot the requester named (accept)

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
