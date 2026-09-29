"""Data schemas for the pipeline (no candidate-edge stage).

    1. Offer          each robot, LLM   -> what I can do / what I have / what I need
    2. Local Plan     each robot, LLM   -> my own steps (nodes), incl. volunteering
    3. Graph Reasoning central          -> edges between steps + joint-plan repair
    4. Joint Plan     template          -> rendered schedule

Step types (written by robots in Local Planning):
    LOCAL     I do this myself.
    ASK_HELP  My body cannot do a task that my part needs  -> another robot does it.
    HELP      I do a task that another robot cannot do.
    RECEIVE   My part needs an item that is not in my room.
    PASS      I give one of my items to another robot.

Pairs (edges are always provider -> requester):
    HELP -> ASK_HELP        PASS -> RECEIVE
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


StepType = Literal["LOCAL", "ASK_HELP", "HELP", "RECEIVE", "PASS"]
Kind = Literal["item", "task"]

REQUEST_TYPES = {"ASK_HELP", "RECEIVE"}
PROVIDER_TYPES = {"HELP", "PASS"}
PAIR = {"ASK_HELP": "HELP", "RECEIVE": "PASS"}          # request type -> provider type


# =============================================================================
# Input
# =============================================================================

class AgentInput(BaseModel):
    """Private input of one robot."""

    task: str
    capability: str
    images: list[str] = Field(default_factory=list)
    hidden_info: list[str] = Field(default_factory=list)


class TaskConfig(BaseModel):
    """Scenario file: shared task + one entry per robot (agent_1, agent_2, ...)."""

    task: str
    agents: list[dict]

    def agent_inputs(self) -> dict[str, AgentInput]:
        return {
            f"agent_{i}": AgentInput(task=self.task, **a)
            for i, a in enumerate(self.agents, start=1)
        }


# =============================================================================
# Stage 1 - Offer
# =============================================================================

class ObsObject(BaseModel):
    object: str
    location: Optional[str] = None
    state: Optional[str] = None


class CanDo(BaseModel):
    action: str
    object: Optional[str] = None


class CannotDo(BaseModel):
    action: str
    object: Optional[str] = None
    reason: Optional[str] = None


class HasItem(BaseModel):
    """A movable object in my room that another robot could receive."""

    object: str
    location: Optional[str] = None


class Need(BaseModel):
    """Something I need from another robot to do my part.

    kind="item": an object that is not in my room.
    kind="task": a task my body cannot do.
    `id` is assigned by code (e.g. "N3-1") so other robots can refer to it.
    """

    kind: Kind
    what: str = Field(min_length=1)
    id: Optional[str] = None


class RawOffer(BaseModel):
    """What the Offer LLM writes."""

    reasoning: str = ""
    capability: str
    obs_scope: list[ObsObject] = Field(default_factory=list)
    can_do: list[CanDo] = Field(default_factory=list)
    cannot_do: list[CannotDo] = Field(default_factory=list)
    has_items: list[HasItem] = Field(default_factory=list)
    need_from_others: list[Need] = Field(default_factory=list)


class Offer(RawOffer):
    agent: str


# Offer fields that never leave the robot.
PRIVATE_OFFER_FIELDS = {"reasoning", "obs_scope", "cannot_do"}


def public_offer(offer: Offer) -> dict:
    return offer.model_dump(exclude=PRIVATE_OFFER_FIELDS)


# =============================================================================
# Stage 2 - Local Plan
# =============================================================================

class RawStep(BaseModel):
    type: StepType
    action: str = Field(min_length=1)
    item: Optional[str] = None      # RECEIVE / PASS only
    target: Optional[str] = None    # collaboration partner hint (agent id), never binding

    @model_validator(mode="after")
    def _check(self):
        if self.type in {"RECEIVE", "PASS"}:
            if not self.item:
                raise ValueError(f"{self.type} requires `item`: {self.action!r}")
        else:
            self.item = None
        if self.type == "LOCAL":
            self.target = None
        return self


class RawLocalPlan(BaseModel):
    reasoning: str = ""
    steps: list[RawStep] = Field(default_factory=list)


class Step(RawStep):
    id: str                                      # "3-2" = agent_3, step 2 (code-assigned)
    order: int
    origin: Literal["plan", "graph"] = "plan"    # "graph" = added by Graph Reasoning


class LocalPlan(BaseModel):
    agent: str
    reasoning: str = ""
    steps: list[Step]

    @classmethod
    def from_raw(cls, agent: str, raw: RawLocalPlan, known_agents: set[str]) -> "LocalPlan":
        num = agent.split("_")[-1]
        steps = []
        for i, s in enumerate(raw.steps, start=1):
            data = s.model_dump()
            if data["target"] is not None and (data["target"] not in known_agents or data["target"] == agent):
                data["target"] = None
            steps.append(Step(id=f"{num}-{i}", order=i, **data))
        return cls(agent=agent, reasoning=raw.reasoning, steps=steps)

    def public(self) -> dict:
        """What is broadcast (the robot's reasoning stays private)."""
        return self.model_dump(exclude={"reasoning"})


# =============================================================================
# Utilities
# =============================================================================

def agent_of(step_id: str) -> str:
    """'3-2' -> 'agent_3'."""
    return f"agent_{step_id.split('-')[0]}"


def robot_label(agent: str) -> str:
    """'agent_3' -> 'R3'."""
    suffix = agent.split("_")[-1]
    return f"R{suffix}" if suffix.isdigit() else agent


def describe_config(config: TaskConfig) -> str:
    lines = [f"TASK: {config.task}", ""]
    for i, a in enumerate(config.agents, start=1):
        lines += [
            f"agent_{i}",
            f"  capability : {a.get('capability', '')}",
            f"  images     : {a.get('images', [])}",
            f"  hidden_info: {a.get('hidden_info') or '(none)'}",
        ]
    return "\n".join(lines)
