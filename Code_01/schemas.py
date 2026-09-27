"""JSON schemas for the messages that flow through the pipeline.

Design notes
- JSON is only the *container*. The content of `action` etc. stays natural language,
  because the Auction (embedding similarity) and the Graph LLM read that text.
- The LLM never writes ids/orders/agent names. The code assigns them, so a whole
  class of format errors cannot happen.
- Step types (HELP is merged into NEED/PASS through `kind`):
    LOCAL                     : step done alone
    NEED  + kind="item"       : I need an item from someone
    NEED  + kind="task"       : I need someone to do a task for me   (= "HELP" request)
    PASS  + kind="item"       : I hand an item to someone
    PASS  + kind="task"       : I volunteer to do a task for someone (= "HELP" volunteer)
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

StepType = Literal["LOCAL", "NEED", "PASS"]
Kind = Literal["item", "task"]


# --------------------------------------------------------------------------- input
class AgentInput(BaseModel):
    """What one robot receives. It never leaves the agent (only Offer / Plan are broadcast)."""

    task: str
    capability: str
    images: list[str] = Field(default_factory=list)       # 2 images of the robot's own view
    hidden_info: list[str] = Field(default_factory=list)  # facts about places it cannot explore


class TaskConfig(BaseModel):
    """One scenario: a shared task + one entry per robot (agent_1, agent_2, ... in order)."""

    task: str
    agents: list[dict]  # each: {"capability": str, "images": [..], "hidden_info": [..]}

    def agent_inputs(self) -> dict[str, AgentInput]:
        return {f"agent_{i}": AgentInput(task=self.task, **a) for i, a in enumerate(self.agents, start=1)}


# --------------------------------------------------------------------------- offer
class OfferNeed(BaseModel):
    kind: Kind
    text: str


class RawOffer(BaseModel):
    """What the LLM writes."""

    capability: str
    obs_scope: list[str] = Field(default_factory=list)  # objects/areas actually visible or in hidden_info
    can_do: list[str] = Field(default_factory=list)
    cannot_do: list[str] = Field(default_factory=list)
    can_provide: list[str] = Field(default_factory=list)
    needs: list[OfferNeed] = Field(default_factory=list)


class Offer(RawOffer):
    """What gets broadcast (agent id added by the code)."""

    agent: str


# --------------------------------------------------------------------------- local plan
class RawStep(BaseModel):
    """One step as the LLM writes it."""

    type: StepType
    action: str = Field(min_length=1)
    kind: Optional[Kind] = None
    item: Optional[str] = None
    target: Optional[str] = None  # non-binding hint; the Auction decides the real match

    @model_validator(mode="after")
    def _check_collab_fields(self):
        if self.type == "LOCAL":
            self.kind = self.item = self.target = None  # meaningless for LOCAL, clear silently
            return self
        if self.kind is None:
            raise ValueError(f"{self.type} step needs 'kind' (item|task): {self.action!r}")
        if self.kind == "item" and not self.item:
            raise ValueError(f"{self.type}/item step needs 'item': {self.action!r}")
        return self


class RawLocalPlan(BaseModel):
    # No min_length: a robot with no relevant action for this task (e.g. a fixed kitchen robot
    # for a "clear the living room" task) should be able to legitimately return zero steps
    # rather than being forced to invent a LOCAL/NEED/PASS step just to satisfy a schema minimum.
    steps: list[RawStep] = Field(default_factory=list)


class Step(RawStep):
    id: str     # "<agent number>-<order>", e.g. "1-3"
    order: int  # execution order inside this agent (an ORDER constraint, not a clock time)


class LocalPlan(BaseModel):
    agent: str
    steps: list[Step]

    @classmethod
    def from_raw(cls, agent: str, raw: RawLocalPlan, known_agents: set[str]) -> "LocalPlan":
        num = agent.split("_")[-1]
        steps: list[Step] = []
        for i, s in enumerate(raw.steps, start=1):
            data = s.model_dump()
            # target is only a hint: drop it if it points to nobody real (or to self)
            if data["target"] is not None and (data["target"] not in known_agents or data["target"] == agent):
                data["target"] = None
            steps.append(Step(id=f"{num}-{i}", order=i, **data))
        return cls(agent=agent, steps=steps)

    def collaboration_steps(self) -> list[Step]:
        return [s for s in self.steps if s.type != "LOCAL"]


def agent_of(step_id: str) -> str:
    """'3-2' -> 'agent_3'"""
    return f"agent_{step_id.split('-')[0]}"


def describe_config(config: TaskConfig) -> str:
    """Human-readable dump of what will actually be sent to the LLM. Print this right after
    building a TaskConfig (CLI and notebook both do) — a wrong result is very often a leftover
    default value here (e.g. hidden_info still describing something not in the real images)
    rather than a pipeline bug, and this catches that before any API call is made."""
    lines = [f"TASK: {config.task}", ""]
    for i, a in enumerate(config.agents, start=1):
        lines.append(f"agent_{i}")
        lines.append(f"  capability : {a.get('capability', '')}")
        lines.append(f"  images     : {a.get('images', [])}")
        hidden = a.get("hidden_info", [])
        lines.append(f"  hidden_info: {hidden if hidden else '(none)'}")
    return "\n".join(lines)
