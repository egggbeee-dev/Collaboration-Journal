"""JSON schemas for the messages that flow through the pipeline.

Stage 1 - OFFER
    Each robot independently describes:
    - what it can do
    - what it cannot do
    - what it can providef
    - what it needs

Stage 2 - LOCAL PLANNING
    Each robot constructs a local plan using:
    - LOCAL
    - ASK_HELP
    - HELP
    - RECEIVE
    - PASS

Stage 3 - AUCTION
    Candidate collaboration edges are generated:
        ASK_HELP <-> HELP
        RECEIVE  <-> PASS

The Auction and Graph Reasoning stages determine
the final collaboration relations.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


# ===========================================================================
# Basic types
# ===========================================================================

StepType = Literal[
    "LOCAL",
    "ASK_HELP",
    "HELP",
    "RECEIVE",
    "PASS",
]

Kind = Literal[
    "item",
    "task",
]


# ===========================================================================
# Input
# ===========================================================================

class AgentInput(BaseModel):
    """What one robot receives.

    Raw images and hidden information remain private to the robot.
    Only structured Offer / LocalPlan information is broadcast.
    """

    task: str

    capability: str

    images: list[str] = Field(
        default_factory=list
    )

    hidden_info: list[str] = Field(
        default_factory=list
    )


class TaskConfig(BaseModel):
    """One shared-task scenario.

    Example:
        {
            "task": "...",
            "agents": [
                {
                    "capability": "...",
                    "images": [...],
                    "hidden_info": [...]
                }
            ]
        }
    """

    task: str

    agents: list[dict]

    def agent_inputs(self) -> dict[str, AgentInput]:
        return {
            f"agent_{i}": AgentInput(
                task=self.task,
                **agent,
            )
            for i, agent in enumerate(
                self.agents,
                start=1,
            )
        }


# ===========================================================================
# Offer
# ===========================================================================

class ObsObject(BaseModel):
    """An object or area observed by the robot."""

    object: str

    location: str

    state: Optional[str] = None


class CanDoAction(BaseModel):
    """A concrete action the robot can physically perform."""

    action: str

    object: Optional[str] = None

    location: Optional[str] = None

    target: Optional[str] = None


class CannotDoAction(BaseModel):
    """A task-relevant action the robot cannot perform."""

    action: str

    object: Optional[str] = None

    location: Optional[str] = None

    reason: str


class CanProvide(BaseModel):
    """A task or physical item that the robot can provide.

    type="task"
        A task that another robot could request this robot to perform.

    type="item"
        A physical item/resource that another robot could receive.
    """

    type: Kind

    object: Optional[str] = None

    location: Optional[str] = None

    action: Optional[str] = None

    @model_validator(mode="after")
    def _validate_provide(self):
        # ---------------------------------------------------------------
        # TASK
        # ---------------------------------------------------------------
        if self.type == "task":
            if not self.action:
                raise ValueError(
                    "CanProvide(type='task') requires 'action'."
                )

            # A task does not need a physical object.
            self.object = None

            return self

        # ---------------------------------------------------------------
        # ITEM
        # ---------------------------------------------------------------
        if self.type == "item":
            if not self.object:
                raise ValueError(
                    "CanProvide(type='item') requires 'object'."
                )

            return self

        return self


class OfferNeed(BaseModel):
    """A genuine dependency declared in an Offer.

    `kind="task"`:
        The robot needs another robot to perform a task.

    `kind="item"`:
        The robot needs another robot to provide an item.

    `text` is kept for backward compatibility with Auction / logging code.
    """

    kind: Kind

    # Backward-compatible canonical text representation.
    text: Optional[str] = None

    # Structured fields used by the new Offer format.
    object: Optional[str] = None
    location: Optional[str] = None
    action: Optional[str] = None

    @model_validator(mode="after")
    def _normalize(self):
        # ---------------------------------------------------------------
        # If the new format provides action, use it as the canonical text.
        # ---------------------------------------------------------------
        if not self.text:
            if self.action:
                self.text = self.action
            elif self.object:
                self.text = self.object

        # ---------------------------------------------------------------
        # If the old format provides text only, keep it usable by the
        # new structured representation as an action.
        # ---------------------------------------------------------------
        if not self.action and self.text:
            self.action = self.text

        # ---------------------------------------------------------------
        # Required semantic content
        # ---------------------------------------------------------------
        if not self.text:
            raise ValueError(
                f"OfferNeed(kind={self.kind!r}) requires "
                f"'text', 'action', or 'object'."
            )

        # ---------------------------------------------------------------
        # Task need
        # ---------------------------------------------------------------
        if self.kind == "task":
            self.object = None

        # ---------------------------------------------------------------
        # Item need
        # ---------------------------------------------------------------
        elif self.kind == "item":
            if not self.object:
                # For backward compatibility, allow the text itself
                # to serve as the item description.
                self.object = self.text

        return self

class RawOffer(BaseModel):
    """What the Offer-generation LLM writes."""

    capability: str

    obs_scope: list[ObsObject] = Field(
        default_factory=list
    )

    can_do: list[CanDoAction] = Field(
        default_factory=list
    )

    cannot_do: list[CannotDoAction] = Field(
        default_factory=list
    )

    can_provide: list[CanProvide] = Field(
        default_factory=list
    )

    needs: list[OfferNeed] = Field(
        default_factory=list
    )


class Offer(RawOffer):
    """Broadcast Offer.

    `agent` is assigned by code, not by the LLM.
    """

    agent: str


# ===========================================================================
# Local Plan
# ===========================================================================

class RawStep(BaseModel):
    """One local-plan step generated by the LLM.

    Step semantics:

    LOCAL
        Perform an action independently.

    ASK_HELP
        Request another robot to perform a task.

    HELP
        Perform a task requested by another robot.

    RECEIVE
        Receive a physical item from another robot.

    PASS
        Provide/pass a physical item to another robot.
    """

    type: StepType

    action: str = Field(
        min_length=1
    )

    kind: Optional[Kind] = None

    item: Optional[str] = None

    target: Optional[str] = None

    # Preferred target only.
    # Final assignment is determined by Auction / Graph Reasoning.

    @model_validator(mode="after")
    def _validate_step(self):

        # ---------------------------------------------------------------
        # LOCAL
        # ---------------------------------------------------------------

        if self.type == "LOCAL":

            self.kind = None
            self.item = None
            self.target = None

            return self

        # ---------------------------------------------------------------
        # ASK_HELP / HELP
        # ---------------------------------------------------------------

        if self.type in {
            "ASK_HELP",
            "HELP",
        }:

            if self.kind is None:
                self.kind = "task"

            if self.kind != "task":
                raise ValueError(
                    f"{self.type} must use kind='task': "
                    f"{self.action!r}"
                )

            # Task-level collaboration does not carry a physical item.
            self.item = None

            return self

        # ---------------------------------------------------------------
        # RECEIVE / PASS
        # ---------------------------------------------------------------

        if self.type in {
            "RECEIVE",
            "PASS",
        }:

            if self.kind is None:
                self.kind = "item"

            if self.kind != "item":
                raise ValueError(
                    f"{self.type} must use kind='item': "
                    f"{self.action!r}"
                )

            if not self.item:
                raise ValueError(
                    f"{self.type} requires an item: "
                    f"{self.action!r}"
                )

            return self

        return self


class RawLocalPlan(BaseModel):
    """Raw local plan generated by one robot.

    Empty plans are valid.
    """

    steps: list[RawStep] = Field(
        default_factory=list
    )


class Step(RawStep):
    """Validated step with code-assigned ID and order."""

    id: str

    order: int


class LocalPlan(BaseModel):
    """Validated local plan for one robot."""

    agent: str

    steps: list[Step]

    @classmethod
    def from_raw(
        cls,
        agent: str,
        raw: RawLocalPlan,
        known_agents: set[str],
    ) -> "LocalPlan":

        num = agent.split("_")[-1]

        steps: list[Step] = []

        for i, s in enumerate(
            raw.steps,
            start=1,
        ):

            data = s.model_dump()

            # -----------------------------------------------------------
            # Target is only a preferred hint.
            #
            # Remove invalid targets and self-targets.
            # -----------------------------------------------------------

            target = data.get("target")

            if (
                target is not None
                and (
                    target not in known_agents
                    or target == agent
                )
            ):
                data["target"] = None

            # -----------------------------------------------------------
            # IDs and order are assigned by code.
            # The LLM never controls these fields.
            # -----------------------------------------------------------

            steps.append(
                Step(
                    id=f"{num}-{i}",
                    order=i,
                    **data,
                )
            )

        return cls(
            agent=agent,
            steps=steps,
        )

    # -------------------------------------------------------------------
    # Collaboration steps
    # -------------------------------------------------------------------

    def collaboration_steps(self) -> list[Step]:
        return [
            s
            for s in self.steps
            if s.type != "LOCAL"
        ]

    # -------------------------------------------------------------------
    # Requests
    # -------------------------------------------------------------------

    def request_steps(self) -> list[Step]:
        """Steps that request another robot."""

        return [
            s
            for s in self.steps
            if s.type in {
                "ASK_HELP",
                "RECEIVE",
            }
        ]

    # -------------------------------------------------------------------
    # Providers
    # -------------------------------------------------------------------

    def provider_steps(self) -> list[Step]:
        """Steps that provide help/items to another robot."""

        return [
            s
            for s in self.steps
            if s.type in {
                "HELP",
                "PASS",
            }
        ]

    # -------------------------------------------------------------------
    # Task-level collaboration
    # -------------------------------------------------------------------

    def ask_help_steps(self) -> list[Step]:
        return [
            s
            for s in self.steps
            if s.type == "ASK_HELP"
        ]

    def help_steps(self) -> list[Step]:
        return [
            s
            for s in self.steps
            if s.type == "HELP"
        ]

    # -------------------------------------------------------------------
    # Item-level collaboration
    # -------------------------------------------------------------------

    def receive_steps(self) -> list[Step]:
        return [
            s
            for s in self.steps
            if s.type == "RECEIVE"
        ]

    def pass_steps(self) -> list[Step]:
        return [
            s
            for s in self.steps
            if s.type == "PASS"
        ]


# ===========================================================================
# Utility
# ===========================================================================

def agent_of(step_id: str) -> str:
    """Convert a step ID to its agent ID.

    Example:
        "3-2" -> "agent_3"
    """

    return f"agent_{step_id.split('-')[0]}"


# ===========================================================================
# Debug / configuration description
# ===========================================================================

def describe_config(
    config: TaskConfig,
) -> str:
    """Human-readable dump of the actual scenario sent to the LLM."""

    lines = [
        f"TASK: {config.task}",
        "",
    ]

    for i, agent in enumerate(
        config.agents,
        start=1,
    ):

        lines.append(
            f"agent_{i}"
        )

        lines.append(
            f"  capability : "
            f"{agent.get('capability', '')}"
        )

        lines.append(
            f"  images     : "
            f"{agent.get('images', [])}"
        )

        hidden = agent.get(
            "hidden_info",
            [],
        )

        lines.append(
            f"  hidden_info: "
            f"{hidden if hidden else '(none)'}"
        )

    return "\n".join(lines)
