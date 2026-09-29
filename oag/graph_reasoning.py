"""
Stage 4 - GRAPH REASONING

Centralized global coordination over independently generated local plans.

Input:
    - Local Plans (incl. HELP / PASS steps self-proposed in the Auction stage)
    - ALL Auction candidate edges (provider_step -> request_step)

Layer 1 (rules) makes the initial 1:1 edge selection and verifies the graph.
Layer 2 (LLM) may reassign / unmatch / add_order among existing candidates.

Step-type semantics:
    LOCAL
        Execute an action locally.

    ASK_HELP
        Request another robot to perform a task.

    HELP
        Volunteer to perform a task requested by another robot.

    RECEIVE
        Request / receive a physical item from another robot.

    PASS
        Provide / pass a physical item to another robot.

Matching channels:
    ASK_HELP <-> HELP
    RECEIVE  <-> PASS

The Auction only generates candidate coordination edges (self-proposed by
the providing robots). Graph Reasoning selects among them and determines the
final dependency structure.

Allowed LLM operations:
    reassign
        Replace the current provider of a request with another
        Auction-generated candidate.

    unmatch
        Remove an incorrect collaboration relation.

    add_order
        Add a cross-robot ordering dependency.

The LLM cannot:
    - create new steps
    - delete steps
    - rewrite actions
    - create collaboration with a robot that was never a candidate
    - access private observations/images
"""

from __future__ import annotations

import copy
import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

from auction import AuctionResult
from llm import BaseLLM
from runtime import EventLog, call_validated
from schemas import LocalPlan, Step, agent_of


# ============================================================================
# Graph
# ============================================================================

@dataclass
class Node:
    id: str
    agent: str
    order: int
    type: str
    kind: Optional[str]
    action: str
    item: Optional[str]
    active: bool = True
    purpose: Optional[str] = None
    origin: str = "plan"
    serves: Optional[str] = None
    # Execution position inside the robot's own sequence.
    # = order for the robot's own steps; for proposed HELP / PASS steps it is
    # chosen by the rule layer (place_proposed_steps), i.e. WHEN, never WHAT.
    position: float = 0.0


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    kind: str
    status: str = "confirmed"
    # kind:
    #   sequence
    #   collaboration
    #
    # status:
    #   confirmed
    #   candidate


def item_key(node: Node) -> Optional[tuple[str, str]]:
    """A physical item belongs to one robot and can be passed only once."""
    if node.type != "PASS" or not node.item:
        return None
    return (node.agent, node.item.strip().lower())


def initial_edge_selection(
    nodes: dict[str, Node],
    candidates: dict[str, list[dict]],
) -> list[tuple[str, str, float]]:
    """Rule-based initial selection over self-proposed candidates.

    Greedy by score, subject to:
        - one provider per request
        - one request per provider step
        - one PASS per (robot, item): an item can only be passed once

    1:N for tasks is already expressed as several HELP steps (one per
    request) chained in the provider's own sequence, so no special case
    is needed here.
    """

    ranked = sorted(
        (
            (c["score"], request_id, c["pass"])
            for request_id, cand_list in candidates.items()
            for c in cand_list
            if request_id in nodes and c["pass"] in nodes
        ),
        key=lambda x: (-x[0], x[1], x[2]),
    )

    used_requests: set[str] = set()
    used_providers: set[str] = set()
    used_items: set[tuple[str, str]] = set()

    selected: list[tuple[str, str, float]] = []

    for score, request_id, provider_id in ranked:

        if request_id in used_requests or provider_id in used_providers:
            continue

        key = item_key(nodes[provider_id])

        if key is not None and key in used_items:
            continue

        used_requests.add(request_id)
        used_providers.add(provider_id)

        if key is not None:
            used_items.add(key)

        selected.append((request_id, provider_id, float(score)))

    return selected


class PlanGraph:

    def __init__(
        self,
        task: str,
        plans: dict[str, LocalPlan],
        auction: AuctionResult,
    ) -> None:

        self.task = task

        # ------------------------------------------------------------
        # Nodes
        # ------------------------------------------------------------

        self.nodes: dict[str, Node] = {}

        for aid, plan in plans.items():
            for step in plan.steps:
                self.nodes[step.id] = Node(
                    id=step.id,
                    agent=aid,
                    order=step.order,
                    type=step.type,
                    kind=step.kind,
                    action=step.action,
                    item=step.item,
                    active=True,
                    purpose=step.purpose,
                    origin=step.origin,
                    serves=step.serves,
                    position=float(step.order),
                )

        # ------------------------------------------------------------
        # ALL candidate edges generated by the Auction.
        # ------------------------------------------------------------

        self.candidates: dict[str, list[dict]] = auction.candidates

        # ------------------------------------------------------------
        # Layer 1 - initial edge selection (rule-based)
        #
        # request -> provider
        #
        # provider can be HELP or PASS
        # request can be ASK_HELP or RECEIVE
        # ------------------------------------------------------------

        self.handoff: dict[str, str] = {}
        self.handoff_score: dict[str, float] = {}

        for request_id, provider_id, score in initial_edge_selection(
            self.nodes,
            self.candidates,
        ):
            self.handoff[request_id] = provider_id
            self.handoff_score[request_id] = score

        # Additional cross-robot ordering constraints.
        self.extra_collaboration: list[tuple[str, str]] = []

        # ------------------------------------------------------------
        # Provider activation
        #
        # A HELP/PASS step that was not selected by the initial
        # rule-based selection is initially inactive.
        #
        # However, it remains in the graph and can be reactivated
        # later by Graph Reasoning through `reassign`.
        # ------------------------------------------------------------

        matched_providers = set(self.handoff.values())

        for node in self.nodes.values():

            if node.type in {"HELP", "PASS"}:
                node.active = node.id in matched_providers

    # ----------------------------------------------------------------
    # Item uniqueness
    # ----------------------------------------------------------------

    def item_in_use(
        self,
        provider_id: str,
        ignore_request: Optional[str] = None,
    ) -> Optional[str]:
        """If provider_id is a PASS whose (robot, item) is already passed
        through another selected handoff, return that request id."""

        key = item_key(self.nodes[provider_id])

        if key is None:
            return None

        for rid, pid in self.handoff.items():
            if rid == ignore_request or pid == provider_id:
                continue
            if pid in self.nodes and item_key(self.nodes[pid]) == key:
                return rid

        return None

    # ----------------------------------------------------------------
    # Views
    # ----------------------------------------------------------------

    def active_ids(self) -> list[str]:
        return [
            node_id
            for node_id, node in self.nodes.items()
            if node.active
        ]

    def sort_key(self, node_id: str):
        node = self.nodes[node_id]

        try:
            agent_num = int(node.agent.split("_")[-1])
        except ValueError:
            agent_num = 0

        return (
            agent_num,
            node.order,
        )

    def agent_sequence(self, agent: str) -> list[Node]:

        return sorted(
            (
                node
                for node in self.nodes.values()
                if node.agent == agent and node.active
            ),
            key=lambda node: (node.position, node.order),
        )

    # ----------------------------------------------------------------
    # Edges
    # ----------------------------------------------------------------

    def edges(
        self,
        include_candidates: bool = False,
    ) -> list[Edge]:

        out: list[Edge] = []

        # ------------------------------------------------------------
        # 1. Intra-robot sequence dependencies
        # ------------------------------------------------------------

        agents = sorted(
            {node.agent for node in self.nodes.values()}
        )

        for agent in agents:

            sequence = self.agent_sequence(agent)

            for a, b in zip(sequence, sequence[1:]):

                out.append(
                    Edge(
                        src=a.id,
                        dst=b.id,
                        kind="sequence",
                        status="confirmed",
                    )
                )

        # ------------------------------------------------------------
        # 2. Confirmed collaboration dependencies
        #
        # provider -> requester
        #
        # Example:
        #
        # R2 HELP
        #      |
        #      v
        # R3 ASK_HELP
        # ------------------------------------------------------------

        for request_id, provider_id in sorted(
            self.handoff.items()
        ):

            if request_id not in self.nodes:
                continue

            if provider_id not in self.nodes:
                continue

            request_node = self.nodes[request_id]
            provider_node = self.nodes[provider_id]

            if not request_node.active:
                continue

            if not provider_node.active:
                continue

            out.append(
                Edge(
                    src=provider_id,
                    dst=request_id,
                    kind="collaboration",
                    status="confirmed",
                )
            )

        # ------------------------------------------------------------
        # 3. Additional cross-robot ordering constraints
        # ------------------------------------------------------------

        for src, dst in self.extra_collaboration:

            if src not in self.nodes or dst not in self.nodes:
                continue

            if not self.nodes[src].active:
                continue

            if not self.nodes[dst].active:
                continue

            out.append(
                Edge(
                    src=src,
                    dst=dst,
                    kind="collaboration",
                    status="confirmed",
                )
            )

        # ------------------------------------------------------------
        # 4. Unresolved Auction candidate edges
        #
        # They are visible to the LLM reasoner but do not affect
        # topological ordering until confirmed.
        # ------------------------------------------------------------

        if include_candidates:

            confirmed_pairs = {
                (provider_id, request_id)
                for request_id, provider_id
                in self.handoff.items()
            }

            for request_id, candidate_list in self.candidates.items():

                if request_id not in self.nodes:
                    continue

                if not self.nodes[request_id].active:
                    continue

                for candidate in candidate_list:

                    provider_id = candidate["pass"]

                    if provider_id not in self.nodes:
                        continue

                    if (
                        provider_id,
                        request_id,
                    ) in confirmed_pairs:
                        continue

                    out.append(
                        Edge(
                            src=provider_id,
                            dst=request_id,
                            kind="collaboration",
                            status="candidate",
                        )
                    )

        if not include_candidates:

            out = [
                edge
                for edge in out
                if edge.status == "confirmed"
            ]

        return out

    # ----------------------------------------------------------------
    # Snapshot / restore
    # ----------------------------------------------------------------

    def snapshot(self):

        return copy.deepcopy(
            (
                {
                    node_id: (node.active, node.position)
                    for node_id, node in self.nodes.items()
                },
                self.handoff,
                self.handoff_score,
                self.extra_collaboration,
            )
        )

    def restore(self, snapshot) -> None:

        (
            active,
            self.handoff,
            self.handoff_score,
            self.extra_collaboration,
        ) = copy.deepcopy(snapshot)

        for node_id, (is_active, position) in active.items():
            self.nodes[node_id].active = is_active
            self.nodes[node_id].position = position

    # ----------------------------------------------------------------
    # Serialization
    # ----------------------------------------------------------------

    def to_dict(self) -> dict:

        return {
            "nodes": [
                node.__dict__
                for node in sorted(
                    self.nodes.values(),
                    key=lambda node: self.sort_key(node.id),
                )
            ],

            "handoffs": [
                {
                    "need": request_id,
                    "pass": provider_id,
                    "score": self.handoff_score.get(request_id),
                }
                for request_id, provider_id
                in sorted(self.handoff.items())
            ],

            "extra_collaboration": [
                list(pair)
                for pair in self.extra_collaboration
            ],

            "edges": [
                edge.__dict__
                for edge in self.edges(
                    include_candidates=True
                )
            ],
        }


# ============================================================================
# Graph algorithms
# ============================================================================

def find_cycle(
    graph: PlanGraph,
) -> list[Edge] | None:

    adjacency: dict[str, list[Edge]] = defaultdict(list)

    for edge in graph.edges(
        include_candidates=False
    ):
        adjacency[edge.src].append(edge)

    color = {
        node_id: 0
        for node_id in graph.active_ids()
    }

    parent: dict[str, Edge] = {}

    def dfs(node_id: str):

        color[node_id] = 1

        for edge in adjacency[node_id]:

            next_id = edge.dst

            if color[next_id] == 0:

                parent[next_id] = edge

                result = dfs(next_id)

                if result:
                    return result

            elif color[next_id] == 1:

                cycle = [edge]
                current = node_id

                while current != next_id:

                    parent_edge = parent[current]
                    cycle.append(parent_edge)

                    current = parent_edge.src

                return cycle

        color[node_id] = 2

        return None

    for node_id in sorted(
        color,
        key=graph.sort_key,
    ):

        if color[node_id] == 0:

            result = dfs(node_id)

            if result:
                return result

    return None


def topological_levels(graph: PlanGraph):

    """
    Kahn's algorithm.

    Returns:
        (topological_order, level)

    or:
        None if the graph contains a cycle.
    """

    node_ids = graph.active_ids()

    indegree = {
        node_id: 0
        for node_id in node_ids
    }

    successors: dict[str, list[str]] = defaultdict(list)

    for edge in graph.edges(
        include_candidates=False
    ):

        successors[edge.src].append(edge.dst)
        indegree[edge.dst] += 1

    level = {
        node_id: 0
        for node_id in node_ids
    }

    ready = sorted(
        [
            node_id
            for node_id in node_ids
            if indegree[node_id] == 0
        ],
        key=graph.sort_key,
    )

    order: list[str] = []

    while ready:

        current = ready.pop(0)

        order.append(current)

        for next_id in successors[current]:

            level[next_id] = max(
                level[next_id],
                level[current] + 1,
            )

            indegree[next_id] -= 1

            if indegree[next_id] == 0:

                ready.append(next_id)

                ready.sort(
                    key=graph.sort_key
                )

    if len(order) != len(node_ids):
        return None

    return order, level


def reachable(
    graph: PlanGraph,
    source: str,
    target: str,
) -> bool:

    adjacency: dict[str, list[str]] = defaultdict(list)

    for edge in graph.edges(
        include_candidates=False
    ):
        adjacency[edge.src].append(edge.dst)

    visited = {source}
    stack = [source]

    while stack:

        current = stack.pop()

        if current == target:
            return True

        for next_id in adjacency[current]:

            if next_id not in visited:

                visited.add(next_id)
                stack.append(next_id)

    return False


# ============================================================================
# Placement of proposed HELP / PASS steps (scheduling = WHEN only)
# ============================================================================

def _schedule_length(graph: PlanGraph) -> int | None:
    result = topological_levels(graph)
    if result is None:
        return None
    _, level = result
    return (max(level.values()) + 1) if level else 0


def place_proposed_steps(graph: PlanGraph) -> list[dict]:
    """Decide where each ACTIVE proposed HELP / PASS step is executed inside
    its provider's own sequence.

    The robot decided WHAT (it volunteered the step); this only decides WHEN.

    For each proposed step (deterministic order), try every insertion slot in
    the provider's sequence and keep the one that
        1. creates no cycle, and
        2. gives the shortest overall schedule (number of Joint Plan steps),
        3. ties -> the latest slot, i.e. the provider's own work is
           interrupted only when that actually shortens the schedule.

    If no slot is acyclic, the step stays at the end and the cycle rule
    handles it.
    """

    # reset: every step back to its original order (proposals = end of plan)
    for node in graph.nodes.values():
        node.position = float(node.order)

    movable = sorted(
        (
            n for n in graph.nodes.values()
            if n.active and n.origin == "proposal"
        ),
        key=lambda n: graph.sort_key(n.id),
    )

    placements: list[dict] = []

    for node in movable:

        others = [
            m for m in graph.agent_sequence(node.agent)
            if m.id != node.id
        ]

        # candidate positions: before each other step, or at the very end
        slots: list[float] = []
        prev = None
        for m in others:
            lo = prev.position if prev is not None else m.position - 1.0
            slots.append((lo + m.position) / 2.0)
            prev = m
        slots.append((others[-1].position if others else 0.0) + 1.0)

        original = node.position
        best: tuple[int, int, float] | None = None   # (length, -slot_index, position)

        for idx, pos in enumerate(slots):
            node.position = pos
            if find_cycle(graph) is not None:
                continue
            length = _schedule_length(graph)
            if length is None:
                continue
            key = (length, -idx, pos)
            if best is None or key < best:
                best = key

        node.position = best[2] if best is not None else original

        if best is not None:
            before = next(
                (m.id for m in others if m.position > node.position),
                None,
            )
            placements.append({
                "rule": "proposal_placed",
                "step": node.id,
                "before": before,          # None = at the end of the robot's plan
                "schedule_length": best[0],
            })

    return placements


# ============================================================================
# Layer 1 - Rule verification
# ============================================================================

@dataclass
class RuleReport:

    fixes: list[dict] = field(
        default_factory=list
    )

    warnings: list[str] = field(
        default_factory=list
    )

    unresolved_needs: list[str] = field(
        default_factory=list
    )

    order: list[str] = field(
        default_factory=list
    )

    levels: dict[str, int] = field(
        default_factory=dict
    )

    # where accepted HELP / PASS proposals were placed (scheduling only)
    placements: list[dict] = field(
        default_factory=list
    )

    ok: bool = True

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def rule_verify(
    graph: PlanGraph,
) -> RuleReport:

    report = RuleReport()

    # ----------------------------------------------------------------
    # 1. Orphan providers
    #
    # HELP / PASS that are not used by any confirmed collaboration
    # are inactive.
    #
    # They are NOT deleted because they may still be reintroduced
    # by Graph Reasoning if they are valid candidates.
    # ----------------------------------------------------------------

    matched_providers = {
        provider_id
        for request_id, provider_id
        in graph.handoff.items()
        if (
            request_id in graph.nodes
            and graph.nodes[request_id].active
        )
    }

    for node in graph.nodes.values():

        if node.type not in {"HELP", "PASS"}:
            continue

        if node.id in matched_providers:
            node.active = True
            continue

        if node.active:

            node.active = False

            report.fixes.append(
                {
                    "rule": "orphan_provider_withdrawn",
                    "step": node.id,
                    "type": node.type,
                }
            )

    # ----------------------------------------------------------------
    # 1b. Place accepted proposals inside the provider's sequence
    #     (earliest-finishing, acyclic slot; WHEN only)
    # ----------------------------------------------------------------

    report.placements = place_proposed_steps(graph)

    # ----------------------------------------------------------------
    # 2. Cycle detection / breaking
    # ----------------------------------------------------------------

    while True:

        cycle = find_cycle(graph)

        if cycle is None:
            break

        collaboration_edges = [
            edge
            for edge in cycle
            if (
                edge.kind == "collaboration"
                and edge.status == "confirmed"
            )
        ]

        # ------------------------------------------------------------
        # Prefer removing the weakest confirmed handoff.
        # ------------------------------------------------------------

        handoff_edges = [
            edge
            for edge in collaboration_edges
            if edge.dst in graph.handoff
            and graph.handoff.get(edge.dst) == edge.src
        ]

        if handoff_edges:

            weakest = min(
                handoff_edges,
                key=lambda edge:
                    graph.handoff_score.get(
                        edge.dst,
                        0.0,
                    ),
            )

            request_id = weakest.dst
            provider_id = weakest.src

            del graph.handoff[request_id]

            graph.handoff_score.pop(
                request_id,
                None,
            )

            if provider_id in graph.nodes:
                graph.nodes[provider_id].active = False

            report.fixes.append(
                {
                    "rule": "cycle_broken",
                    "removed_handoff": [
                        provider_id,
                        request_id,
                    ],
                }
            )

            continue

        # ------------------------------------------------------------
        # Additional cross-robot ordering edge
        # ------------------------------------------------------------

        extra_edges = [
            edge
            for edge in collaboration_edges
            if (
                edge.src,
                edge.dst,
            ) in graph.extra_collaboration
        ]

        if extra_edges:

            edge = extra_edges[-1]

            graph.extra_collaboration.remove(
                (edge.src, edge.dst)
            )

            report.fixes.append(
                {
                    "rule": "cycle_broken",
                    "removed_order": [
                        edge.src,
                        edge.dst,
                    ],
                }
            )

            continue

        # ------------------------------------------------------------
        # If the cycle only exists inside one robot's sequence,
        # we cannot repair it here.
        # ------------------------------------------------------------

        report.ok = False

        report.warnings.append(
            "unbreakable cycle inside a single robot's own sequence"
        )

        return report

    # ----------------------------------------------------------------
    # 3. Unresolved collaboration requests
    #
    # ASK_HELP / RECEIVE without a confirmed provider.
    # ----------------------------------------------------------------

    report.unresolved_needs = [
        node.id
        for node in graph.nodes.values()
        if (
            node.active
            and node.type in {"ASK_HELP", "RECEIVE"}
            and node.id not in graph.handoff
        )
    ]

    # ----------------------------------------------------------------
    # 4. Item use-before-receive check
    # ----------------------------------------------------------------

    for request in graph.nodes.values():

        if not (
            request.active
            and request.type == "RECEIVE"
            and request.kind == "item"
            and request.item
        ):
            continue

        for node in graph.agent_sequence(
            request.agent
        ):

            if (
                node.type == "LOCAL"
                and node.order < request.order
                and request.item.lower()
                in node.action.lower()
            ):

                report.warnings.append(
                    f"{node.id} seems to use "
                    f"'{request.item}' before it is "
                    f"received at {request.id}"
                )

    # ----------------------------------------------------------------
    # 5. Topological schedule
    # ----------------------------------------------------------------

    result = topological_levels(graph)

    if result is None:

        report.ok = False

    else:

        report.order, report.levels = result

    return report


# ============================================================================
# Layer 2 - LLM Graph Reasoner
# ============================================================================

class GraphOp(BaseModel):

    op: Literal[
        "reassign",
        "unmatch",
        "add_order",
    ]

    # ------------------------------------------------------------
    # Reassign / unmatch
    # ------------------------------------------------------------

    need: Optional[str] = None

    to_pass: Optional[str] = None

    # ------------------------------------------------------------
    # Cross-robot ordering
    # ------------------------------------------------------------

    before: Optional[str] = None
    after: Optional[str] = None

    reason: str = ""

    @model_validator(mode="after")
    def validate_fields(self):

        required_fields = {
            "reassign": (
                "need",
                "to_pass",
            ),

            "unmatch": (
                "need",
            ),

            "add_order": (
                "before",
                "after",
            ),
        }[self.op]

        for field_name in required_fields:

            if not getattr(
                self,
                field_name,
            ):

                raise ValueError(
                    f"op '{self.op}' needs "
                    f"field '{field_name}'"
                )

        return self


class RawGraphOps(BaseModel):

    ops: list[GraphOp] = Field(
        default_factory=list
    )


# ============================================================================
# LLM prompt
# ============================================================================

GRAPH_SYSTEM = """
You are the centralized graph reasoner of a heterogeneous robot team.

You are NOT a planner.

Each robot independently generated its own local plan
(LOCAL / ASK_HELP / RECEIVE). After all plans were shared, each robot
volunteered HELP / PASS steps for other robots' requests; these are the
candidate providers. A rule-based pass already made an initial 1:1
selection among them.

Your job is to determine whether those relations form a globally
consistent dependency graph.

------------------------------------------------------------
STEP TYPES
------------------------------------------------------------

LOCAL
    A robot performs an action locally.

ASK_HELP
    A robot requests another robot to perform a task that its own
    embodiment cannot do. It waits until that task is done.
    `purpose` names the robot's own later step that depends on it.

HELP
    A robot volunteers to perform a task for another robot.

RECEIVE
    A robot requests / receives a physical item.
    `purpose` names the robot's own later step that uses it.

PASS
    A robot provides / passes a physical item.

Valid collaboration relations are ONLY:

    ASK_HELP <-> HELP

    RECEIVE <-> PASS

------------------------------------------------------------
CURRENT GRAPH
------------------------------------------------------------

A confirmed handoff is represented as:

    provider -> requester

For example:

    R2_HELP
        |
        v
    R3_ASK_HELP

Each request lists its candidate providers (self-proposed by robots).

Candidate providers are the ONLY providers that may be selected.
Inactive candidates are not in the robots' sequences until selected.
Selected HELP / PASS steps are placed by a rule inside the provider's
sequence at the slot that avoids cycles and shortens the schedule.
A physical item (same robot, same item) can be passed only once.

------------------------------------------------------------
AVAILABLE OPERATIONS
------------------------------------------------------------

1. reassign

Replace the current provider of a request.

Format:

{
    "op": "reassign",
    "need": "<request step id>",
    "to_pass": "<candidate provider step id>",
    "reason": "..."
}

The `to_pass` MUST already appear in the request's
candidate list.

It is not allowed to create a new collaboration relation.

The candidate may be either:

    HELP
    PASS

depending on the request type.

------------------------------------------------------------

2. unmatch

Remove an incorrect collaboration relation.

Format:

{
    "op": "unmatch",
    "need": "<request step id>",
    "reason": "..."
}

Use this when the confirmed provider does not actually
satisfy the request.

------------------------------------------------------------

3. add_order

Add a cross-robot ordering dependency.

Format:

{
    "op": "add_order",
    "before": "<step id>",
    "after": "<step id>",
    "reason": "..."
}

Use this only when one robot's action must happen before
another robot's action.

The two steps must belong to different robots.

Do not add an ordering relation if it is already implied
by the existing dependency graph.

------------------------------------------------------------
IMPORTANT CONSTRAINTS
------------------------------------------------------------

You may NOT:

- create a new step
- delete a step
- rewrite an action
- change a robot's action
- invent a provider
- create a collaboration that is not in the Auction candidates
- access private observations or images
- perform task planning from scratch

You may ONLY modify:

- collaboration relations
- provider assignment
- cross-robot ordering relations

------------------------------------------------------------
WHAT TO CHECK
------------------------------------------------------------

For every ASK_HELP:

    Does the selected HELP provider actually perform
    the requested task?

For every RECEIVE:

    Does the selected PASS provider actually provide
    the requested item?

Also check:

- candidate alternatives
- collaboration consistency
- dependency ordering
- cycles
- use-before-receive problems
- unnecessary cross-robot dependencies

Prefer minimal intervention.

If the graph is already consistent:

    {"ops": []}

Return exactly ONE JSON object:

{
    "ops": [...]
}
"""


# ============================================================================
# LLM serialization
# ============================================================================

def serialize_for_llm(
    graph: PlanGraph,
    report: RuleReport,
    scope: str = "full",
) -> str:

    def brief(node: Node) -> dict:

        data = {
            "id": node.id,
            "type": node.type,
            "action": node.action,
        }

        if node.kind:
            data["kind"] = node.kind

        if node.item:
            data["item"] = node.item

        if node.purpose:
            data["purpose"] = node.purpose

        if node.serves:
            data["serves"] = node.serves

        return data

    # ------------------------------------------------------------
    # Robot-local plans
    # ------------------------------------------------------------

    robots: dict[str, list[dict]] = {}

    for agent in sorted(
        {
            node.agent
            for node in graph.nodes.values()
        }
    ):

        sequence = graph.agent_sequence(
            agent
        )

        robots[agent] = [
            brief(node)
            for node in sequence
            if (
                scope == "full"
                or node.type != "LOCAL"
            )
        ]

    # ------------------------------------------------------------
    # Candidate edges
    # ------------------------------------------------------------

    candidates = {}

    for request_id, candidate_list in graph.candidates.items():

        if request_id not in graph.nodes:
            continue

        if not graph.nodes[request_id].active:
            continue

        formatted = []

        for candidate in candidate_list:

            provider_id = candidate["pass"]

            if provider_id not in graph.nodes:
                continue

            provider = graph.nodes[provider_id]

            used_by_other = (
                provider_id in graph.handoff.values()
                and graph.handoff.get(request_id)
                != provider_id
            )

            formatted.append(
                {
                    "pass": provider_id,
                    "score": candidate["score"],
                    "type": provider.type,
                    "action": provider.action,
                    "item": provider.item,
                    "used_by_other_need": used_by_other,
                }
            )

        candidates[request_id] = formatted

    # ------------------------------------------------------------
    # Confirmed handoffs
    # ------------------------------------------------------------

    handoffs = []

    for request_id, provider_id in sorted(
        graph.handoff.items()
    ):

        if request_id not in graph.nodes:
            continue

        if provider_id not in graph.nodes:
            continue

        if not graph.nodes[request_id].active:
            continue

        if not graph.nodes[provider_id].active:
            continue

        handoffs.append(
            {
                "need": request_id,
                "pass": provider_id,
                "need_type": graph.nodes[request_id].type,
                "pass_type": graph.nodes[provider_id].type,
                "score": graph.handoff_score.get(
                    request_id
                ),
            }
        )

    # ------------------------------------------------------------
    # Collaboration graph
    # ------------------------------------------------------------

    collaboration_edges = []

    for edge in graph.edges(
        include_candidates=True
    ):

        if edge.kind != "collaboration":
            continue

        collaboration_edges.append(
            {
                "src": edge.src,
                "dst": edge.dst,
                "status": edge.status,
            }
        )

    # ------------------------------------------------------------
    # Final payload
    # ------------------------------------------------------------

    payload = {
        "task": graph.task,

        "robots": robots,

        "handoffs": handoffs,

        "collaboration_edges": collaboration_edges,

        "candidates": candidates,

        "unresolved_requests": report.unresolved_needs,

        "rule_warnings": report.warnings,
    }

    return json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
    )


# ============================================================================
# Apply Graph Operation
# ============================================================================

def apply_op(
    graph: PlanGraph,
    op: GraphOp,
) -> tuple[bool, str]:

    nodes = graph.nodes

    snapshot = graph.snapshot()

    # ========================================================================
    # REASSIGN
    # ========================================================================

    if op.op == "reassign":

        request_id = op.need
        provider_id = op.to_pass

        if request_id not in nodes:

            return (
                False,
                "unknown request step",
            )

        request = nodes[request_id]

        # ------------------------------------------------------------
        # Request type validation
        # ------------------------------------------------------------

        if request.type not in {
            "ASK_HELP",
            "RECEIVE",
        }:

            return (
                False,
                "step is not a collaboration request",
            )

        if not request.active:

            return (
                False,
                "request step is inactive",
            )

        # ------------------------------------------------------------
        # Provider validation
        # ------------------------------------------------------------

        if provider_id not in nodes:

            return (
                False,
                "unknown provider step",
            )

        provider = nodes[provider_id]

        expected_provider_type = {
            "ASK_HELP": "HELP",
            "RECEIVE": "PASS",
        }[request.type]

        if provider.type != expected_provider_type:

            return (
                False,
                (
                    f"invalid provider type: "
                    f"{request.type} requires "
                    f"{expected_provider_type}"
                ),
            )

        # ------------------------------------------------------------
        # Candidate validation
        #
        # VERY IMPORTANT:
        # Graph Reasoning cannot invent a new collaboration.
        # ------------------------------------------------------------

        candidate_scores = {
            candidate["pass"]: candidate["score"]
            for candidate in graph.candidates.get(
                request_id,
                [],
            )
        }

        if provider_id not in candidate_scores:

            return (
                False,
                (
                    "provider is not an Auction-generated "
                    "candidate for this request"
                ),
            )

        # ------------------------------------------------------------
        # Same robot is not allowed
        # ------------------------------------------------------------

        if (
            provider.agent
            == request.agent
        ):

            return (
                False,
                "request and provider belong to the same robot",
            )

        # ------------------------------------------------------------
        # Provider already serving another request
        # ------------------------------------------------------------

        current_request = None

        for rid, pid in graph.handoff.items():

            if (
                pid == provider_id
                and rid != request_id
            ):

                current_request = rid
                break

        if current_request is not None:

            return (
                False,
                (
                    "provider is already assigned "
                    f"to request {current_request}"
                ),
            )

        # ------------------------------------------------------------
        # An item can be passed only once
        # ------------------------------------------------------------

        other = graph.item_in_use(provider_id, ignore_request=request_id)

        if other is not None:

            return (
                False,
                f"item is already passed to request {other}",
            )

        # ------------------------------------------------------------
        # Remove old provider
        # ------------------------------------------------------------

        old_provider = graph.handoff.get(
            request_id
        )

        if old_provider is not None:

            if old_provider in nodes:

                nodes[old_provider].active = False

        # ------------------------------------------------------------
        # Activate new provider
        # ------------------------------------------------------------

        provider.active = True

        graph.handoff[request_id] = provider_id

        graph.handoff_score[request_id] = (
            candidate_scores[provider_id]
        )

    # ========================================================================
    # UNMATCH
    # ========================================================================

    elif op.op == "unmatch":

        request_id = op.need

        if request_id not in graph.handoff:

            return (
                False,
                "request has no confirmed handoff",
            )

        provider_id = graph.handoff[
            request_id
        ]

        del graph.handoff[
            request_id
        ]

        graph.handoff_score.pop(
            request_id,
            None,
        )

        if provider_id in nodes:

            nodes[provider_id].active = False

    # ========================================================================
    # ADD ORDER
    # ========================================================================

    else:

        before = op.before
        after = op.after

        if (
            before not in nodes
            or after not in nodes
        ):

            return (
                False,
                "unknown step",
            )

        if (
            not nodes[before].active
            or not nodes[after].active
        ):

            return (
                False,
                "step is inactive",
            )

        # ------------------------------------------------------------
        # Cross-robot only
        # ------------------------------------------------------------

        if (
            nodes[before].agent
            == nodes[after].agent
        ):

            return (
                False,
                "same robot: local sequence already defines the order",
            )

        # ------------------------------------------------------------
        # Already implied?
        # ------------------------------------------------------------

        if reachable(
            graph,
            before,
            after,
        ):

            return (
                False,
                "ordering is already implied",
            )

        # ------------------------------------------------------------
        # Add
        # ------------------------------------------------------------

        graph.extra_collaboration.append(
            (
                before,
                after,
            )
        )

    # ========================================================================
    # Re-place proposals (a newly activated provider may fit earlier),
    # then cycle guard
    # ========================================================================

    place_proposed_steps(graph)

    if find_cycle(graph) is not None:

        graph.restore(snapshot)

        return (
            False,
            "operation would create a cycle; reverted",
        )

    return True, "ok"


# ============================================================================
# LLM reasoning
# ============================================================================

def _salvage_ops(raw: dict | None) -> RawGraphOps:
    """Fallback after the last retry: keep only well-formed ops.
    With nothing usable, the graph stays as the rule layer left it."""

    ops: list[GraphOp] = []

    items = raw.get("ops", []) if isinstance(raw, dict) else []

    for item in items if isinstance(items, list) else []:
        try:
            ops.append(GraphOp.model_validate(item))
        except (ValueError, TypeError):
            continue

    return RawGraphOps(ops=ops)


async def llm_reason(
    graph: PlanGraph,
    report: RuleReport,
    llm: BaseLLM,
    log: EventLog,
    *,
    scope: str = "full",
    max_ops: int = 8,
    max_retries: int = 2,
) -> list[dict]:

    raw: RawGraphOps = await call_validated(
        llm,
        log,
        phase="graph",
        who="reasoner",
        system=GRAPH_SYSTEM,
        user=serialize_for_llm(
            graph,
            report,
            scope,
        ),
        parse=RawGraphOps.model_validate,
        images=None,
        max_retries=max_retries,
        fallback=_salvage_ops,
    )

    records = []

    # ------------------------------------------------------------
    # Apply operations
    # ------------------------------------------------------------

    for op in raw.ops[:max_ops]:

        ok, reason = apply_op(
            graph,
            op,
        )

        record = {
            **op.model_dump(
                exclude_none=True
            ),
            "status": (
                "applied"
                if ok
                else "rejected"
            ),
            "why": reason,
        }

        records.append(record)

        log.log(
            "graph",
            "reasoner",
            (
                "op_applied"
                if ok
                else "op_rejected"
            ),
            **{
                key: value
                for key, value in record.items()
                if key != "status"
            },
        )

    # ------------------------------------------------------------
    # Max-op overflow
    # ------------------------------------------------------------

    for op in raw.ops[max_ops:]:

        records.append(
            {
                **op.model_dump(
                    exclude_none=True
                ),
                "status": "rejected",
                "why": (
                    f"more than max_ops={max_ops}"
                ),
            }
        )

    return records


# ============================================================================
# Result
# ============================================================================

@dataclass
class GraphResult:

    graph: PlanGraph

    report: RuleReport

    first_report: RuleReport

    ops: list[dict]

    stats: dict


# ============================================================================
# Main entry point
# ============================================================================

async def graph_reasoning(
    task: str,
    plans: dict[str, LocalPlan],
    auction: AuctionResult,
    llm: BaseLLM,
    log: EventLog,
    *,
    use_llm: bool = True,
    scope: str = "full",
    max_ops: int = 8,
) -> GraphResult:

    # ----------------------------------------------------------------
    # Build graph
    # ----------------------------------------------------------------

    graph = PlanGraph(
        task,
        plans,
        auction,
    )

    initial_handoffs = len(
        graph.handoff
    )

    log.log(
        "graph",
        "rules",
        "initial_selection",
        handoffs=[
            [request_id, provider_id]
            for request_id, provider_id in sorted(graph.handoff.items())
        ],
    )

    # ----------------------------------------------------------------
    # Layer 1
    # ----------------------------------------------------------------

    first_report = rule_verify(
        graph
    )

    log.log(
        "graph",
        "rules",
        "verified",
        fixes=len(first_report.fixes),
        warnings=len(first_report.warnings),
        unresolved=len(
            first_report.unresolved_needs
        ),
    )

    # ----------------------------------------------------------------
    # Layer 2
    # ----------------------------------------------------------------

    ops: list[dict] = []

    final_report = first_report

    if use_llm:

        ops = await llm_reason(
            graph,
            first_report,
            llm,
            log,
            scope=scope,
            max_ops=max_ops,
        )

        # ------------------------------------------------------------
        # Guard pass
        # ------------------------------------------------------------

        final_report = rule_verify(
            graph
        )

        log.log(
            "graph",
            "rules",
            "verified_after_llm",
            fixes=len(
                final_report.fixes
            ),
            warnings=len(
                final_report.warnings
            ),
            unresolved=len(
                final_report.unresolved_needs
            ),
        )

    # ----------------------------------------------------------------
    # Statistics
    # ----------------------------------------------------------------

    applied = [
        operation
        for operation in ops
        if operation["status"] == "applied"
    ]

    changed_matches = sum(
        1
        for operation in applied
        if operation["op"]
        in {
            "reassign",
            "unmatch",
        }
    )

    stats = {
        "ops_proposed": len(ops),

        "ops_applied": len(applied),

        "ops_rejected": (
            len(ops) - len(applied)
        ),

        "reassigned": sum(
            1
            for operation in applied
            if operation["op"]
            == "reassign"
        ),

        "unmatched": sum(
            1
            for operation in applied
            if operation["op"]
            == "unmatch"
        ),

        "orders_added": sum(
            1
            for operation in applied
            if operation["op"]
            == "add_order"
        ),

        "rule_fixes": (
            len(first_report.fixes)
            + (
                len(final_report.fixes)
                if use_llm
                else 0
            )
        ),

        "proposals_moved_earlier": sum(
            1
            for p in final_report.placements
            if p["before"] is not None
        ),

        "initial_handoffs": initial_handoffs,

        "final_handoffs": len(
            graph.handoff
        ),

        "modification_ratio": round(
            changed_matches
            / max(
                1,
                initial_handoffs,
            ),
            4,
        ),

        "unresolved_needs": len(
            final_report.unresolved_needs
        ),
    }

    return GraphResult(
        graph=graph,
        report=final_report,
        first_report=first_report,
        ops=ops,
        stats=stats,
    )
