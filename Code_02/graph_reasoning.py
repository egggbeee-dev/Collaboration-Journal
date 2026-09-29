"""Stage 3 - GRAPH REASONING (central: rules + one LLM call + rules).

Nodes are the robots' steps. This stage makes and fixes the EDGES and repairs the joint plan.
With Stage 3 (Edge Proposal), edges proposed by BOTH robots of a pair are connected first (mutual)
and one-sided proposals are given to the LLM as evidence. Without Stage 3, the LLM connects alone.

Edges
    sequence       a robot's own order                     (from the plans)
    collaboration  provider -> requester                   HELP -> ASK_HELP, PASS -> RECEIVE
    order          cross-robot ordering added by the LLM   (e.g. same-room conflicts)

LLM operations (every op is checked by rules; a violating op is rolled back and logged)
    connect(request, provider)            link an existing request to an existing provider step
    disconnect(request)                   remove a wrong link
    complete_transfer(request | provider, robot, item)
                                          fill the missing half of an ITEM transfer, only when the
                                          robot itself declared it: PASS needs the item in the
                                          robot's has_items; RECEIVE needs an own step that uses it
    add_order(before, after, reason)      cross-robot ordering
    drop(step, reason)                    remove an unrelated / duplicate LOCAL step
    flag_gap(description)                 report a needed part nobody does (graph unchanged)

The graph never invents new work: it links what robots wrote and completes item transfers the
robots themselves declared.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Literal, Optional

from pydantic import BaseModel, Field

from llm import BaseLLM
from runtime import EventLog, banner, call_validated, log_block
from schemas import PAIR, PROVIDER_TYPES, REQUEST_TYPES, LocalPlan, Offer, robot_label


# =============================================================================
# Text helpers (only used to CHECK declarations, never to decide matches)
# =============================================================================

_STOP = {"the", "a", "an", "to", "of", "in", "on", "and", "from", "for", "with", "my", "your", "it"}


def _words(text: str | None) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower())) - _STOP


def same_item(a: str | None, b: str | None) -> bool:
    """'bath mat' ~ 'mat', 'water bottle' ~ 'bottle of water' (word-set containment)."""
    wa, wb = _words(a), _words(b)
    return bool(wa and wb) and (wa <= wb or wb <= wa)


def mentions(action: str, item: str | None) -> bool:
    """Does a step's action text mention the item? (all item words, or its head noun)"""
    wi, wa = _words(item), _words(action)
    if not wi:
        return False
    head = re.findall(r"[a-z0-9]+", (item or "").lower())[-1]
    return wi <= wa or head in wa


# =============================================================================
# Graph
# =============================================================================

@dataclass
class Node:
    id: str
    agent: str
    type: str
    action: str
    item: Optional[str]
    target: Optional[str]
    order: int
    origin: str = "plan"            # "plan" | "graph"
    position: float = 0.0           # execution position in the robot's own sequence
    anchor: Optional[str] = None    # graph-added RECEIVE must precede this own step
    dropped: bool = False           # removed by `drop`
    idle: bool = False              # unmatched HELP / PASS at the end: not scheduled

    @property
    def active(self) -> bool:
        return not self.dropped and not self.idle


@dataclass
class Edge:
    request: str
    provider: str
    source: str        # "mutual" (Stage 3, both robots) | "llm" (connect) | "completion" (complete_transfer)
    seq: int
    reason: str = ""


class PlanGraph:
    def __init__(self, task: str, plans: dict[str, LocalPlan], offers: dict[str, Offer],
                 proposals: list[dict] | None = None) -> None:
        self.task = task
        self.proposals = proposals or []          # Stage 3 output (may be empty)
        self.offers = offers                      # PUBLIC offers only
        self.agents = sorted(plans, key=lambda a: int(a.split("_")[-1]))
        self.nodes: dict[str, Node] = {}
        for aid in self.agents:
            for s in plans[aid].steps:
                self.nodes[s.id] = Node(s.id, aid, s.type, s.action, s.item, s.target,
                                        s.order, s.origin, float(s.order))
        self.edges: dict[str, Edge] = {}          # request id -> edge
        self.orders: list[dict] = []              # {"before", "after", "reason", "seq"}
        self.gaps: list[str] = []
        self.drops: list[dict] = []
        self.added: list[str] = []                # ids of graph-added steps
        self._seq = 0

    # ---------------------------------------------------------------- basics
    def next_seq(self) -> int:
        self._seq += 1
        return self._seq

    @staticmethod
    def sort_key(nid: str) -> tuple[int, int]:
        a, b = nid.split("-")
        return int(a), int(b)

    def provider_of(self, request_id: str) -> Optional[str]:
        e = self.edges.get(request_id)
        return e.provider if e else None

    def request_of(self, provider_id: str) -> Optional[str]:
        return next((r for r, e in self.edges.items() if e.provider == provider_id), None)

    def sequence(self, agent: str) -> list[Node]:
        return sorted((n for n in self.nodes.values() if n.agent == agent and n.active),
                      key=lambda n: (n.position, n.order))

    def edge_list(self) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        for aid in self.agents:
            seq = self.sequence(aid)
            out += [(a.id, b.id) for a, b in zip(seq, seq[1:])]
        for e in self.edges.values():
            if self.nodes[e.provider].active and self.nodes[e.request].active:
                out.append((e.provider, e.request))
        for o in self.orders:
            if self.nodes[o["before"]].active and self.nodes[o["after"]].active:
                out.append((o["before"], o["after"]))
        return out

    def reachable(self, src: str, dst: str) -> bool:
        adj: dict[str, list[str]] = {}
        for a, b in self.edge_list():
            adj.setdefault(a, []).append(b)
        stack, seen = [src], set()
        while stack:
            x = stack.pop()
            if x == dst:
                return True
            if x in seen:
                continue
            seen.add(x)
            stack += adj.get(x, [])
        return False

    # ---------------------------------------------------------------- items
    def item_passed_to(self, owner: str, item: str, ignore_request: str | None = None) -> Optional[str]:
        """If `owner` already passes `item` through an edge, return that request id."""
        for r, e in self.edges.items():
            p = self.nodes[e.provider]
            if r != ignore_request and p.type == "PASS" and p.agent == owner and same_item(p.item, item):
                return r
        return None

    def declared_item(self, owner: str, item: str) -> Optional[str]:
        """The owner's own has_items entry matching `item`, if any."""
        offer = self.offers.get(owner)
        if offer is None:
            return None
        return next((h.object for h in offer.has_items if same_item(h.object, item)), None)

    # ---------------------------------------------------------------- mutation
    def add_node(self, agent: str, type: str, action: str, item: str | None,
                 target: str | None, anchor: str | None = None) -> Node:
        num = agent.split("_")[-1]
        order = max((n.order for n in self.nodes.values() if n.agent == agent), default=0) + 1
        node = Node(f"{num}-{order}", agent, type, action, item, target, order,
                    origin="graph", position=float(order), anchor=anchor)
        self.nodes[node.id] = node
        self.added.append(node.id)
        return node

    def remove_node(self, nid: str) -> None:
        self.nodes.pop(nid, None)
        if nid in self.added:
            self.added.remove(nid)
        self.orders = [o for o in self.orders if nid not in (o["before"], o["after"])]

    def snapshot(self):
        return copy.deepcopy((self.nodes, self.edges, self.orders, self.gaps, self.drops, self.added, self._seq))

    def restore(self, snap) -> None:
        (self.nodes, self.edges, self.orders, self.gaps, self.drops, self.added, self._seq) = copy.deepcopy(snap)

    def to_dict(self) -> dict:
        return {
            "nodes": {k: {**asdict(v), "active": v.active} for k, v in self.nodes.items()},
            "edges": [asdict(e) for e in self.edges.values()],
            "orders": self.orders,
            "gaps": self.gaps,
            "drops": self.drops,
            "added_steps": self.added,
        }


# =============================================================================
# Graph algorithms
# =============================================================================

def find_cycle(graph: PlanGraph) -> Optional[list[str]]:
    adj: dict[str, list[str]] = {n: [] for n, x in graph.nodes.items() if x.active}
    for a, b in graph.edge_list():
        adj[a].append(b)
    color = {n: 0 for n in adj}
    parent: dict[str, str] = {}

    for start in sorted(adj, key=graph.sort_key):
        if color[start]:
            continue
        stack = [(start, iter(adj[start]))]
        color[start] = 1
        while stack:
            v, it = stack[-1]
            nxt = next(it, None)
            if nxt is None:
                color[v] = 2
                stack.pop()
            elif color[nxt] == 0:
                color[nxt] = 1
                parent[nxt] = v
                stack.append((nxt, iter(adj[nxt])))
            elif color[nxt] == 1:
                cyc, x = [nxt], v
                while x != nxt:
                    cyc.append(x)
                    x = parent[x]
                return cyc[::-1]
    return None


def topological_levels(graph: PlanGraph) -> Optional[tuple[list[str], dict[str, int]]]:
    active = [n for n, x in graph.nodes.items() if x.active]
    indeg = {n: 0 for n in active}
    adj: dict[str, list[str]] = {n: [] for n in active}
    for a, b in graph.edge_list():
        adj[a].append(b)
        indeg[b] += 1
    level = {n: 0 for n in active}
    ready = sorted((n for n in active if indeg[n] == 0), key=graph.sort_key)
    order: list[str] = []
    while ready:
        v = ready.pop(0)
        order.append(v)
        for w in adj[v]:
            level[w] = max(level[w], level[v] + 1)
            indeg[w] -= 1
            if indeg[w] == 0:
                ready.append(w)
                ready.sort(key=graph.sort_key)
    if len(order) != len(active):
        return None
    return order, level


def _schedule_length(graph: PlanGraph) -> Optional[int]:
    res = topological_levels(graph)
    if res is None:
        return None
    return (max(res[1].values()) + 1) if res[1] else 0


def place_graph_steps(graph: PlanGraph) -> list[dict]:
    """Choose WHEN graph-added steps run inside their robot's own sequence.

    Plan steps keep the robot's own order. Each graph-added step tries every slot
    (a RECEIVE only slots before its anchor) and keeps the acyclic one with the
    shortest schedule; ties -> the latest slot (least interruption of own work).
    """
    for n in graph.nodes.values():
        n.position = float(n.order) + (1e6 if n.origin == "graph" else 0.0)

    placements = []
    for node in sorted((n for n in graph.nodes.values() if n.origin == "graph" and n.active),
                       key=lambda n: graph.sort_key(n.id)):
        others = [m for m in graph.sequence(node.agent) if m.id != node.id]
        last = len(others)
        if node.anchor:
            idx = next((i for i, m in enumerate(others) if m.id == node.anchor), None)
            if idx is not None:
                last = idx                       # only slots before the anchor
        slots: list[float] = []
        prev = None
        for m in others[: last]:
            lo = prev.position if prev is not None else m.position - 1.0
            slots.append((lo + m.position) / 2.0)
            prev = m
        if node.anchor and last < len(others):
            a = others[last]
            lo = prev.position if prev is not None else a.position - 1.0
            slots.append((lo + a.position) / 2.0)     # right before the anchor
        else:
            slots.append((others[-1].position if others else 0.0) + 1.0)

        best = None
        for i, pos in enumerate(slots):
            node.position = pos
            if find_cycle(graph) is not None:
                continue
            length = _schedule_length(graph)
            if length is None:
                continue
            key = (length, -i, pos)
            if best is None or key < best:
                best = key
        if best is not None:
            node.position = best[2]
            placements.append({"step": node.id, "schedule_length": best[0]})
    return placements


# =============================================================================
# Rules
# =============================================================================

@dataclass
class RuleReport:
    fixes: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)       # requests with no provider
    idle_providers: list[str] = field(default_factory=list)   # HELP / PASS nobody uses
    placements: list[dict] = field(default_factory=list)
    order: list[str] = field(default_factory=list)
    levels: dict[str, int] = field(default_factory=dict)
    ok: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def rule_verify(graph: PlanGraph, *, final: bool = False) -> RuleReport:
    rep = RuleReport()

    # 1. edges touching dropped steps are removed
    for r, e in list(graph.edges.items()):
        if not (graph.nodes[r].active and graph.nodes[e.provider].active):
            del graph.edges[r]
            rep.fixes.append({"rule": "edge_removed_inactive_step", "request": r, "provider": e.provider})

    # 2. graph-added steps without an edge are removed
    for nid in list(graph.added):
        n = graph.nodes[nid]
        linked = graph.request_of(nid) if n.type in PROVIDER_TYPES else graph.provider_of(nid)
        if linked is None:
            graph.remove_node(nid)
            rep.fixes.append({"rule": "unlinked_graph_step_removed", "step": nid})

    # 3. where graph-added steps run
    rep.placements = place_graph_steps(graph)

    # 4. break cycles: remove the most recently made link in the cycle
    while (cyc := find_cycle(graph)) is not None:
        pairs = set(zip(cyc, cyc[1:] + cyc[:1]))
        collab = [e for e in graph.edges.values() if (e.provider, e.request) in pairs]
        if collab:
            e = max(collab, key=lambda x: x.seq)
            del graph.edges[e.request]
            if graph.nodes[e.provider].origin == "graph":
                graph.remove_node(e.provider)
            rep.fixes.append({"rule": "cycle_broken", "request": e.request, "provider": e.provider, "cycle": cyc})
        else:
            ords = [o for o in graph.orders if (o["before"], o["after"]) in pairs]
            if not ords:
                rep.ok = False
                rep.warnings.append(f"unbreakable cycle: {cyc}")
                break
            o = max(ords, key=lambda x: x["seq"])
            graph.orders.remove(o)
            rep.fixes.append({"rule": "cycle_broken_order", "before": o["before"], "after": o["after"]})
        place_graph_steps(graph)

    # 5. unresolved requests / idle providers
    for n in sorted(graph.nodes.values(), key=lambda n: graph.sort_key(n.id)):
        if not n.active:
            continue
        if n.type in REQUEST_TYPES and n.id not in graph.edges:
            rep.unresolved.append(n.id)
        if n.type in PROVIDER_TYPES and graph.request_of(n.id) is None:
            rep.idle_providers.append(n.id)
    if final:
        for nid in rep.idle_providers:
            graph.nodes[nid].idle = True           # volunteered but not needed: not scheduled

    # 6. warnings
    for r, e in graph.edges.items():
        req, prov = graph.nodes[r], graph.nodes[e.provider]
        if req.type == "RECEIVE":
            for s in graph.sequence(req.agent):
                if s.id == r:
                    break
                if s.type == "LOCAL" and mentions(s.action, req.item):
                    rep.warnings.append(f"{s.id} uses '{req.item}' before receiving it ({r})")
        if prov.type == "PASS" and prov.origin == "plan" and graph.declared_item(prov.agent, prov.item) is None:
            rep.warnings.append(f"{prov.id} passes '{prov.item}', which {prov.agent} did not declare in has_items")

    # 7. schedule
    res = topological_levels(graph)
    if res is None:
        rep.ok = False
    else:
        rep.order, rep.levels = res
    return rep


# =============================================================================
# Stage 3 proposals -> mutual edges (connected before the LLM)
# =============================================================================

def connect_mutual(graph: PlanGraph) -> list[dict]:
    """Connect edges that BOTH robots of the pair proposed. Rule-checked like `connect`."""
    records = []
    for p in graph.proposals:
        if not p.get("mutual"):
            continue
        ok, why = apply_op(graph, GraphOp(op="connect", request=p["request"], provider=p["provider"]))
        if ok:
            graph.edges[p["request"]].source = "mutual"
        records.append({"request": p["request"], "provider": p["provider"], "connected": ok, "why": why})
    return records


# =============================================================================
# LLM operations
# =============================================================================

class GraphOp(BaseModel):
    op: Literal["connect", "disconnect", "complete_transfer", "add_order", "drop", "flag_gap"]
    request: Optional[str] = None
    provider: Optional[str] = None
    robot: Optional[str] = None
    item: Optional[str] = None
    before: Optional[str] = None
    after: Optional[str] = None
    step: Optional[str] = None
    reason: Optional[str] = None
    description: Optional[str] = None


class RawGraphOps(BaseModel):
    reasoning: str = ""
    ops: list[GraphOp] = Field(default_factory=list)


def _node(graph: PlanGraph, nid: str | None) -> Optional[Node]:
    n = graph.nodes.get(nid or "")
    return n if n is not None and n.active else None


def _apply(graph: PlanGraph, op: GraphOp) -> tuple[bool, str]:
    if op.op == "connect":
        r, p = _node(graph, op.request), _node(graph, op.provider)
        if r is None or p is None:
            return False, "unknown or inactive step"
        if r.type not in REQUEST_TYPES:
            return False, f"{r.id} is {r.type}, not a request (ASK_HELP / RECEIVE)"
        if p.type != PAIR[r.type]:
            return False, f"{r.type} must be served by {PAIR[r.type]}, not {p.type}"
        if r.agent == p.agent:
            return False, "cannot connect a robot to itself"
        if r.id in graph.edges:
            return False, f"{r.id} is already served by {graph.edges[r.id].provider}; disconnect first"
        if graph.request_of(p.id) is not None:
            return False, f"{p.id} already serves {graph.request_of(p.id)}"
        if p.type == "PASS" and (other := graph.item_passed_to(p.agent, p.item)) is not None:
            return False, f"{p.agent} already passes '{p.item}' to {other}"
        graph.edges[r.id] = Edge(r.id, p.id, "llm", graph.next_seq(), op.reason or "")
        return True, "ok"

    if op.op == "disconnect":
        e = graph.edges.get(op.request or "")
        if e is None:
            return False, f"{op.request} has no edge"
        if e.source == "mutual" and not op.reason:
            return False, "disconnecting a mutual edge needs a reason"
        del graph.edges[e.request]
        if graph.nodes[e.provider].origin == "graph":
            graph.remove_node(e.provider)
        return True, "ok"

    if op.op == "complete_transfer":
        if op.robot not in graph.agents:
            return False, f"unknown robot {op.robot!r}"

        # (a) an unmatched RECEIVE -> add a PASS to a robot that DECLARED the item
        if op.request:
            r = _node(graph, op.request)
            if r is None or r.type != "RECEIVE":
                return False, "complete_transfer(request=...) needs an active RECEIVE"
            if r.id in graph.edges:
                return False, f"{r.id} is already served"
            if op.robot == r.agent:
                return False, "the owner must be another robot"
            declared = graph.declared_item(op.robot, op.item or r.item)
            if declared is None:
                return False, f"{op.robot} did not declare '{op.item or r.item}' in has_items"
            if (other := graph.item_passed_to(op.robot, declared)) is not None:
                return False, f"{op.robot} already passes '{declared}' to {other}"
            unused = next((n for n in graph.nodes.values() if n.agent == op.robot and n.type == "PASS"
                           and n.active and same_item(n.item, declared) and graph.request_of(n.id) is None), None)
            if unused is not None:
                return False, f"{op.robot} already has PASS {unused.id} for it; use connect"
            p = graph.add_node(op.robot, "PASS", f"Pass the {declared} to {robot_label(r.agent)}",
                               declared, r.agent)
            graph.edges[r.id] = Edge(r.id, p.id, "completion", graph.next_seq(), op.reason or "")
            return True, "ok"

        # (b) an unmatched PASS -> add a RECEIVE to a robot whose own step USES the item
        if op.provider:
            p = _node(graph, op.provider)
            if p is None or p.type != "PASS":
                return False, "complete_transfer(provider=...) needs an active PASS"
            if graph.request_of(p.id) is not None:
                return False, f"{p.id} already serves {graph.request_of(p.id)}"
            if op.robot == p.agent:
                return False, "the receiver must be another robot"
            anchor = next((s for s in graph.sequence(op.robot)
                           if s.type == "LOCAL" and mentions(s.action, p.item)), None)
            if anchor is None:
                return False, f"{op.robot} has no own step that uses '{p.item}'"
            r = graph.add_node(op.robot, "RECEIVE", f"Receive the {p.item} from {robot_label(p.agent)}",
                               p.item, p.agent, anchor=anchor.id)
            graph.edges[r.id] = Edge(r.id, p.id, "completion", graph.next_seq(), op.reason or "")
            return True, "ok"

        return False, "complete_transfer needs `request` (a RECEIVE) or `provider` (a PASS)"

    if op.op == "add_order":
        a, b = _node(graph, op.before), _node(graph, op.after)
        if a is None or b is None:
            return False, "unknown or inactive step"
        if a.agent == b.agent:
            return False, "add_order is for steps of different robots"
        if graph.reachable(a.id, b.id):
            return False, "already ordered"
        graph.orders.append({"before": a.id, "after": b.id, "reason": op.reason or "", "seq": graph.next_seq()})
        return True, "ok"

    if op.op == "drop":
        s = _node(graph, op.step)
        if s is None:
            return False, "unknown or inactive step"
        if s.type != "LOCAL" or s.origin != "plan":
            return False, "only a robot's own LOCAL step can be dropped"
        if not op.reason:
            return False, "drop needs a reason"
        if any(n.anchor == s.id for n in graph.nodes.values()):
            return False, f"{s.id} uses a received item; disconnect that transfer first"
        s.dropped = True
        graph.drops.append({"step": s.id, "agent": s.agent, "action": s.action, "reason": op.reason})
        return True, "ok"

    if op.op == "flag_gap":
        if not op.description:
            return False, "flag_gap needs a description"
        graph.gaps.append(op.description)
        return True, "ok"

    return False, f"unknown op {op.op}"


def apply_op(graph: PlanGraph, op: GraphOp) -> tuple[bool, str]:
    """Apply one op; roll back if it breaks a rule or creates a cycle."""
    snap = graph.snapshot()
    ok, why = _apply(graph, op)
    if not ok:
        graph.restore(snap)
        return False, why
    place_graph_steps(graph)
    if find_cycle(graph) is not None:
        graph.restore(snap)
        return False, "would create a cycle"
    return True, "ok"


# =============================================================================
# LLM layer
# =============================================================================

GRAPH_SYSTEM = """You are the graph reasoner of a multi-robot team. Each robot is in a different room
and wrote its OWN plan. Their steps are the nodes. Your job is to build and fix the EDGES and
make the plans work as ONE joint plan for the TASK.

STEP TYPES
- LOCAL: the robot does it itself.
- ASK_HELP: the robot needs another robot to do a task (its body cannot).  served by HELP
- HELP: the robot volunteers to do a task for another robot.
- RECEIVE: the robot needs an item from another robot.                  served by PASS
- PASS: the robot gives one of its items.
An edge links a request (ASK_HELP / RECEIVE) to the provider step that serves it.

PROPOSED EDGES (if given)
Robots proposed edges from their own point of view, using private knowledge you do not have.
- "mutual": both robots proposed it. It is ALREADY connected. Disconnect only with a clear reason.
- one-sided: one robot proposed it. Strong evidence, but check it before you connect.
Robots may also have missed links; you can still connect pairs nobody proposed.

OPERATIONS (JSON objects in "ops")
1. {"op": "connect", "request": "<id>", "provider": "<id>", "reason": "..."}
   ASK_HELP <- HELP, RECEIVE <- PASS, different robots, one provider per request.
   Match by MEANING (e.g. "exercise mat" can be served by a "bath mat").
2. {"op": "disconnect", "request": "<id>", "reason": "..."}
3. {"op": "complete_transfer", "request": "<RECEIVE id>", "robot": "<owner>", "item": "<its has_items name>", "reason": "..."}
   A RECEIVE has no PASS, but another robot DECLARED the item in has_items -> a PASS is added to it.
   {"op": "complete_transfer", "provider": "<PASS id>", "robot": "<receiver>", "reason": "..."}
   A PASS has no RECEIVE, but a robot has an own step that USES the item -> a RECEIVE is added to it.
   Items only. Never for tasks.
4. {"op": "add_order", "before": "<id>", "after": "<id>", "reason": "..."}
   Steps of different robots that conflict (e.g. same room, place an item after the furniture is moved).
5. {"op": "drop", "step": "<LOCAL id>", "reason": "unrelated to the task | duplicate of <id>"}
6. {"op": "flag_gap", "description": "..."}   a part the TASK needs that nobody does.

PRIORITIES
1) connect every request you can, by meaning (use the proposed edges);  2) complete_transfer for remaining item requests;
3) add_order for real conflicts;  4) drop unrelated or duplicate LOCAL steps;  5) flag_gap.

RULES
- Use only step ids that exist. You cannot invent tasks or write steps.
- Every op is checked; invalid ops are rejected.
- Leave a request unresolved rather than making a wrong link.

Return ONE JSON object: {"reasoning": "...", "ops": [...]}
"""


def serialize_for_llm(graph: PlanGraph, report: RuleReport, scope: str = "full") -> str:
    robots = {}
    for aid in graph.agents:
        o = graph.offers.get(aid)
        steps = [
            {k: v for k, v in {"id": n.id, "type": n.type, "action": n.action,
                               "item": n.item, "target": n.target}.items() if v}
            for n in graph.sequence(aid)
            if scope == "full" or n.type != "LOCAL"
        ]
        robots[aid] = {
            "capability": o.capability if o else "",
            "can_do": [f"{c.action} {c.object or ''}".strip() for c in o.can_do] if o else [],
            "has_items": [h.object for h in o.has_items] if o else [],
            "need_from_others": [f"{n.id} ({n.kind}) {n.what}" for n in o.need_from_others] if o else [],
            "steps": steps,
        }
    return json.dumps({
        "task": graph.task,
        "robots": robots,
        "edges": [{"request": e.request, "provider": e.provider, "source": e.source}
                  for e in graph.edges.values()],
        "proposed_edges": [
            {"request": p["request"], "provider": p["provider"], "proposed_by": p["proposed_by"],
             "mutual": p["mutual"], "why": p["why"],
             "connected": graph.provider_of(p["request"]) == p["provider"]}
            for p in graph.proposals
        ],
        "unmatched_requests": report.unresolved,
        "unmatched_providers": report.idle_providers,
        "warnings": report.warnings,
    }, ensure_ascii=False, indent=2)


async def llm_reason(graph: PlanGraph, report: RuleReport, llm: BaseLLM, log: EventLog, *,
                     scope: str = "full", max_ops: int = 20, max_retries: int = 0,
                     verbose: bool = False) -> list[dict]:
    user = serialize_for_llm(graph, report, scope)
    if verbose:
        banner("GRAPH REASONING — input to the LLM")
        log_block("GRAPH INPUT", user)

    raw: RawGraphOps = await call_validated(
        llm, log, phase="graph", who="reasoner", system=GRAPH_SYSTEM, user=user,
        parse=RawGraphOps.model_validate, images=None, max_retries=max_retries,
        verbose=verbose, banner_label="GRAPH REASONING RAW",
    )

    records = []
    for i, op in enumerate(raw.ops):
        if i >= max_ops:
            ok, why = False, f"over the limit of {max_ops} ops"
        else:
            ok, why = apply_op(graph, op)
        rec = {**op.model_dump(exclude_none=True), "status": "applied" if ok else "rejected", "why": why}
        records.append(rec)
        log.log("graph", "reasoner", "op_applied" if ok else "op_rejected", **rec)
        if verbose:
            print(f"  [GRAPH OP] {rec['status'].upper():8s} {op.model_dump(exclude_none=True)} — {why}")
    if verbose and not raw.ops:
        print("  [GRAPH OP] LLM proposed no operations")
    return records


# =============================================================================
# Driver
# =============================================================================

@dataclass
class GraphResult:
    graph: PlanGraph
    first_report: RuleReport
    report: RuleReport
    ops: list[dict]
    stats: dict


async def graph_reasoning(task: str, plans: dict[str, LocalPlan], offers: dict[str, Offer],
                          llm: BaseLLM, log: EventLog, *, proposals: list[dict] | None = None,
                          use_llm: bool = True, scope: str = "full",
                          max_ops: int = 20, max_retries: int = 0, verbose: bool = False) -> GraphResult:
    graph = PlanGraph(task, plans, offers, proposals)

    mutual = connect_mutual(graph)
    log.log("graph", "rules", "mutual_connected", records=mutual)

    first = rule_verify(graph)
    log.log("graph", "rules", "verified", unresolved=len(first.unresolved), idle=len(first.idle_providers))
    if verbose:
        banner("GRAPH REASONING — Layer 1 (rules)")
        if graph.proposals:
            print(f"  proposed edges     : {[(p['provider'], p['request'], 'mutual' if p['mutual'] else 'one-sided') for p in graph.proposals]}")
            print(f"  mutual connected   : {[(m['provider'], m['request']) for m in mutual if m['connected']] or '(none)'}")
        print(f"  unmatched requests : {first.unresolved or '(none)'}")
        print(f"  unmatched providers: {first.idle_providers or '(none)'}")
        print(f"  warnings           : {first.warnings or '(none)'}")

    ops: list[dict] = []
    if use_llm:
        ops = await llm_reason(graph, first, llm, log, scope=scope, max_ops=max_ops,
                               max_retries=max_retries, verbose=verbose)

    final = rule_verify(graph, final=True)
    log.log("graph", "rules", "final", ok=final.ok, unresolved=len(final.unresolved))

    applied = [o for o in ops if o["status"] == "applied"]
    props = graph.proposals
    stats = {
        "proposals": len(props),
        "proposals_mutual": sum(1 for p in props if p["mutual"]),
        "proposals_one_sided": sum(1 for p in props if not p["mutual"]),
        "one_sided_accepted": sum(1 for p in props if not p["mutual"]
                                  and graph.provider_of(p["request"]) == p["provider"]),
        "mutual_kept": sum(1 for p in props if p["mutual"]
                           and graph.provider_of(p["request"]) == p["provider"]),
        "ops_proposed": len(ops),
        "ops_applied": len(applied),
        "ops_rejected": len(ops) - len(applied),
        **{f"applied_{k}": sum(1 for o in applied if o["op"] == k)
           for k in ("connect", "disconnect", "complete_transfer", "add_order", "drop", "flag_gap")},
        "edges": len(graph.edges),
        "edges_by_mutual": sum(1 for e in graph.edges.values() if e.source == "mutual"),
        "edges_by_connect": sum(1 for e in graph.edges.values() if e.source == "llm"),
        "edges_by_completion": sum(1 for e in graph.edges.values() if e.source == "completion"),
        "graph_added_steps": len(graph.added),
        "cycles_broken": sum(1 for f in final.fixes if f["rule"].startswith("cycle_broken")),
        "unresolved_requests": len(final.unresolved),
        "idle_providers": len(final.idle_providers),
        "schedule_length": (max(final.levels.values()) + 1) if final.levels else 0,
    }
    return GraphResult(graph, first, final, ops, stats)
