"""Stage 4 - GRAPH REASONING (layered verification + minimal-intervention repair).

Graph
  node = one step of one robot        (id, agent, order, type, kind, action, item, active)
  edge = "sequence" (consecutive active steps of ONE robot, owned by that robot)
         "handoff"  (PASS -> NEED, made by the Auction)
         "order"    (extra cross-robot ordering added by the LLM reasoner)
  A PASS step that lost the Auction is inactive (its owner withdrew it) but is remembered
  as a self-nominated candidate, so it can be re-instated by a `reassign`.

Layer 1 - RuleVerifier (deterministic, no LLM)
  orphan PASS removal, cycle breaking, unresolved NEED report, use-before-receive warning,
  topological order + levels.
Layer 2 - LLMReasoner (one central LLM call, text only, never sees images / hidden info)
  Looks for semantic errors and repairs them DIRECTLY, but only through 3 guarded operations:
    reassign : move a NEED to another *self-nominated* candidate (no new collaboration)
    unmatch  : cut a wrong handoff
    add_order: add a cross-robot ordering constraint between two existing steps
  It cannot create, delete or rewrite steps. Every operation is validated and reverted if it
  would create a cycle.
Layer 1 runs again after the LLM (guard).
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
from schemas import LocalPlan, agent_of


# =========================================================================== graph
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


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    kind: str  # sequence | handoff | order


class PlanGraph:
    def __init__(self, task: str, plans: dict[str, LocalPlan], auction: AuctionResult) -> None:
        self.task = task
        self.nodes: dict[str, Node] = {}
        for aid, plan in plans.items():
            for s in plan.steps:
                self.nodes[s.id] = Node(s.id, aid, s.order, s.type, s.kind, s.action, s.item)
        self.handoff: dict[str, str] = {m.need: m.passed for m in auction.matches}          # need -> pass
        self.handoff_score: dict[str, float] = {m.need: m.score for m in auction.matches}
        self.extra_order: list[tuple[str, str]] = []
        self.candidates: dict[str, list[dict]] = auction.candidates
        matched = set(self.handoff.values())
        for n in self.nodes.values():
            if n.type == "PASS" and n.id not in matched:
                n.active = False  # withdrawn by its owner after losing the Auction

    # -- views
    def active_ids(self) -> list[str]:
        return [i for i, n in self.nodes.items() if n.active]

    def sort_key(self, nid: str):
        n = self.nodes[nid]
        return (int(n.agent.split("_")[-1]), n.order)

    def agent_sequence(self, agent: str) -> list[Node]:
        return sorted((n for n in self.nodes.values() if n.agent == agent and n.active), key=lambda n: n.order)

    def edges(self) -> list[Edge]:
        out: list[Edge] = []
        for agent in sorted({n.agent for n in self.nodes.values()}):
            seq = self.agent_sequence(agent)
            out += [Edge(a.id, b.id, "sequence") for a, b in zip(seq, seq[1:])]
        for need, p in sorted(self.handoff.items()):
            if self.nodes[need].active and self.nodes[p].active:
                out.append(Edge(p, need, "handoff"))
        for a, b in self.extra_order:
            if self.nodes[a].active and self.nodes[b].active:
                out.append(Edge(a, b, "order"))
        return out

    def snapshot(self):
        return copy.deepcopy(({i: n.active for i, n in self.nodes.items()}, self.handoff, self.handoff_score, self.extra_order))

    def restore(self, snap) -> None:
        active, self.handoff, self.handoff_score, self.extra_order = copy.deepcopy(snap)
        for i, a in active.items():
            self.nodes[i].active = a

    def to_dict(self) -> dict:
        return {
            "nodes": [n.__dict__ for n in sorted(self.nodes.values(), key=lambda n: self.sort_key(n.id))],
            "handoffs": [{"need": n, "pass": p, "score": self.handoff_score.get(n)} for n, p in sorted(self.handoff.items())],
            "extra_order": [list(t) for t in self.extra_order],
            "edges": [e.__dict__ for e in self.edges()],
        }


# =========================================================================== graph algorithms
def find_cycle(graph: PlanGraph) -> list[Edge] | None:
    adj: dict[str, list[Edge]] = defaultdict(list)
    for e in graph.edges():
        adj[e.src].append(e)
    color = {n: 0 for n in graph.active_ids()}
    parent: dict[str, Edge] = {}

    def dfs(u: str):
        color[u] = 1
        for e in adj[u]:
            v = e.dst
            if color[v] == 0:
                parent[v] = e
                r = dfs(v)
                if r:
                    return r
            elif color[v] == 1:  # back edge -> cycle v ... u -> v
                cyc, x = [e], u
                while x != v:
                    pe = parent[x]
                    cyc.append(pe)
                    x = pe.src
                return cyc
        color[u] = 2
        return None

    for n in sorted(color, key=graph.sort_key):
        if color[n] == 0:
            r = dfs(n)
            if r:
                return r
    return None


def topological_levels(graph: PlanGraph):
    """Kahn's algorithm with deterministic tie-breaks. Returns (order, level) or None if cyclic."""
    ids = graph.active_ids()
    indeg = {i: 0 for i in ids}
    succ: dict[str, list[str]] = defaultdict(list)
    for e in graph.edges():
        succ[e.src].append(e.dst)
        indeg[e.dst] += 1
    level = {i: 0 for i in ids}
    ready = sorted([i for i in ids if indeg[i] == 0], key=graph.sort_key)
    order: list[str] = []
    while ready:
        u = ready.pop(0)
        order.append(u)
        for v in succ[u]:
            level[v] = max(level[v], level[u] + 1)
            indeg[v] -= 1
            if indeg[v] == 0:
                ready.append(v)
                ready.sort(key=graph.sort_key)
    return (order, level) if len(order) == len(ids) else None


def reachable(graph: PlanGraph, a: str, b: str) -> bool:
    adj: dict[str, list[str]] = defaultdict(list)
    for e in graph.edges():
        adj[e.src].append(e.dst)
    seen, stack = {a}, [a]
    while stack:
        u = stack.pop()
        if u == b:
            return True
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return False


# =========================================================================== layer 1: rules
@dataclass
class RuleReport:
    fixes: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    unresolved_needs: list[str] = field(default_factory=list)
    order: list[str] = field(default_factory=list)
    levels: dict[str, int] = field(default_factory=dict)
    ok: bool = True

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def rule_verify(graph: PlanGraph) -> RuleReport:
    rep = RuleReport()

    # 1. orphan PASS (nobody receives it) -> its owner withdraws it
    matched = {p for n, p in graph.handoff.items() if graph.nodes[n].active}
    for n in graph.nodes.values():
        if n.active and n.type == "PASS" and n.id not in matched:
            n.active = False
            rep.fixes.append({"rule": "orphan_pass_withdrawn", "step": n.id})

    # 2. cycles -> remove the least-committed repair edge first.  We prefer an LLM-added
    # order edge; if a handoff itself closes the cycle, remove the lowest-confidence handoff.
    # The verifier never invents a new action/provider.
    while (cyc := find_cycle(graph)) is not None:
        extra = [e for e in cyc if e.kind == "order"]
        if extra:
            e = extra[-1]
            graph.extra_order.remove((e.src, e.dst))
            rep.fixes.append({"rule": "cycle_broken", "removed_order": [e.src, e.dst]})
            continue
        hand = [e for e in cyc if e.kind == "handoff"]
        if hand:
            e = min(hand, key=lambda e: graph.handoff_score.get(e.dst, 0.0))
            del graph.handoff[e.dst]
            graph.handoff_score.pop(e.dst, None)
            graph.nodes[e.src].active = False
            rep.fixes.append({"rule": "cycle_broken", "removed_handoff": [e.src, e.dst]})
            continue
        rep.ok = False
        rep.warnings.append("unbreakable cycle inside a single robot's own sequence")
        return rep

    # 3. NEED without a provider -> report (we never fabricate a provider)
    rep.unresolved_needs = [n.id for n in graph.nodes.values() if n.active and n.type == "NEED" and n.id not in graph.handoff]

    # 4. an item is used before the step that receives it (same robot)
    for r in graph.nodes.values():
        if not (r.active and r.type == "NEED" and r.kind == "item" and r.item):
            continue
        for u in graph.agent_sequence(r.agent):
            if u.type == "LOCAL" and u.order < r.order and r.item.lower() in u.action.lower():
                rep.warnings.append(f"{u.id} seems to use '{r.item}' before it is received at {r.id}")

    # 5. schedule
    res = topological_levels(graph)
    if res is None:
        rep.ok = False
    else:
        rep.order, rep.levels = res
    return rep


# =========================================================================== layer 2: LLM reasoner
class GraphOp(BaseModel):
    op: Literal["reassign", "unmatch", "add_order"]
    need: Optional[str] = None
    to_pass: Optional[str] = None
    before: Optional[str] = None
    after: Optional[str] = None
    reason: str = ""

    @model_validator(mode="after")
    def _fields(self):
        need_fields = {"reassign": ("need", "to_pass"), "unmatch": ("need",), "add_order": ("before", "after")}[self.op]
        for f in need_fields:
            if not getattr(self, f):
                raise ValueError(f"op '{self.op}' needs field '{f}'")
        return self


class RawGraphOps(BaseModel):
    ops: list[GraphOp] = Field(default_factory=list)


GRAPH_SYSTEM = """You are the graph reasoner of a robot team. You are NOT a planner.
Each robot wrote its own plan (a list of steps in order). A matching process then linked NEED steps to PASS steps ("handoffs"): the PASS step provides what the NEED step asks for. Several robots may have volunteered for one need; the "candidates" list shows the volunteers ("pass" ids) with their similarity score.

Your job is to check the plan thoroughly and repair problems directly, using ONLY these operations:
  {"op": "reassign", "need": <NEED step id>, "to_pass": <PASS step id>, "reason": string}
      Give a NEED to a different volunteer. `to_pass` must be listed in that need's candidates.
      If that PASS currently serves another NEED, the system will treat the reassign as a
      swap when the two assignments exchange providers; do not avoid a necessary swap merely
      because the PASS is currently occupied.
  {"op": "unmatch", "need": <NEED step id>, "reason": string}
      Cut a handoff that is wrong (the PASS step does not really provide what the NEED asks for).
  {"op": "add_order", "before": <step id>, "after": <step id>, "reason": string}
      Add a cross-robot ordering constraint: `before` must finish before `after` starts. Use it only for steps of DIFFERENT robots when the plan would not work without it (e.g. space must be cleared before furniture is moved, something must be set up before it is used) and it is not already implied by the existing sequence/handoff edges.

You may NOT create, delete or rewrite steps, and you may NOT create a collaboration that no robot volunteered for. Robots keep ownership of what they do and who does it; you only fix connections and ordering.

Check every handoff (does the PASS give exactly what the NEED asks for? is another volunteer clearly better?) and every cross-robot dependency. Change only what is necessary; if the plan is already consistent return {"ops": []}.
Return ONE JSON object: {"ops": [ ... ]}"""


def serialize_for_llm(graph: PlanGraph, report: RuleReport, scope: str = "full") -> str:
    def brief(n: Node) -> dict:
        d = {"id": n.id, "type": n.type, "action": n.action}
        if n.kind:
            d["kind"] = n.kind
        if n.item:
            d["item"] = n.item
        return d

    agents: dict[str, list[dict]] = {}
    for a in sorted({n.agent for n in graph.nodes.values()}):
        seq = graph.agent_sequence(a)
        agents[a] = [brief(n) for n in seq if (scope == "full" or n.type != "LOCAL")]
    cands = {}
    for need, lst in graph.candidates.items():
        if not graph.nodes[need].active:
            continue
        cands[need] = [{"pass": c["pass"], "score": c["score"], "action": graph.nodes[c["pass"]].action,
                        "used_by_other_need": c["pass"] in graph.handoff.values() and graph.handoff.get(need) != c["pass"]} for c in lst]
    payload = {
        "task": graph.task,
        "robots": agents,
        "handoffs": [{"need": n, "pass": p, "score": graph.handoff_score.get(n)} for n, p in sorted(graph.handoff.items()) if graph.nodes[n].active and graph.nodes[p].active],
        "extra_order": [{"before": a, "after": b} for a, b in graph.extra_order],
        "candidates": cands,
        "unresolved_needs": report.unresolved_needs,
        "rule_warnings": report.warnings,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def apply_op(graph: PlanGraph, op: GraphOp) -> tuple[bool, str]:
    """Validate and apply one operation; revert it if it would create a cycle."""
    nodes = graph.nodes
    snap = graph.snapshot()

    if op.op == "reassign":
        need, to_pass = op.need, op.to_pass
        if need not in nodes or nodes[need].type != "NEED" or not nodes[need].active:
            return False, "unknown or inactive NEED step"
        if to_pass not in nodes or nodes[to_pass].type != "PASS":
            return False, "unknown PASS step"
        cand = {c["pass"]: c["score"] for c in graph.candidates.get(need, [])}
        if to_pass not in cand:
            return False, "not a self-nominated candidate for this need (no new collaboration allowed)"
        if graph.handoff.get(need) == to_pass:
            return False, "already matched to that step"
        old = graph.handoff.get(need)
        occupied_need = next((n for n, p in graph.handoff.items() if p == to_pass and n != need), None)

        if occupied_need is not None:
            # A common repair is a two-way exchange: A currently owns PASS-1 and B owns
            # PASS-2, but the semantic evidence says A should receive PASS-2 and B PASS-1.
            # Apply the swap atomically instead of rejecting the first reassign as a
            # one-to-one conflict. This preserves the no-new-collaboration boundary.
            if old is None or old not in {c["pass"] for c in graph.candidates.get(occupied_need, [])}:
                return False, "PASS is occupied by another need and the requested exchange is not a valid swap"
            other_candidates = {c["pass"]: c["score"] for c in graph.candidates.get(occupied_need, [])}
            if old not in other_candidates:
                return False, "PASS is occupied and the displaced need cannot use the old provider"

            graph.handoff[need] = to_pass
            graph.handoff_score[need] = cand[to_pass]
            graph.handoff[occupied_need] = old
            graph.handoff_score[occupied_need] = other_candidates[old]
            nodes[to_pass].active = True
            nodes[old].active = True
        else:
            if old:
                nodes[old].active = False  # old provider withdraws (stays a candidate)
            nodes[to_pass].active = True   # new provider re-instates its own volunteering step
            graph.handoff[need] = to_pass
            graph.handoff_score[need] = cand[to_pass]

    elif op.op == "unmatch":
        need = op.need
        if need not in graph.handoff:
            return False, "that need has no handoff"
        nodes[graph.handoff[need]].active = False
        del graph.handoff[need]
        graph.handoff_score.pop(need, None)

    else:  # add_order
        a, b = op.before, op.after
        if a not in nodes or b not in nodes or not nodes[a].active or not nodes[b].active:
            return False, "unknown or inactive step"
        if nodes[a].agent == nodes[b].agent:
            return False, "same robot: its own order is owned by that robot"
        if reachable(graph, a, b):
            return False, "already implied by existing edges"
        graph.extra_order.append((a, b))

    if find_cycle(graph) is not None:
        graph.restore(snap)
        return False, "would create a cycle (reverted)"
    return True, "ok"


async def llm_reason(graph: PlanGraph, report: RuleReport, llm: BaseLLM, log: EventLog, *, scope: str = "full", max_ops: int = 8, max_retries: int = 2) -> list[dict]:
    raw: RawGraphOps = await call_validated(
        llm, log, phase="graph", who="reasoner", system=GRAPH_SYSTEM, user=serialize_for_llm(graph, report, scope),
        parse=RawGraphOps.model_validate, images=None, max_retries=max_retries,  # text only: no observations reach the center
    )
    records = []
    for op in raw.ops[:max_ops]:
        ok, why = apply_op(graph, op)
        rec = {**op.model_dump(exclude_none=True), "status": "applied" if ok else "rejected", "why": why}
        records.append(rec)
        log.log("graph", "reasoner", f"op_{rec['status']}", **{k: v for k, v in rec.items() if k != "status"})
    for op in raw.ops[max_ops:]:
        records.append({**op.model_dump(exclude_none=True), "status": "rejected", "why": f"more than max_ops={max_ops}"})
    return records


# =========================================================================== entry point
@dataclass
class GraphResult:
    graph: PlanGraph
    report: RuleReport            # final report (after the guard pass)
    first_report: RuleReport
    ops: list[dict]
    stats: dict


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
    graph = PlanGraph(task, plans, auction)
    initial_handoffs = len(graph.handoff)
    first = rule_verify(graph)
    log.log("graph", "rules", "verified", fixes=len(first.fixes), warnings=len(first.warnings), unresolved=len(first.unresolved_needs))

    ops: list[dict] = []
    final = first
    if use_llm:
        ops = await llm_reason(graph, first, llm, log, scope=scope, max_ops=max_ops)
        final = rule_verify(graph)  # guard: re-check structure after the LLM edits
        log.log("graph", "rules", "verified_after_llm", fixes=len(final.fixes), warnings=len(final.warnings), unresolved=len(final.unresolved_needs))

    applied = [o for o in ops if o["status"] == "applied"]
    changed_matches = sum(1 for o in applied if o["op"] in ("reassign", "unmatch"))
    stats = {
        "ops_proposed": len(ops),
        "ops_applied": len(applied),
        "ops_rejected": len(ops) - len(applied),
        "reassigned": sum(1 for o in applied if o["op"] == "reassign"),
        "unmatched": sum(1 for o in applied if o["op"] == "unmatch"),
        "orders_added": sum(1 for o in applied if o["op"] == "add_order"),
        "rule_fixes": len(first.fixes) + (len(final.fixes) if use_llm else 0),
        "initial_handoffs": initial_handoffs,
        "final_handoffs": len(graph.handoff),
        "modification_ratio": round(changed_matches / max(1, initial_handoffs), 4),  # share of auction matches the graph changed
        "unresolved_needs": len(final.unresolved_needs),
    }
    return GraphResult(graph=graph, report=final, first_report=first, ops=ops, stats=stats)
