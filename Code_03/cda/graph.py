"""Collaboration Dependency Graph.

Nodes = steps. Edges = "src must finish before dst can finish".
    SEQ       consecutive active steps of the same robot (derived from each robot's order list)
    TRANSFER  PASS -> RECEIVE
    HELP      HELP -> ASK_HELP

Invariants (asserted in `check_invariants`):
    1. no step is ever created here
    2. no CONFIRMED edge is ever removed, and no endpoint of one is dropped
"""
from __future__ import annotations

import itertools

import networkx as nx

from .schemas import (CONFIRMED, HELP, LOCAL, PROPOSED, PROVIDER_TYPES, RECEIVE,
                      REQUEST_TYPES, SEQ, Edge, Node, Offer, TaskConfig, norm)


class PlanGraph:
    def __init__(self, cfg: TaskConfig, plans: dict[str, list[Node]], collab: list[Edge],
                 offers: dict[str, Offer] | None = None, judgments: list[dict] | None = None):
        self.cfg = cfg
        self.offers = offers or {}
        self.nodes: dict[str, Node] = {n.id: n for a in cfg.ids for n in plans[a]}
        self.order: dict[str, list[str]] = {a: [n.id for n in plans[a]] for a in cfg.ids}
        self.collab: dict[tuple[str, str], Edge] = {e.key: e for e in collab}
        self.reasons = {(j.get("provider"), j["request"]): j.get("reason", "")
                        for j in (judgments or []) if j.get("provider")}
        self._initial_nodes = set(self.nodes)
        self._confirmed = {k for k, e in self.collab.items() if e.status == CONFIRMED}
        self.ops_log: list[dict] = []
        self.drops: list[dict] = []
        self.unresolved: list[dict] = []
        self.selected_by_graph: set[str] = set()     # request ids whose provider the graph chose
        self.selected_by_rule: set[str] = set()      # single volunteer kept by rule

    # ------------------------------------------------------------ basic views
    def active(self, nid: str) -> bool:
        return nid in self.nodes and self.nodes[nid].status == "active"

    def seq_edges(self) -> list[Edge]:
        out = []
        for ids in self.order.values():
            act = [i for i in ids if self.active(i)]
            out += [Edge(a, b, SEQ) for a, b in zip(act, act[1:])]
        return out

    def collab_edges(self, only_active: bool = True) -> list[Edge]:
        return [e for e in self.collab.values()
                if not only_active or (self.active(e.src) and self.active(e.dst))]

    def edges(self) -> list[Edge]:
        return self.seq_edges() + self.collab_edges()

    def nx(self) -> nx.DiGraph:
        g = nx.DiGraph()
        g.add_nodes_from(i for i in self.nodes if self.active(i))
        for e in self.edges():
            g.add_edge(e.src, e.dst, kind=e.kind, status=e.status)
        return g

    def providers_of(self, rid: str) -> list[Edge]:
        return [e for e in self.collab.values() if e.dst == rid and self.active(e.src)]

    def has_confirmed(self, nid: str) -> bool:
        return any(e.status == CONFIRMED and nid in e.key for e in self.collab.values())

    # ------------------------------------------------------------ primitive mutations
    def drop(self, nid: str, reason: str, by: str) -> str | None:
        n = self.nodes.get(nid)
        if n is None or n.status != "active":
            return "node not active"
        if n.type in REQUEST_TYPES:
            return "request steps cannot be dropped"
        if self.has_confirmed(nid):
            return "node has a CONFIRMED edge"
        n.status = "dropped"
        for k in [k for k in self.collab if nid in k]:
            del self.collab[k]
        self.drops.append({"node": nid, "agent": n.agent, "type": n.type, "text": n.text(),
                           "reason": reason, "by": by})
        return None

    def move(self, nid: str, after: str) -> str | None:
        n = self.nodes.get(nid)
        if n is None or not self.active(nid) or n.type not in PROVIDER_TYPES:
            return "only active HELP/PASS steps can be moved"
        ids = self.order[n.agent]
        if after != "START" and (after not in ids or after == nid):
            return f"'after' must be START or another step of {n.agent}"
        ids.remove(nid)
        ids.insert(0 if after == "START" else ids.index(after) + 1, nid)
        return None

    # ------------------------------------------------------------ 4b: rule layer
    def apply_rules(self) -> None:
        """Deterministic cleanup that needs no judgment."""
        for n in list(self.nodes.values()):
            if n.status == "active" and n.type == LOCAL and n.violations:
                self.drop(n.id, "capability check failed: " + "; ".join(n.violations), "rule")
        for rid, r in self.nodes.items():
            if r.type not in REQUEST_TYPES or not self.active(rid):
                continue
            provs = self.providers_of(rid)
            if any(e.status == CONFIRMED for e in provs):
                for e in provs:
                    if e.status == PROPOSED:
                        self.drop(e.src, "target robot accepted the request", "rule")
            elif len(provs) == 1 and rid not in self.selected_by_graph:
                self.selected_by_rule.add(rid)

    def detect_issues(self) -> list[dict]:
        issues: list[dict] = []
        cid = itertools.count(1)
        for rid, r in self.nodes.items():
            if r.type not in REQUEST_TYPES or not self.active(rid):
                continue
            provs = self.providers_of(rid)
            if len(provs) >= 2:
                issues.append({"id": f"I{next(cid)}", "issue": "MULTIPLE_VOLUNTEERS",
                               "request": r.brief() | {"location": r.location},
                               "target_reply": "rejected or no answer",
                               "candidates": [self._cand(e) for e in provs]})
            elif len(provs) == 1 and r.type == RECEIVE:
                e = provs[0]
                p = self.nodes[e.src]
                if norm(p.item) != norm(r.item) and e.status == PROPOSED:
                    issues.append({"id": f"I{next(cid)}", "issue": "ITEM_MISMATCH",
                                   "request": r.brief(), "provider": self._cand(e)})
        g = self.nx()
        for cyc in itertools.islice(nx.simple_cycles(g), 5):
            pairs = list(zip(cyc, cyc[1:] + cyc[:1]))
            issues.append({
                "id": f"I{next(cid)}", "issue": "CYCLE",
                "nodes": [self.nodes[i].brief() for i in cyc],
                "edges": [{"src": a, "dst": b, **g.edges[a, b]} for a, b in pairs],
                "movable": [{"node": i, "own_plan_order": [x for x in self.order[self.nodes[i].agent]
                                                           if self.active(x)]}
                            for i in cyc if self.nodes[i].type in PROVIDER_TYPES],
            })
        return issues

    def _cand(self, e: Edge) -> dict:
        p = self.nodes[e.src]
        off = self.offers.get(p.agent)
        return p.brief() | {"edge": e.status, "capability": off.capability if off else "",
                            "reason": self.reasons.get((e.src, e.dst), "")}

    # ------------------------------------------------------------ 4c/4d: LLM ops
    def apply_op(self, op: dict, issues: list[dict]) -> str | None:
        kind = op.get("op")
        by_req = {i["request"]["id"]: i for i in issues if i["issue"] == "MULTIPLE_VOLUNTEERS"}
        mentioned = _ids_in(issues)
        if kind == "connect":
            src, dst = op.get("src"), op.get("dst")
            iss = by_req.get(dst)
            if iss is None:
                return "connect only allowed on a MULTIPLE_VOLUNTEERS request"
            cands = {c["id"] for c in iss["candidates"]}
            if src not in cands:
                return f"src must be one of the candidates {sorted(cands)}"
            for c in cands - {src}:
                err = self.drop(c, f"graph chose {src} for {dst}", "graph")
                if err:
                    return err
            self.collab[(src, dst)].source = "graph"
            self.selected_by_graph.add(dst)
            return None
        if kind == "disconnect":
            k = (op.get("src"), op.get("dst"))
            e = self.collab.get(k)
            if e is None or k[0] not in mentioned:
                return "edge not found among the issues"
            if e.status == CONFIRMED:
                return "CONFIRMED edges cannot be removed"
            return self.drop(k[0], "graph disconnected: " + str(op.get("why", "")), "graph")
        if kind == "move":
            nid = op.get("node")
            if nid not in mentioned:
                return "node not mentioned in the issues"
            return self.move(nid, str(op.get("after", "START")))
        if kind == "drop":
            nid = op.get("node")
            if nid not in mentioned:
                return "node not mentioned in the issues"
            return self.drop(nid, "graph: " + str(op.get("why", "")), "graph")
        if kind == "unresolved":
            self.unresolved.append({"issue": op.get("issue"), "why": op.get("why", ""), "by": "graph"})
            return None
        return f"unknown op '{kind}'"

    # ------------------------------------------------------------ 4e: finalize + schedule
    def finalize(self) -> None:
        """Release leftover ambiguity, block what cannot run, then schedule."""
        for rid, r in self.nodes.items():           # still >1 volunteer -> ambiguous, release all
            if r.type in REQUEST_TYPES and self.active(rid):
                provs = self.providers_of(rid)
                if len(provs) >= 2:
                    for e in provs:
                        self.drop(e.src, "ambiguous volunteers left unresolved", "rule")
                    self.unresolved.append({"request": rid, "why": "multiple volunteers, no choice",
                                            "by": "rule"})
        g = self.nx()
        blocked_roots = set()
        for rid, r in self.nodes.items():
            if r.type in REQUEST_TYPES and self.active(rid) and not self.providers_of(rid):
                blocked_roots.add(rid)
                self.unresolved.append({"request": rid, "agent": r.agent, "text": r.text(),
                                        "why": "no robot provides this", "by": "rule"})
        for cyc in nx.simple_cycles(g):
            blocked_roots |= set(cyc)
            self.unresolved.append({"cycle": cyc, "why": "dependency cycle remains", "by": "rule"})
        to_block = set(blocked_roots)
        for b in blocked_roots:
            to_block |= nx.descendants(g, b)
        for b in to_block:
            self.nodes[b].status = "blocked"
        self.schedule()

    def schedule(self) -> None:
        g = self.nx()
        for nid in nx.topological_sort(g):
            n = self.nodes[nid]
            seq_preds = [p for p in g.predecessors(nid) if g.edges[p, nid]["kind"] == SEQ]
            collab_preds = [p for p in g.predecessors(nid) if g.edges[p, nid]["kind"] != SEQ]
            n.travel = self.cfg.travel_min if (n.type == HELP and norm(n.location) !=
                                               norm(self.cfg.agent(n.agent).profile.room)) else 0
            n.t_start = max([self.nodes[p].t_end for p in seq_preds] or [0])
            if n.type in REQUEST_TYPES:
                ready = max([n.t_start] + [self.nodes[p].t_end for p in collab_preds])
                n.t_end = ready + (n.duration if n.type == RECEIVE else 0)
            else:
                n.t_end = n.t_start + n.travel + n.duration

    # ------------------------------------------------------------ checks / export
    def check_invariants(self) -> None:
        assert set(self.nodes) <= self._initial_nodes, "graph stage created a step"
        for k in self._confirmed:
            assert k in self.collab, f"CONFIRMED edge {k} was removed"
            assert all(self.nodes[i].status != "dropped" for i in k), f"endpoint of CONFIRMED {k} dropped"

    def makespan(self) -> int:
        ends = [n.t_end for n in self.nodes.values() if n.status == "active" and n.t_end is not None]
        return max(ends or [0])

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return {"nodes": [asdict(n) for n in self.nodes.values()],
                "order": self.order,
                "edges": [asdict(e) for e in self.seq_edges() + list(self.collab.values())],
                "drops": self.drops, "unresolved": self.unresolved, "ops_log": self.ops_log}


def _ids_in(obj) -> set[str]:
    out: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("id", "src", "dst", "node") and isinstance(v, str):
                out.add(v)
            out |= _ids_in(v)
    elif isinstance(obj, list):
        for v in obj:
            out |= _ids_in(v) if not isinstance(v, str) else ({v} if "_s" in v else set())
    return out


__all__ = ["PlanGraph"]
