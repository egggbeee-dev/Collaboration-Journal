"""Collaboration Dependency Graph.

Nodes = steps. Edges = "src must finish before dst can finish".
    SEQ       consecutive active steps of the same robot (derived from each robot's order list)
    TRANSFER  PASS -> RECEIVE
    HELP      HELP -> ASK_HELP
    ORDER     added here: an existing step must finish before another robot's step starts
              (same space / same object), decided by the Graph LLM on code-given candidates

Collaboration edges arrive from Stage 3 as CANDIDATES (PROPOSED). Here they are confirmed:
    one candidate for a request -> rule confirms it
    several (or another robot's existing work could cover it) -> the Graph LLM selects one

Invariants (asserted in `check_invariants`):
    1. no step is ever created here
    2. every agreement CONFIRMED here (rule or graph) still holds at the end.
"""
from __future__ import annotations

import itertools
import re

import networkx as nx

from .schemas import (ASK_HELP, CONFIRMED, HELP, HELP_EDGE, LOCAL, ORDER, PASS, PHASE_MIN, PROPOSED,
                      PROVIDER_TYPES, RECEIVE, REQUEST_TYPES, SEQ, Edge, Node, Offer, TaskConfig, norm)

_STOP = set("a an the to of on in at into onto from for with and or by it its them their this that "
            "your my our be is are up down near next robot".split())


def _tokens(text: str) -> set[str]:
    out = set()
    for w in re.findall(r"[a-z]+", (text or "").lower()):
        if w in _STOP or len(w) < 3:
            continue
        out.add(w[:-1] if w.endswith("s") and len(w) > 3 else w)
    return out


def _jaccard(a: str, b: str) -> float:
    x, y = _tokens(a), _tokens(b)
    return len(x & y) / len(x | y) if x and y else 0.0


# "move the left red chair from the workout area to the wall" -> "left red chair"
_OBJ = re.compile(r"^\s*(?:use\s+\S+(?:\s+\S+)?\s+(?:capability\s+)?to\s+)?[a-z]+(?:\s+(?:up|out|away|back|on|off))?\s+"
                  r"(?:the\s+|a\s+|an\s+)?(.+?)(?:\s+(?:from|to|onto|into|on|at|in|out of|off|away|"
                  r"against|beside|next to|near|under|so|and|without|for|with)\b|$)", re.I)


def object_of(action: str) -> set[str]:
    """Tokens of the object a step acts on (the phrase after the verb, before the first place word)."""
    m = _OBJ.search(action or "")
    return _tokens(m.group(1)) if m else set()


def same_object(a: str, b: str) -> bool | None:
    """True/False if both objects can be read; None if one cannot (fall back to whole-text similarity)."""
    x, y = object_of(a), object_of(b)
    if not x or not y:
        return None
    return x <= y or y <= x or len(x & y) / len(x | y) >= 0.8   # "left chair" vs "right chair" differ


_TRANSFORM = re.compile(r"\b(fill|refill|rinse|wash|clean|wipe|dry|heat|warm|cool|chill|cut|slice|peel|chop|"
                        r"cook|toast|brew|pour|unwrap|unpack|fold|charge|switch on|turn on|toggle)\w*\b", re.I)


def is_real_preparation(action: str) -> bool:
    """Changes the object's state (fill, rinse, heat, cut, ...), not only where it lies."""
    return bool(_TRANSFORM.search(action or ""))


DUP_THRESHOLD = 0.5
RELEASE_THRESHOLD = 0.6     # stricter: used without an LLM, only for steps that cannot run anyway
WORK_TYPES = {LOCAL, HELP}


class PlanGraph:
    def __init__(self, cfg: TaskConfig, plans: dict[str, list[Node]], collab: list[Edge],
                 offers: dict[str, Offer] | None = None, judgments: list[dict] | None = None):
        self.cfg = cfg
        self.offers = offers or {}
        self.nodes: dict[str, Node] = {n.id: n for a in cfg.ids for n in plans[a]}
        self.order: dict[str, list[str]] = {a: [n.id for n in plans[a]] for a in cfg.ids}
        self.collab: dict[tuple[str, str], Edge] = {e.key: e for e in collab}
        judgments = [j for j in (judgments or []) if j.get("request")]     # offer records handled below
        self.reasons = {(j.get("provider"), j["request"]): j.get("reason", "")
                        for j in judgments if j.get("provider")}
        self.replies: dict[str, list[str]] = {}          # request id -> why others did not serve it
        for j in judgments or []:
            if not j.get("provider") and j.get("decision") not in (None, "ignore"):
                self.replies.setdefault(j["request"], []).append(
                    f"{j['agent']}: {j.get('result', j.get('decision'))} ({j.get('reason', '')})")
        self.target_reply: dict[str, str] = {}          # request id -> what its target answered
        for j in judgments or []:
            r = self.nodes.get(j["request"])
            if r is not None and j.get("agent") == r.target:
                self.target_reply[j["request"]] = str(j.get("result") or j.get("decision") or "")
        self.n_released = 0
        self.n_tightened = 0
        self.n_prep_dropped = 0
        self.n_offers_untaken = 0
        self.n_prep_released = 0
        self.request_outcome: dict[str, str] = {}        # request id -> declined | failed | moot
        self._initial_nodes = set(self.nodes)
        self._agreements: set[tuple[str, str]] = set()       # (provider agent, request id), confirmed here
        self.n_candidates = len(collab)
        self.order_edges: list[Edge] = []                    # ORDER edges added by the Graph LLM
        self.dismissed_orders: set[frozenset] = set()        # pairs the Graph LLM judged independent
        self.n_order_added = 0
        self.n_order_independent = 0
        self.size_source = None
        self.n_size_missing = 0
        self.ops_log: list[dict] = []
        self.drops: list[dict] = []
        self.unresolved: list[dict] = []
        self.warnings: list[dict] = []
        self.dismissed_dups: set[frozenset] = set()
        self.n_merged = 0
        self.n_dup_dropped = 0
        for n in self.nodes.values():          # requests withdrawn in Stage 2 (no own step needed them)
            if n.status == "dropped":
                self.drops.append({"node": n.id, "agent": n.agent, "type": n.type, "text": n.text(),
                                   "reason": "withdrawn in Stage 2: " + "; ".join(n.violations),
                                   "by": "stage2-check"})
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
        order = [e for e in self.order_edges if self.active(e.src) and self.active(e.dst)]
        return self.seq_edges() + self.collab_edges() + order

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

    # ------------------------------------------------------------ confirming candidates
    def confirm(self, e: Edge, by: str) -> None:
        e.status = CONFIRMED
        e.source = by
        self._agreements.add((self.nodes[e.src].agent, e.dst))
        (self.selected_by_graph if by == "graph" else self.selected_by_rule).add(e.dst)

    def release(self, e: Edge, reason: str, by: str) -> str | None:
        """A candidate that was not selected. A provider step that exists only for this answer
        (accept / volunteer) is dropped; a step the robot does anyway (already_doing, own offer that
        still serves something) only loses this edge."""
        if e.status == CONFIRMED:
            return "CONFIRMED edges cannot be released"
        p = self.nodes[e.src]
        self.collab.pop(e.key, None)
        if p.origin in ("accept", "volunteer") and not any(x.src == p.id for x in self.collab.values()):
            return self.drop(p.id, reason, by)
        return None

    def existing_work(self, r: Node) -> list[str]:
        """Another robot's existing LOCAL/HELP step in the requester's room that acts on the same
        object as an ASK_HELP (it may already do what is asked)."""
        if r.type != ASK_HELP:
            return []
        linked = {e.src for e in self.collab.values() if e.dst == r.id}
        return [n.id for n in self.nodes.values()
                if n.status == "active" and n.type in WORK_TYPES and n.agent != r.agent
                and n.id not in linked and norm(n.location) == norm(r.location)
                and same_object(n.action, r.action) is True]

    # ------------------------------------------------------------ primitive mutations
    def enabling_requests(self, nid: str) -> list[Node]:
        return [r for r in self.nodes.values() if r.type in REQUEST_TYPES and r.enables == nid
                and r.status == "active"]

    def drop(self, nid: str, reason: str, by: str) -> str | None:
        """Release a step. Dropping a LOCAL step also withdraws the requests that existed only to
        enable it (a request is a dependency of its own step, nothing else). Refused if that would
        break an agreement another robot already made (a CONFIRMED provider)."""
        n = self.nodes.get(nid)
        if n is None or n.status != "active":
            return "node not active"
        if n.type in REQUEST_TYPES:
            return "request steps cannot be dropped"
        if self.has_confirmed(nid):
            return "node has a CONFIRMED edge"
        reqs = self.enabling_requests(nid)
        if any(e.status == CONFIRMED for r in reqs for e in self.providers_of(r.id)):
            return "another robot already agreed to supply this step's inputs"
        n.status = "dropped"
        for k in [k for k in self.collab if nid in k]:
            del self.collab[k]
        self.drops.append({"node": nid, "agent": n.agent, "type": n.type, "text": n.text(),
                           "reason": reason, "by": by})
        for r in reqs:                                    # withdraw now-moot requests (+ volunteers)
            for e in self.providers_of(r.id):
                self.drop(e.src, f"request {r.id} withdrawn", by)
            r.status = "dropped"
            self.drops.append({"node": r.id, "agent": r.agent, "type": r.type, "text": r.text(),
                               "reason": f"withdrawn: the step it enabled ({nid}) was released", "by": by})
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

    def merge(self, nid: str, into: str) -> str | None:
        a, b = self.nodes.get(nid), self.nodes.get(into)
        if a is None or b is None or nid == into or not self.active(nid) or not self.active(into):
            return "both nodes must be active and different"
        if a.agent != b.agent:
            return "merge only inside one robot; use drop across robots"
        if a.type not in WORK_TYPES or b.type not in WORK_TYPES:
            return "merge only LOCAL/HELP steps"
        if a.type in REQUEST_TYPES:
            return "request steps cannot be merged"
        for k in [k for k in self.collab if k[0] == nid]:
            e = self.collab.pop(k)
            if (into, e.dst) not in self.collab:
                self.collab[(into, e.dst)] = Edge(into, e.dst, e.kind, e.status, "graph")
        for k in [k for k in self.collab if k[1] == nid]:
            del self.collab[k]
        a.status = "dropped"
        self.drops.append({"node": nid, "agent": a.agent, "type": a.type, "text": a.text(),
                           "reason": f"duplicate of {into}, merged", "by": "graph"})
        self.n_merged += 1
        return None

    def _release_prep(self, p: Node, reason: str) -> None:
        prep = self.nodes.get(p.prepared_by) if p.prepared_by else None
        if prep is not None and prep.status == "active" and self.drop(prep.id, reason, "rule") is None:
            self.n_prep_released += 1

    def _pass_after_prep(self) -> None:
        """An offered PASS can only happen after the step that prepared its object."""
        for p in self.nodes.values():
            if p.type != PASS or not p.prepared_by or not self.active(p.id) or not self.active(p.prepared_by):
                continue
            ids = self.order[p.agent]
            if ids.index(p.id) < ids.index(p.prepared_by):
                ids.remove(p.id)
                ids.insert(ids.index(p.prepared_by) + 1, p.id)

    # ------------------------------------------------------------ 4b: rule layer
    def _link_preparations(self) -> None:
        """An offered PASS without "prepared_by": if an earlier LOCAL step of the same robot handles that
        object (e.g. "rinse the Metal tumbler and fill it with water"), it is the preparation."""
        for p in self.nodes.values():
            if p.type != PASS or p.prepared_by or p.status != "active":
                continue
            item = _tokens(p.item or "")
            ids = [x for x in self.order[p.agent] if self.active(x)]
            for x in reversed(ids[:ids.index(p.id)] if p.id in ids else []):
                n = self.nodes[x]
                if n.type == LOCAL and item and len(item & _tokens(n.action)) >= min(2, len(item)) \
                        and is_real_preparation(n.action):
                    p.prepared_by = x
                    break
            else:                                           # written after the PASS: look anywhere
                for x in ids:
                    n = self.nodes[x]
                    if n.type == LOCAL and item and len(item & _tokens(n.action)) >= min(2, len(item)) and \
                            is_real_preparation(n.action) and \
                            not any(q.prepared_by == x for q in self.nodes.values() if q.type == PASS):
                        p.prepared_by = x
                        break

    def apply_rules(self) -> None:
        """Deterministic cleanup that needs no judgment."""
        self._link_preparations()
        for n in list(self.nodes.values()):
            if n.status == "active" and n.type == PASS and n.origin == "offer" and \
                    not any(e.src == n.id for e in self.collab.values()):
                if self.drop(n.id, f"offer not taken by {n.target}", "rule") is None:
                    self.n_offers_untaken += 1
                    self._release_prep(n, f"prepared only for {n.id}, an offer {n.target} did not take")
            if n.status == "active" and n.type == LOCAL and n.violations:
                self.drop(n.id, "capability check failed: " + "; ".join(n.violations), "rule")
        for rid, r in self.nodes.items():
            if r.type not in REQUEST_TYPES or not self.active(rid):
                continue
            provs = self.providers_of(rid)
            conf = [e for e in provs if e.status == CONFIRMED]
            if conf:
                for e in provs:
                    if e.status == PROPOSED:
                        self.release(e, f"{self.nodes[conf[0].src].agent} was selected for {rid}", "rule")
            elif len(provs) == 1 and not self.existing_work(r):
                self.confirm(provs[0], "rule")              # a single candidate: nothing to choose

    def detect_issues(self) -> list[dict]:
        issues: list[dict] = []
        cid = itertools.count(1)
        for rid, r in self.nodes.items():
            if r.type not in REQUEST_TYPES or not self.active(rid):
                continue
            provs = self.providers_of(rid)
            existing = self.existing_work(r) if not any(e.status == CONFIRMED for e in provs) else []
            if not any(e.status == CONFIRMED for e in provs) and (len(provs) >= 2 or (provs and existing)):
                issues.append({"id": f"I{next(cid)}", "issue": "MULTIPLE_CANDIDATES",
                               "request": r.brief() | {"location": r.location},
                               "candidates": [self._cand(e) for e in provs]
                               + [self.nodes[x].brief() | {"existing_work": True,
                                                           "location": self.nodes[x].location}
                                  for x in existing]})
            elif len(provs) == 1 and r.type == RECEIVE:
                e = provs[0]
                p = self.nodes[e.src]
                if norm(p.item) != norm(r.item) and e.status == PROPOSED:
                    issues.append({"id": f"I{next(cid)}", "issue": "ITEM_MISMATCH",
                                   "request": r.brief(), "provider": self._cand(e)})
        work = [n for n in self.nodes.values() if n.status == "active" and n.type in WORK_TYPES]
        for x, y in itertools.combinations(work, 2):
            pair = frozenset((x.id, y.id))
            if pair in self.dismissed_dups:
                continue
            if x.agent == y.agent and x.origin == "local" and y.origin == "local":
                continue           # a robot's own planned steps are intentional (e.g. two different chairs)
            sx = {e.dst for e in self.collab.values() if e.src == x.id}
            sy = {e.dst for e in self.collab.values() if e.src == y.id}
            if sx & sy:            # competing answers to one request -> MULTIPLE_CANDIDATES handles it
                continue
            if x.agent == y.agent and sx and sy:
                continue           # one robot answering two different requests: two different jobs
            if x.location and y.location and norm(x.location) != norm(y.location):
                continue           # work in different rooms is never the same work
            obj = same_object(x.action, y.action)
            if obj is False:
                continue           # different objects (e.g. the left chair vs the right chair)
            sim = _jaccard(x.action, y.action)
            if sim >= DUP_THRESHOLD or obj is True and sim >= DUP_THRESHOLD - 0.2:
                issues.append({"id": f"I{next(cid)}", "issue": "DUPLICATE_WORK",
                               "similarity": round(sim, 2),
                               "nodes": [self._work_view(x), self._work_view(y)],
                               "same_robot": x.agent == y.agent})
        g = self.nx()
        issues += self._order_candidates(g, cid)
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

    def plan_view(self) -> dict:
        """Compact joint plan for the Graph LLM: each robot's active steps in its own order."""
        return {a: [self.nodes[i].brief() for i in ids if self.active(i)] for a, ids in self.order.items()}

    def _order_candidates(self, g, cid, limit: int = 12) -> list[dict]:
        """Pairs of steps of DIFFERENT robots in the same room that touch a shared object and have no
        order between them yet (e.g. R4 moves the coffee table while R2 clears what is on it).
        Only confirmed HELP steps are paired; the Graph LLM decides 'A before B' or 'independent'."""
        out = []
        helps = [self.nodes[e.src] for e in self.collab.values()
                 if e.status == CONFIRMED and e.kind == HELP_EDGE and self.active(e.src) and self.active(e.dst)
                 and self.nodes[e.src].type == HELP]
        others = [n for n in self.nodes.values() if n.status == "active" and n.type in WORK_TYPES]
        seen = set()
        for h in helps:
            for s in others:
                pair = frozenset((h.id, s.id))
                if s.agent == h.agent or pair in seen or pair in self.dismissed_orders:
                    continue
                if norm(s.location) != norm(h.location):
                    continue
                shared = (object_of(h.action) & _tokens(s.action)) | (object_of(s.action) & _tokens(h.action))
                if not shared:
                    continue
                if nx.has_path(g, h.id, s.id) or nx.has_path(g, s.id, h.id):
                    continue
                seen.add(pair)
                out.append({"id": f"I{next(cid)}", "issue": "ORDER_CANDIDATE",
                            "steps": [h.brief() | {"location": h.location}, s.brief() | {"location": s.location}],
                            "shared": sorted(shared)})
                if len(out) >= limit:
                    return out
        return out

    def _work_view(self, n: Node) -> dict:
        serves = [{"request": e.dst, "edge": e.status} for e in self.collab.values() if e.src == n.id]
        inputs = []
        for r in self.enabling_requests(n.id):
            provs = self.providers_of(r.id)
            inputs.append(r.brief() | {"served_by": [e.src for e in provs] or None,
                                       "replies": self.replies.get(r.id, [])})
        return n.brief() | {"location": n.location, "serves": serves, "inputs": inputs,
                            "can_run": all(i["served_by"] for i in inputs),
                            "has_confirmed": self.has_confirmed(n.id)}

    def _cand(self, e: Edge) -> dict:
        p = self.nodes[e.src]
        off = self.offers.get(p.agent)
        return p.brief() | {"targeted": e.targeted, "capability": off.capability if off else "",
                            "own_plan_length": sum(1 for x in self.order[p.agent] if self.active(x)),
                            "reason": self.reasons.get((e.src, e.dst), "")}

    # ------------------------------------------------------------ 4c/4d: LLM ops
    def apply_op(self, op: dict, issues: list[dict]) -> str | None:
        op = normalize_op(op)
        kind = op.get("op")
        by_req = {i["request"]["id"]: i for i in issues if i["issue"] == "MULTIPLE_CANDIDATES"}
        mentioned = _ids_in(issues)
        if kind == "connect":
            src, dst = op.get("src"), op.get("dst")
            iss = by_req.get(dst)
            if iss is None:
                return "connect only allowed on a MULTIPLE_CANDIDATES request"
            cands = {c["id"]: c for c in iss["candidates"]}
            if src not in cands:
                return f"src must be one of the candidates {sorted(cands)}"
            if cands[src].get("existing_work"):        # another robot's existing step already does it
                e = Edge(src, dst, HELP_EDGE, PROPOSED, "graph")
                self.collab[e.key] = e
            for e in [e for e in self.providers_of(dst) if e.src != src]:
                err = self.release(e, f"graph selected {src} for {dst}", "graph")
                if err:
                    return err
            self.confirm(self.collab[(src, dst)], "graph")
            return None
        if kind == "disconnect":
            k = (op.get("src"), op.get("dst"))
            e = self.collab.get(k)
            if e is None or k[0] not in mentioned:
                return "edge not found among the issues"
            return self.release(e, "graph disconnected: " + str(op.get("why", "")), "graph")
        if kind == "order":
            a, b = op.get("before"), op.get("after")
            iss = next((i for i in issues if i["issue"] == "ORDER_CANDIDATE"
                        and {a, b} == {x["id"] for x in i["steps"]}), None)
            if iss is None:
                return "order only allowed on an ORDER_CANDIDATE pair"
            e = Edge(a, b, ORDER, CONFIRMED, "graph")
            self.order_edges.append(e)
            if not nx.is_directed_acyclic_graph(self.nx()):
                self.order_edges.remove(e)
                return "this order would create a cycle"
            self.n_order_added += 1
            return None
        if kind == "independent":
            iss = next((i for i in issues if i["id"] == op.get("issue")), None)
            if iss is None or iss["issue"] != "ORDER_CANDIDATE":
                return "independent only allowed on an ORDER_CANDIDATE issue"
            self.dismissed_orders.add(frozenset(x["id"] for x in iss["steps"]))
            self.n_order_independent += 1
            return None
        if kind == "move":
            nid = op.get("node")
            if nid not in mentioned:
                return "node not mentioned in the issues"
            return self.move(nid, str(op.get("after", "START")))
        if kind == "drop":
            nid = op.get("node")
            if nid not in mentioned:
                return "node not mentioned in the issues"
            err = self.drop(nid, "graph: " + str(op.get("why", "")), "graph")
            if err is None and any(i["issue"] == "DUPLICATE_WORK" and nid in _ids_in(i) for i in issues):
                self.n_dup_dropped += 1
            return err
        if kind == "merge":
            nid, into = op.get("node"), op.get("into")
            if nid not in mentioned or into not in mentioned:
                return "nodes not mentioned in the issues"
            return self.merge(nid, into)
        if kind == "unresolved":
            iss = next((i for i in issues if i["id"] == op.get("issue")), None)
            if iss and iss["issue"] == "DUPLICATE_WORK":
                self.dismissed_dups.add(frozenset(n["id"] for n in iss["nodes"]))
                self.warnings.append({"issue": "DUPLICATE_WORK kept", "nodes": [n["id"] for n in iss["nodes"]],
                                      "why": op.get("why", "")})
                return None
            self.unresolved.append({"issue": op.get("issue"), "why": op.get("why", ""), "by": "graph"})
            return None
        return f"unknown op '{kind}'"

    # ------------------------------------------------------------ 4e: finalize + schedule
    def finalize(self) -> None:
        """Release leftover ambiguity, block what cannot run, then schedule."""
        for rid, r in self.nodes.items():           # candidates the graph did not settle
            if r.type not in REQUEST_TYPES or not self.active(rid):
                continue
            provs = self.providers_of(rid)
            if not provs or any(e.status == CONFIRMED for e in provs):
                continue
            pick = [e for e in provs if e.targeted] if len(provs) >= 2 else provs
            if len(pick) == 1:                                     # fallback: the robot that was asked
                for e in provs:
                    if e is not pick[0]:
                        self.release(e, f"{self.nodes[pick[0].src].agent} was asked directly", "rule")
                self.confirm(pick[0], "rule")
            else:
                for e in provs:
                    self.release(e, "ambiguous candidates left unresolved", "rule")
                self.unresolved.append({"request": rid, "why": "multiple candidates, no choice", "by": "rule"})
        self._tighten_requests()
        self._release_redundant_branches()
        for iss in self.detect_issues():
            if iss["issue"] == "DUPLICATE_WORK":
                self.warnings.append({"issue": "possible DUPLICATE_WORK left", "nodes": [n["id"] for n in iss["nodes"]],
                                      "similarity": iss["similarity"]})
        self._drop_unrequested_handoff_prep()
        self._pass_after_prep()
        roots: list[tuple[str, str]] = []             # (node id, cause)
        g = self.nx()
        for rid, r in self.nodes.items():
            if r.type in REQUEST_TYPES and self.active(rid) and not self.providers_of(rid):
                reply = self.target_reply.get(rid, "")
                declined = reply.startswith("reject") or reply.startswith("rejected_by_check")
                cause = "declined" if declined else "failed"
                self.request_outcome[rid] = cause
                roots.append((rid, cause))
                self.unresolved.append({"request": rid, "agent": r.agent, "text": r.text(), "outcome": cause,
                                        "why": f"{r.target} declined: {reply}" if declined
                                        else "no robot provides this", "by": "rule"})
        for cyc in nx.simple_cycles(g):
            roots += [(c, "failed") for c in cyc]
            self.unresolved.append({"cycle": cyc, "why": "dependency cycle remains", "by": "rule"})
        self._propagate_block(roots)
        self.schedule()

    def _propagate_block(self, roots: list[tuple[str, str]]) -> None:
        """Block only what really depends on a failed request (not the robot's later, unrelated steps):
        request -> the step it enables -> requests that step was serving (other robots) -> ...
        Other inputs of a blocked step are no longer needed: those requests and their providers are
        'skipped'. The robot simply skips blocked/skipped steps and continues its own sequence."""
        queue = list(roots)
        while queue:
            nid, cause = queue.pop()
            n = self.nodes.get(nid)
            if n is None or n.status != "active":
                continue
            n.status = "blocked"
            n.violations = n.violations + [f"blocked ({cause})"]
            if n.type in REQUEST_TYPES:
                if n.enables:
                    queue.append((n.enables, cause))
                else:                                         # no dependency info: block what follows
                    ids = self.order[n.agent]
                    queue += [(x, cause) for x in ids[ids.index(nid) + 1:]]
                continue
            for e in [e for e in self.collab.values() if e.src == nid]:      # it served other robots
                queue.append((e.dst, cause))
                self.request_outcome.setdefault(e.dst, cause)
            for r in self.enabling_requests(nid):                             # its other inputs: moot
                r.status = "skipped"
                self.request_outcome[r.id] = "moot"
                for e in self.providers_of(r.id):
                    p = self.nodes[e.src]
                    if not any(self.active(x.dst) for x in self.collab.values() if x.src == p.id):
                        p.status = "skipped"
                        if p.type == PASS:
                            self._release_prep(p, f"prepared only for {p.id}, which is no longer needed")
                        self.drops.append({"node": p.id, "agent": p.agent, "type": p.type, "text": p.text(),
                                           "reason": f"not needed: {r.id} is moot ({nid} cannot run)",
                                           "by": "rule"})

    _PREP = re.compile(r"\b(pick-?up|hand-?off|hand over|handover|collection|pass(ing)?|transfer|staging)\s+"
                       r"(area|spot|point|zone|station|location|position|place|counter|shelf)\b"
                       r"|\bfor (collection|pick-?up|hand-?off|hand-?over|delivery|transfer)\b"
                       r"|\b(side of|by|near|beside|next to|at) the (\w+ ){0,2}door(way)?\b", re.I)

    def _drop_unrequested_handoff_prep(self) -> None:
        """PASS already includes picking the object up. A LOCAL step that only stages an object at a
        pickup/handoff spot is either redundant (a PASS exists) or serves nobody: release it."""
        preps = {p.prepared_by for p in self.nodes.values() if p.type == PASS and p.status == "active"}
        givers = {p.agent for p in self.nodes.values() if p.type == PASS}
        receivers = {r.agent for r in self.nodes.values() if r.type == RECEIVE}
        for n in list(self.nodes.values()):
            if n.agent in receivers and n.agent not in givers:   # pure receivers do not stage handoffs
                continue
            if n.id in preps and is_real_preparation(n.action):   # real preparation of an offer: keep it
                continue
            if n.status == "active" and n.type == LOCAL and self._PREP.search(n.action or ""):
                if self.drop(n.id, "only moves the object to a handoff spot: the PASS itself picks it up "
                                   "and hands it over", "rule") is None:
                    self.n_prep_dropped += 1
                    for p in self.nodes.values():
                        if p.type == PASS and p.prepared_by == n.id:
                            p.prepared_by = None

    def _tighten_requests(self) -> None:
        """A request only has to be satisfied before the step it enables. If a robot wrote it earlier,
        it would sit idle waiting while it could do its independent work. Move each request to just
        before its enabled step (WHEN only; the robot's own actions and their order are unchanged).
        Reverted if it would create a dependency cycle."""
        self.n_tightened = 0
        for r in [n for n in self.nodes.values() if n.type in REQUEST_TYPES and n.status == "active"]:
            if not r.enables or not self.active(r.enables):
                continue
            ids = self.order[r.agent]
            i, j = ids.index(r.id), ids.index(r.enables)
            if j <= i + 1:
                continue
            between = [x for x in ids[i + 1:j] if self.active(x)]
            if not between:
                continue
            old = list(ids)
            ids.remove(r.id)
            ids.insert(ids.index(r.enables), r.id)
            if not nx.is_directed_acyclic_graph(self.nx()):
                self.order[r.agent] = old
                continue
            self.n_tightened += 1

    def _release_redundant_branches(self) -> None:
        """A step that cannot run (an input request nobody serves) while another robot runs the same
        work with all inputs served is redundant, not a failure: release it and its requests."""
        for r in [r for r in self.nodes.values() if r.type in REQUEST_TYPES and r.status == "active"]:
            if r.status != "active" or self.providers_of(r.id) or not r.enables:
                continue
            step = self.nodes.get(r.enables)
            if step is None or step.status != "active":
                continue
            twin = next((o for o in self.nodes.values()
                         if o.id != step.id and o.status == "active" and o.type in WORK_TYPES
                         and frozenset((o.id, step.id)) not in self.dismissed_dups   # LLM said: different
                         and norm(o.location) == norm(step.location)
                         and _jaccard(o.action, step.action) >= RELEASE_THRESHOLD
                         and all(self.providers_of(q.id) for q in self.enabling_requests(o.id))), None)
            if twin and self.drop(step.id, f"redundant: {twin.id} ({twin.agent}) does the same work "
                                           f"and can run", "rule") is None:
                self.n_released += 1

    def schedule(self) -> None:
        """Logical time: every step takes one unit; no minutes, no travel time.
        A step starts when the robot's previous step (SEQ) and any ORDER predecessor are done and, for a RECEIVE / HELP-waiting
        ASK_HELP, when the other robot's provider step is done. ASK_HELP is pure waiting: it ends
        when its HELP ends and takes no unit of its own."""
        g = self.nx()
        for nid in nx.topological_sort(g):
            n = self.nodes[nid]
            seq_preds = [p for p in g.predecessors(nid) if g.edges[p, nid]["kind"] in (SEQ, ORDER)]
            collab_preds = [p for p in g.predecessors(nid) if g.edges[p, nid]["kind"] not in (SEQ, ORDER)]
            ready = max([self.nodes[p].t_end for p in seq_preds] or [0])
            given = max([self.nodes[p].t_end for p in collab_preds] or [0])
            if n.type == ASK_HELP:
                n.t_start = ready
                n.t_end = max(ready, given)
            elif n.type == RECEIVE:
                n.t_start = max(ready, given)
                n.t_end = n.t_start + 1
            else:
                n.t_start = ready
                n.t_end = n.t_start + 1

    # ------------------------------------------------------------ 4f: 5-minute phases
    SIZE_UNITS = {"short": 1, "medium": 2, "long": 4}     # coarse duration classes (~minutes)

    def n_phases(self) -> int:
        return max(1, -(-self.cfg.deadline_min // PHASE_MIN))

    def size_view(self) -> list[dict]:
        """What the size judge sees: every active step (except ASK_HELP, which is only waiting)."""
        out = []
        for nid in sorted((i for i in self.nodes if self.active(i)),
                          key=lambda i: (self.nodes[i].t_start, self.nodes[i].agent, i)):
            n = self.nodes[nid]
            if n.type == ASK_HELP:
                continue
            d = n.brief()
            if n.type == HELP:
                d["goes_to"] = n.location
            elif n.type == PASS and self.cfg.mover(n.agent, n.target) == n.agent:
                d["brings_it_to"] = self.cfg.agent(n.target).profile.room
            elif n.type == RECEIVE:
                giver = self.providers_of(n.id)
                if giver and self.cfg.mover(self.nodes[giver[0].src].agent, n.agent) == n.agent:
                    d["collects_it_from"] = self.cfg.agent(self.nodes[giver[0].src].agent).profile.room
            out.append(d)
        return out

    def _default_size(self, n: Node) -> str:
        if n.type == HELP:
            return "medium"
        if n.type == RECEIVE and self.providers_of(n.id) and \
                self.cfg.mover(self.nodes[self.providers_of(n.id)[0].src].agent, n.agent) == n.agent:
            return "medium"
        if n.type == PASS and self.cfg.mover(n.agent, n.target) == n.agent:
            return "medium"
        return "short"

    def assign_phases(self, sizes: dict | None) -> None:
        """Place steps in 5-minute phases, deterministically.
        `sizes` = {node id: short|medium|long} judged by the Graph LLM (None -> defaults by type).
        Each size is a coarse duration class (short=1, medium=2, long=4 units of ~1 min). Walking the
        graph in dependency order, a step starts when all its predecessors (own previous step, ORDER,
        the object / help it waits for) have ended; its phase is the 5-minute window it ends in.
        ASK_HELP is waiting: it ends when its helper ends and takes no time of its own."""
        g = self.nx()
        self.size_source = "graph_llm" if sizes else "rule"
        self.n_size_missing = 0
        end: dict[str, float] = {}
        for nid in nx.topological_sort(g):
            n = self.nodes[nid]
            start = max([end[p] for p in g.predecessors(nid)] or [0])
            if n.type == ASK_HELP:
                u = 0
            else:
                size = str((sizes or {}).get(nid, "")).lower().strip()
                if size not in self.SIZE_UNITS:
                    if sizes:
                        self.n_size_missing += 1
                    size = self._default_size(n)
                n.size = size
                u = self.SIZE_UNITS[size]
            end[nid] = start + u
            n.phase = max(1, -(-int(end[nid]) // PHASE_MIN))

    def makespan_min(self) -> int:
        ph = [n.phase for n in self.nodes.values() if n.status == "active" and n.phase]
        return max(ph or [0]) * PHASE_MIN

    def check_invariants(self) -> None:
        assert set(self.nodes) <= self._initial_nodes, "graph stage created a step"
        for agent, rid in self._agreements:
            ok = any(e.dst == rid and e.status == CONFIRMED and self.nodes[e.src].agent == agent
                     and self.nodes[e.src].status != "dropped" for e in self.collab.values())
            assert ok, f"CONFIRMED agreement ({agent} serves {rid}) was broken"

    def makespan(self) -> int:
        ends = [n.t_end for n in self.nodes.values() if n.status == "active" and n.t_end is not None]
        return max(ends or [0])

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return {"nodes": [asdict(n) for n in self.nodes.values()],
                "order": self.order,
                "edges": [asdict(e) for e in self.seq_edges() + list(self.collab.values())],
                "drops": self.drops, "unresolved": self.unresolved, "warnings": self.warnings,
                "ops_log": self.ops_log}


_OP_KEYS = ("op", "operation", "type", "action", "decision")
_OP_NAMES = {"connect", "disconnect", "move", "drop", "merge", "unresolved", "order", "independent"}


def normalize_op(op) -> dict:
    """LLMs sometimes write {"operation": "drop"}, {"type": "unresolved"}, {"drop": "r1_s2"} or
    {"issue": "I1", "decision": "not duplicate"}. Map these onto {"op": ...}."""
    if not isinstance(op, dict):
        return {"op": None}
    op = dict(op)
    kind = next((str(op[k]).lower().strip() for k in _OP_KEYS
                 if isinstance(op.get(k), str) and str(op[k]).lower().strip() in _OP_NAMES), None)
    if kind is None:
        hit = next((k for k in op if k.lower() in _OP_NAMES), None)
        if hit:
            kind, val = hit.lower(), op.pop(hit)
            if isinstance(val, str):
                op.setdefault("issue" if kind == "unresolved" else "node", val)
    if kind is None and "issue" in op:
        kind = "unresolved"
    op["op"] = kind
    if kind == "unresolved" and "why" not in op:
        op["why"] = str(op.get("decision") or op.get("reason") or "")
    return op


def _ids_in(obj) -> set[str]:
    out: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("id", "src", "dst", "node", "before", "after") and isinstance(v, str):
                out.add(v)
            out |= _ids_in(v)
    elif isinstance(obj, list):
        for v in obj:
            out |= _ids_in(v) if not isinstance(v, str) else ({v} if "_s" in v else set())
    return out


__all__ = ["PlanGraph"]
