"""Stage 4 - Graph Reasoning (central, structure-only).

4a build graph            (code)
4b rule layer + issues    (code)
4c semantic choice        (LLM, only on issues, only among code-given candidates)
4d validate + apply ops   (code), re-check, at most `max_rounds`
4e finalize + schedule    (code): block unresolvable parts, compute the logical order t_start / t_end
4f 5-minute phases        (LLM proposes a phase per step, code enforces the dependencies;
                           rule fallback without an LLM)
"""
from __future__ import annotations

import json

from .graph import PlanGraph, normalize_op
from .llm import BaseLLM
from .log import EventLog
from .prompts import GRAPH_SYSTEM, GRAPH_USER, PHASE_SYSTEM, PHASE_USER, _j
from .schemas import Edge, Node, Offer, TaskConfig


async def graph_reasoning(cfg: TaskConfig, plans: dict[str, list[Node]], collab: list[Edge],
                          offers: dict[str, Offer], judgments: list[dict], llm: BaseLLM | None,
                          log: EventLog, max_rounds: int = 2) -> PlanGraph:
    g = PlanGraph(cfg, plans, collab, offers, judgments)
    log.log("graph", "-", "built", nodes=len(g.nodes), collab_edges=len(g.collab))

    for rnd in range(1, max_rounds + 1):
        g.apply_rules()
        issues = g.detect_issues()
        log.log("graph", "-", f"round{rnd}_issues", n=len(issues),
                kinds=",".join(sorted({i["issue"] for i in issues})) or "none")
        if not issues or llm is None:
            break
        d = await llm.complete(GRAPH_SYSTEM, GRAPH_USER.format(task=cfg.task, issues=_j(issues)),
                               key=f"graph:{rnd}")
        for op in d.get("ops", []) or []:
            err = g.apply_op(op, issues)
            g.ops_log.append({"round": rnd, "op": op, "applied": err is None, "error": err})
            extra = {"error": err, "raw": json.dumps(op, ensure_ascii=False)[:160]} if err else {}
            log.log("graph", "llm", f"op_{normalize_op(op).get('op')}", ok=err is None, **extra)

    g.apply_rules()
    g.finalize()
    g.check_invariants()
    log.log("graph", "-", "scheduled", makespan=g.makespan(), unresolved=len(g.unresolved))

    proposed = None
    if llm is not None:
        robots = {a.id: {"room": a.profile.room, "mobile": a.profile.mobile, "body": a.profile.embodiment}
                  for a in cfg.agents}
        try:
            d = await llm.complete(PHASE_SYSTEM.format(n_phases=g.n_phases()),
                                   PHASE_USER.format(task=cfg.task, robots=_j(robots), steps=_j(g.phase_view())),
                                   key="graph:phase")
            proposed = d.get("phases") if isinstance(d.get("phases"), dict) else None
        except Exception as e:  # noqa: BLE001  - phases are optional: fall back to the rule
            log.log("graph", "llm", "phase_failed", error=str(e)[:120])
    g.assign_phases(proposed)
    log.log("graph", "-", "phased", source=g.phase_source, makespan_min=g.makespan_min(),
            deadline=cfg.deadline_min, fixed=g.n_phase_fixed, missing=g.n_phase_missing)
    return g
