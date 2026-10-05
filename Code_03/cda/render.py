"""Rule-based (template) rendering of the Joint Plan, plus metrics. No LLM here."""
from __future__ import annotations

from .graph import PlanGraph
from .schemas import ASK_HELP, CONFIRMED, HELP, PASS, RECEIVE, REQUEST_TYPES


def _line(g: PlanGraph, nid: str) -> str:
    n = g.nodes[nid]
    prov = g.providers_of(nid)
    partner = g.nodes[prov[0].src].agent if prov else None
    if n.type == ASK_HELP:
        body = f"wait for {partner} to: {n.action}"
    elif n.type == RECEIVE:
        body = f"receive {n.item} from {partner}"
    elif n.type == HELP:
        body = f"{n.action} (help {n.target})" + (f" [+{n.travel}m travel to {n.location}]" if n.travel else "")
    elif n.type == PASS:
        body = f"pass {n.item} to {n.target}"
    else:
        body = n.action
    return f"[t={n.t_start:>2}–{n.t_end:<2}] {n.agent} [{n.type}] {body}  ({nid})"


def render(g: PlanGraph) -> str:
    act = sorted((nid for nid, n in g.nodes.items() if n.status == "active"),
                 key=lambda i: (g.nodes[i].t_start, g.nodes[i].agent, i))
    ms = g.makespan()
    ok = "OK" if ms <= g.cfg.deadline_min else "MISSED"
    lines = [f"### Joint Plan — {g.cfg.task_id}  (makespan {ms} min / deadline {g.cfg.deadline_min} min: {ok})", ""]
    lines += [_line(g, i) for i in act]
    blocked = [n for n in g.nodes.values() if n.status == "blocked"]
    if g.unresolved:
        lines += ["", "### Unresolved"]
        lines += [f"- {u}" for u in g.unresolved]
    if blocked:
        lines += ["", "### Blocked (cannot run)"]
        lines += [f"- {n.agent} [{n.type}] {n.text()} ({n.id})" for n in blocked]
    if g.drops:
        lines += ["", "### Dropped"]
        lines += [f"- {d['agent']} [{d['type']}] {d['text']} — {d['reason']} (by {d['by']})" for d in g.drops]
    return "\n".join(lines)


def metrics(g: PlanGraph, plan_meta: dict, llm_usage: dict) -> dict:
    nodes = list(g.nodes.values())
    active = [n for n in nodes if n.status == "active"]
    final_collab = g.collab_edges()
    requests = [n for n in nodes if n.type in REQUEST_TYPES]
    n_conf = sum(e.status == CONFIRMED for e in final_collab)
    n_graph = len(g.selected_by_graph)
    return {
        "task_id": g.cfg.task_id,
        "n_steps_total": len(nodes),
        "n_steps_active": len(active),
        "n_blocked": sum(n.status == "blocked" for n in nodes),
        "n_dropped": len(g.drops),
        "n_requests": len(requests),
        "n_requests_served": sum(1 for r in requests if r.status == "active" and g.providers_of(r.id)),
        "n_collab_edges_final": len(final_collab),
        "confirmed_ratio": round(n_conf / len(final_collab), 3) if final_collab else None,
        "selected_by_graph": n_graph,
        "selected_by_rule_single_volunteer": len(g.selected_by_rule & {e.dst for e in final_collab}),
        "capability_violations_after_fix": sum(bool(n.violations) for n in nodes),
        "plan_fix_rounds": {a: m["fix_rounds"] for a, m in plan_meta.items()},
        "graph_ops_applied": sum(o["applied"] for o in g.ops_log),
        "graph_ops_rejected": sum(not o["applied"] for o in g.ops_log),
        "makespan": g.makespan(),
        "deadline": g.cfg.deadline_min,
        "deadline_ok": g.makespan() <= g.cfg.deadline_min,
        "success_no_blocked": all(n.status != "blocked" for n in nodes),
        **llm_usage,
    }
