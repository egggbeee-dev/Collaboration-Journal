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
        body = f"pass {n.item} to {n.target}" + (" (offered)" if n.origin == "offer" else "")
    else:
        body = n.action
        serves = [g.nodes[e.dst].agent for e in g.collab_edges() if e.src == nid]
        if serves:
            body += f" (also serves {', '.join(serves)}'s request)"
        if n.travel:
            body += f" [+{n.travel}m back to {n.location}]"
    if n.type == RECEIVE and prov:
        body += f" [door handoff {g.cfg.handoff_min}m]"
    return f"- {n.agent} [{n.type}] {body}  ({nid})  → t={n.t_end}"


def render(g: PlanGraph) -> str:
    act = sorted((nid for nid, n in g.nodes.items() if n.status == "active"),
                 key=lambda i: (g.nodes[i].t_start, g.nodes[i].agent, i))
    ms = g.makespan()
    ok = "OK" if ms <= g.cfg.deadline_min else "MISSED"
    lines = [f"### Joint Plan — {g.cfg.task_id}  (makespan {ms} min / deadline {g.cfg.deadline_min} min: {ok})", ""]
    last = None
    for i in act:
        if g.nodes[i].t_start != last:
            last = g.nodes[i].t_start
            lines += ([""] if len(lines) > 2 else []) + [f"[t={last}]"]
        lines.append(_line(g, i))
    blocked = [n for n in g.nodes.values() if n.status == "blocked"]
    skipped = [n for n in g.nodes.values() if n.status == "skipped"]
    if g.unresolved:
        lines += ["", "### Unresolved"]
        lines += [f"- {u}" for u in g.unresolved]
    if blocked:
        lines += ["", "### Blocked (cannot run; the robot skips them and continues)"]
        lines += [f"- {n.agent} [{n.type}] {n.text()} ({n.id}) — {n.violations[-1] if n.violations else ''}"
                  for n in blocked]
    if skipped:
        lines += ["", "### Skipped (no longer needed)"]
        lines += [f"- {n.agent} [{n.type}] {n.text()} ({n.id})" for n in skipped]
    if g.warnings:
        lines += ["", "### Warnings"]
        lines += [f"- {w}" for w in g.warnings]
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
        "capability_violations_after_fix": sum(1 for n in nodes if n.type not in REQUEST_TYPES and
                                               any(not v.startswith("blocked (") for v in n.violations)),
        "n_requests_withdrawn": sum(1 for d in g.drops if d["by"] == "stage2-check"),
        "n_served_by_own_existing_step": sum(1 for e in final_collab if g.nodes[e.src].type == "LOCAL"),
        "n_duplicates_merged": g.n_merged,
        "n_duplicates_dropped": g.n_dup_dropped,
        "n_redundant_branches_released": g.n_released,
        "n_requests_moved_later": g.n_tightened,
        "n_warnings": len(g.warnings),
        "plan_fix_rounds": {a: m["fix_rounds"] for a, m in plan_meta.items()},
        "graph_ops_applied": sum(o["applied"] for o in g.ops_log),
        "graph_ops_rejected": sum(not o["applied"] for o in g.ops_log),
        "makespan": g.makespan(),
        "deadline": g.cfg.deadline_min,
        "deadline_ok": g.makespan() <= g.cfg.deadline_min,
        "success_no_blocked": all(n.status != "blocked" for n in nodes),
        "n_requests_declined": sum(v == "declined" for v in g.request_outcome.values()),
        "n_requests_failed": sum(v == "failed" for v in g.request_outcome.values()),
        "n_requests_moot": sum(v == "moot" for v in g.request_outcome.values()),
        "n_blocked_by_decline": sum(1 for n in nodes if n.status == "blocked" and "blocked (declined)" in n.violations),
        "n_blocked_by_failure": sum(1 for n in nodes if n.status == "blocked" and "blocked (failed)" in n.violations),
        "success_no_failure": not any(n.status == "blocked" and "blocked (failed)" in n.violations for n in nodes),
        "n_handoff_prep_dropped": g.n_prep_dropped,
        "n_offers_made": sum(1 for n in nodes if n.type == "PASS" and n.origin == "offer"),
        "n_offers_taken": sum(1 for e in final_collab if g.nodes[e.src].origin == "offer"),
        "n_offers_untaken": g.n_offers_untaken,
        "n_preparations_released": g.n_prep_released,
        **llm_usage,
    }
