"""Stage 4 - JOINT PLAN RENDERING (template, no LLM).

Active steps are grouped by dependency level: steps in the same [Step k] can run in parallel.
"""

from __future__ import annotations

from graph_reasoning import PlanGraph, RuleReport
from schemas import robot_label


def _partner(graph: PlanGraph, nid: str) -> str | None:
    n = graph.nodes[nid]
    if n.type in {"ASK_HELP", "RECEIVE"}:
        p = graph.provider_of(nid)
        return graph.nodes[p].agent if p else None
    if n.type in {"HELP", "PASS"}:
        r = graph.request_of(nid)
        return graph.nodes[r].agent if r else None
    return None


def _text(graph: PlanGraph, nid: str) -> str:
    n = graph.nodes[nid]
    partner = _partner(graph, nid)
    who = robot_label(partner) if partner else None
    tag = " ⟨added by graph⟩" if n.origin == "graph" else ""

    if n.type == "LOCAL":
        return f"[LOCAL] {n.action}"
    if n.type == "ASK_HELP":
        return (f"[ASK_HELP] Wait for {who} to: {n.action}" if who
                else f"[ASK_HELP] Needs another robot to: {n.action}")
    if n.type == "HELP":
        return f"[HELP] {n.action}" + (f" (for {who})" if who else "")
    if n.type == "RECEIVE":
        if not who:
            return f"[RECEIVE] Needs: {n.item}"
        given = graph.nodes[graph.provider_of(nid)].item
        gives = f" [{who} gives: {given}]" if given and given.lower() != (n.item or "").lower() else ""
        return f"[RECEIVE] Receive {n.item} from {who}{gives}{tag}"
    if n.type == "PASS":
        return f"[PASS] Pass {n.item}" + (f" to {who}" if who else "") + tag
    return f"[{n.type}] {n.action}"


def render_joint_plan(graph: PlanGraph, report: RuleReport) -> tuple[dict, str]:
    by_level: dict[int, list[str]] = {}
    for nid in report.order:
        if graph.nodes[nid].active:
            by_level.setdefault(report.levels[nid], []).append(nid)

    steps = []
    for k, (lvl, ids) in enumerate(sorted(by_level.items()), start=1):
        steps.append({
            "step": k,
            "actions": [{
                "id": nid,
                "agent": graph.nodes[nid].agent,
                "label": robot_label(graph.nodes[nid].agent),
                "type": graph.nodes[nid].type,
                "origin": graph.nodes[nid].origin,
                "partner": _partner(graph, nid),
                "text": _text(graph, nid),
            } for nid in ids],
        })

    def _brief(nid: str) -> dict:
        n = graph.nodes[nid]
        return {"id": nid, "label": robot_label(n.agent), "type": n.type, "action": n.action, "item": n.item}

    warnings = list(report.warnings)
    if not report.ok:
        warnings.append("No valid schedule: the dependency graph still contains a cycle.")

    joint = {
        "ok": report.ok,
        "steps": steps,
        "unresolved_requests": [_brief(n) for n in report.unresolved],
        "unused_offers": [_brief(n) for n in report.idle_providers],
        "dropped_steps": graph.drops,
        "coverage_gaps": graph.gaps,
        "warnings": warnings,
    }

    lines = ["### Joint Plan", ""]
    for s in steps:
        lines.append(f"[Step {s['step']}]")
        lines += [f"- [{a['label']}] {a['text']}" for a in s["actions"]]
        lines.append("")

    def section(title: str, rows: list[str]) -> None:
        if rows:
            lines.append(f"### {title}")
            lines.extend(f"- {r}" for r in rows)
            lines.append("")

    section("Unresolved Requests",
            [f"[{r['label']}] [{r['type']}] {r['item'] or r['action']}" for r in joint["unresolved_requests"]])
    section("Unused Offers (volunteered, not needed)",
            [f"[{r['label']}] [{r['type']}] {r['item'] or r['action']}" for r in joint["unused_offers"]])
    section("Dropped Steps", [f"[{robot_label(d['agent'])}] {d['action']} — {d['reason']}" for d in graph.drops])
    section("Coverage Gaps", graph.gaps)
    section("Warnings", warnings)

    return joint, "\n".join(lines).rstrip()
