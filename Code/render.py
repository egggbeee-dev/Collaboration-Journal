"""Rule-based (template) conversion of the final graph into a natural-language Joint Plan.
No LLM: the same input always gives the same text, and every compared method can use the
same renderer, so wording never changes the score.

Steps are grouped into slots (levels of the dependency graph); steps in the same slot can
run in parallel.
"""
from __future__ import annotations

from graph_reasoning import PlanGraph, RuleReport


def _line(graph: PlanGraph, nid: str) -> tuple[str, str | None]:
    n = graph.nodes[nid]
    provider = graph.handoff.get(nid) if n.type == "NEED" else None
    receiver = next((need for need, p in graph.handoff.items() if p == nid), None) if n.type == "PASS" else None
    partner_step = provider or receiver
    partner = graph.nodes[partner_step].agent if partner_step else None
    what = n.item if (n.kind == "item" and n.item) else n.action

    if n.type == "LOCAL":
        text = f"{n.agent}: {n.action}"
    elif n.type == "PASS" and n.kind == "item":
        text = f"{n.agent} hands over {what} to {partner}"
    elif n.type == "PASS":  # task volunteer
        text = f"{n.agent} does for {partner}: {n.action}"
    elif n.type == "NEED" and n.kind == "item":
        text = f"{n.agent} receives {what}" + (f" from {partner}" if partner else " (NO PROVIDER - unresolved)")
    else:  # NEED task
        text = f"{n.agent} waits until this is done" + (f" by {partner}" if partner else " (NO PROVIDER - unresolved)") + f": {n.action}"
    return text, partner


def _slots(graph: PlanGraph, report: RuleReport) -> dict[str, int]:
    """Start from the earliest possible slot (graph level), then move item hand-overs so that
    'X hands over item' sits in the same slot as 'Y receives item' (a hand-over is one joint
    event). A step is moved only if everything after it still comes later."""
    slot = dict(report.levels)
    succ: dict[str, list[str]] = {}
    for e in graph.edges():
        succ.setdefault(e.src, []).append(e.dst)
    for nid in reversed(report.order):  # successors are settled before their predecessors
        n = graph.nodes[nid]
        if n.type != "PASS" or n.kind != "item":
            continue
        need = next((k for k, p in graph.handoff.items() if p == nid), None)
        if need is None or slot[need] <= slot[nid]:
            continue
        target = slot[need]
        if all(slot[v] > target for v in succ.get(nid, []) if v != need):
            slot[nid] = target
    return slot


def render_joint_plan(graph: PlanGraph, report: RuleReport) -> tuple[dict, str]:
    slot_of = _slots(graph, report)
    slots: dict[int, list[dict]] = {}
    for nid in report.order:
        text, partner = _line(graph, nid)
        n = graph.nodes[nid]
        slots.setdefault(slot_of[nid], []).append(
            {"id": nid, "agent": n.agent, "type": n.type, "kind": n.kind, "partner": partner, "text": text}
        )
    joint = {
        "slots": [{"slot": i + 1, "steps": steps} for i, (_, steps) in enumerate(sorted(slots.items()))],
        "unresolved_needs": [{"id": nid, "agent": graph.nodes[nid].agent, "action": graph.nodes[nid].action} for nid in report.unresolved_needs],
        "warnings": report.warnings,
    }
    lines = [f"Joint plan ({len(joint['slots'])} slots; steps in the same slot can run in parallel)"]
    for s in joint["slots"]:
        lines.append(f"[Slot {s['slot']}]")
        lines += [f"  - {st['text']}" for st in s["steps"]]
    if joint["unresolved_needs"]:
        lines.append("Unresolved needs (nobody provides them):")
        lines += [f"  - {u['agent']}: {u['action']}" for u in joint["unresolved_needs"]]
    if joint["warnings"]:
        lines.append("Warnings:")
        lines += [f"  - {w}" for w in joint["warnings"]]
    return joint, "\n".join(lines)
