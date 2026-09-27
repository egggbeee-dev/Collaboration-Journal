"""Rule-based conversion of the final graph into a human-readable Joint Plan.

The dependency graph and slot information are still retained in the JSON output for
analysis. The human-readable text is rendered robot-centrically: each robot gets its
own ordered list of actions, while hand-off steps explicitly name the partner robot.

No LLM is used here, so every compared method receives the same deterministic renderer.
"""
from __future__ import annotations

from graph_reasoning import PlanGraph, RuleReport


def _robot_label(agent: str) -> str:
    """Convert internal ids such as ``agent_1`` to compact labels such as ``R1``."""
    if agent.startswith("agent_"):
        suffix = agent[len("agent_"):]
        if suffix.isdigit():
            return f"R{suffix}"
    return agent


def _partner(graph: PlanGraph, nid: str) -> str | None:
    n = graph.nodes[nid]
    if n.type == "NEED":
        provider = graph.handoff.get(nid)
        return graph.nodes[provider].agent if provider else None
    if n.type == "PASS":
        need = next((need_id for need_id, pass_id in graph.handoff.items() if pass_id == nid), None)
        return graph.nodes[need].agent if need else None
    return None


def _line(graph: PlanGraph, nid: str) -> tuple[str, str | None]:
    """Render one graph node without repeating the robot id in every line."""
    n = graph.nodes[nid]
    partner = _partner(graph, nid)
    partner_label = _robot_label(partner) if partner else None
    what = n.item if (n.kind == "item" and n.item) else n.action

    if n.type == "LOCAL":
        text = n.action
    elif n.type == "PASS" and n.kind == "item":
        text = f"Hand {what} to {partner_label}" if partner_label else f"Hand {what} over"
    elif n.type == "PASS":  # task volunteer
        text = f"{n.action} for {partner_label}" if partner_label else n.action
    elif n.type == "NEED" and n.kind == "item":
        text = f"Receive {what} from {partner_label}" if partner_label else f"Receive {what} (unresolved)"
    else:  # NEED task
        if partner_label:
            text = f"Wait for {partner_label} to complete: {n.action}"
        else:
            text = f"Wait for this task to be completed: {n.action} (unresolved)"
    return text, partner


def _slots(graph: PlanGraph, report: RuleReport) -> dict[str, int]:
    """Compute graph levels while keeping hand-off item transfer/receipt in one slot."""
    slot = dict(report.levels)
    succ: dict[str, list[str]] = {}
    for e in graph.edges():
        succ.setdefault(e.src, []).append(e.dst)
    for nid in reversed(report.order):
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
    """Return both analysis-friendly JSON and a robot-centric natural-language Joint Plan."""
    slot_of = _slots(graph, report)

    # Keep the original slot representation for evaluation/debugging.
    slots: dict[int, list[dict]] = {}
    for nid in report.order:
        text, partner = _line(graph, nid)
        n = graph.nodes[nid]
        slots.setdefault(slot_of[nid], []).append(
            {"id": nid, "agent": n.agent, "type": n.type, "kind": n.kind, "partner": partner, "text": text}
        )

    # Robot-centric representation: preserve each robot's local execution order.
    robot_steps: dict[str, list[dict]] = {}
    robot_order: list[str] = []
    for nid, n in graph.nodes.items():
        if n.agent not in robot_steps:
            robot_steps[n.agent] = []
            robot_order.append(n.agent)
        if n.active:
            text, partner = _line(graph, nid)
            robot_steps[n.agent].append(
                {
                    "id": nid,
                    "order": n.order,
                    "type": n.type,
                    "kind": n.kind,
                    "partner": partner,
                    "text": text,
                }
            )

    for agent in robot_order:
        robot_steps[agent].sort(key=lambda x: (x["order"], x["id"]))

    joint = {
        "format": "robot_centric",
        "robots": [
            {
                "agent": agent,
                "label": _robot_label(agent),
                "steps": steps,
            }
            for agent, steps in ((agent, robot_steps[agent]) for agent in robot_order)
        ],
        "slots": [{"slot": i + 1, "steps": steps} for i, (_, steps) in enumerate(sorted(slots.items()))],
        "unresolved_needs": [
            {
                "id": nid,
                "agent": graph.nodes[nid].agent,
                "action": graph.nodes[nid].action,
            }
            for nid in report.unresolved_needs
        ],
        "warnings": report.warnings,
    }

    lines = ["### Joint Plan", ""]
    for robot in joint["robots"]:
        lines.append(f"**{robot['label']}**")
        if not robot["steps"]:
            lines.append("1. Wait.")
        else:
            for i, step in enumerate(robot["steps"], start=1):
                lines.append(f"{i}. {step['text']}")
        lines.append("")

    if joint["unresolved_needs"]:
        lines.append("**Unresolved Needs**")
        for u in joint["unresolved_needs"]:
            lines.append(f"- {_robot_label(u['agent'])}: {u['action']}")
        lines.append("")

    if joint["warnings"]:
        lines.append("**Warnings**")
        lines.extend(f"- {w}" for w in joint["warnings"])

    return joint, "\n".join(lines).rstrip()
