"""
Stage 5 - JOINT PLAN RENDERING

Final Dependency Graph
        ↓
  Topological Levels
        ↓
    Joint Plan
        ↓
[Step 1]
- [R1] [LOCAL] ...
- [R2] [LOCAL] ...

[Step 2]
- [R2] [HELP] ...
...

No LLM is used in this stage.
The final Joint Plan is deterministically rendered from the Dependency Graph.
"""

from __future__ import annotations

from graph_reasoning import PlanGraph, RuleReport


# ============================================================================
# helpers
# ============================================================================

def _robot_label(agent: str) -> str:
    """Convert agent_1 -> R1."""
    if agent.startswith("agent_"):
        suffix = agent[len("agent_"):]
        if suffix.isdigit():
            return f"R{suffix}"
    return agent


def _partner(graph: PlanGraph, nid: str) -> str | None:
    """
    Return the partner robot for a collaboration step.

    handoff:
        request -> provider
    """

    node = graph.nodes[nid]

    # ASK_HELP / RECEIVE
    if node.type in {"ASK_HELP", "RECEIVE"}:
        provider_id = graph.handoff.get(nid)

        if provider_id and provider_id in graph.nodes:
            return graph.nodes[provider_id].agent

        return None

    # HELP / PASS
    if node.type in {"HELP", "PASS"}:
        request_id = next(
            (
                request_id
                for request_id, provider_id in graph.handoff.items()
                if provider_id == nid
            ),
            None,
        )

        if request_id and request_id in graph.nodes:
            return graph.nodes[request_id].agent

        return None

    return None


def _what(node) -> str:
    """Return the main object/action description."""

    if node.kind == "item" and node.item:
        return node.item

    return node.action


# ============================================================================
# natural-language rendering
# ============================================================================

def _line(graph: PlanGraph, nid: str) -> tuple[str, str | None]:
    """
    Render one graph node.

    Output format:
        [TYPE] natural-language action
    """

    node = graph.nodes[nid]

    partner = _partner(graph, nid)
    partner_label = _robot_label(partner) if partner else None

    what = _what(node)

    # ------------------------------------------------------------------
    # LOCAL
    # ------------------------------------------------------------------
    if node.type == "LOCAL":
        text = f"[LOCAL] {node.action}"

    # ------------------------------------------------------------------
    # ASK_HELP
    # ------------------------------------------------------------------
    elif node.type == "ASK_HELP":
        if partner_label:
            text = (
                f"[ASK_HELP] Ask {partner_label} to {node.action}."
            )
        else:
            text = (
                f"[ASK_HELP] Ask another robot to {node.action}."
            )

    # ------------------------------------------------------------------
    # HELP
    # ------------------------------------------------------------------
    elif node.type == "HELP":
        if partner_label:
            text = (
                f"[HELP] {node.action} for {partner_label}."
            )
        else:
            text = f"[HELP] {node.action}."

    # ------------------------------------------------------------------
    # RECEIVE
    # ------------------------------------------------------------------
    elif node.type == "RECEIVE":
        if partner_label:
            text = (
                f"[RECEIVE] Receive {what} from {partner_label}."
            )
        else:
            text = (
                f"[RECEIVE] Receive {what}."
            )

    # ------------------------------------------------------------------
    # PASS
    # ------------------------------------------------------------------
    elif node.type == "PASS":
        if partner_label:
            text = (
                f"[PASS] Pass {what} to {partner_label}."
            )
        else:
            text = (
                f"[PASS] Pass {what}."
            )

    # ------------------------------------------------------------------
    # Backward compatibility
    # ------------------------------------------------------------------
    elif node.type == "NEED":
        if partner_label:
            text = (
                f"[ASK_HELP] Ask {partner_label} to {node.action}."
            )
        else:
            text = (
                f"[ASK_HELP] Ask another robot to {node.action}."
            )

    else:
        text = f"[{node.type}] {node.action}"

    return text, partner


# ============================================================================
# step / level computation
# ============================================================================

def _steps(
    graph: PlanGraph,
    report: RuleReport,
) -> dict[int, list[str]]:
    """
    Group active graph nodes by dependency level.

    report.levels:
        node_id -> topological level

    Example:
        {
            "R1_S1": 0,
            "R2_S1": 0,
            "R2_S2": 1,
            "R2_S3": 2,
            "R1_S2": 3,
        }

    becomes:

        Step 1
            R1_S1
            R2_S1

        Step 2
            R2_S2

        Step 3
            R2_S3

        Step 4
            R1_S2
    """

    steps: dict[int, list[str]] = {}

    for nid in report.order:

        if nid not in graph.nodes:
            continue

        node = graph.nodes[nid]

        # Inactive HELP/PASS candidates are not part of
        # the final Joint Plan.
        if not node.active:
            continue

        level = report.levels.get(nid)

        if level is None:
            continue

        steps.setdefault(level, []).append(nid)

    return steps


# ============================================================================
# main renderer
# ============================================================================

def render_joint_plan(
    graph: PlanGraph,
    report: RuleReport,
) -> tuple[dict, str]:
    """
    Convert the final Dependency Graph into:

        1. Joint Plan JSON
        2. Natural-language Joint Plan

    The output follows the dependency order rather than grouping
    actions by robot.
    """

    step_groups = _steps(graph, report)

    # ------------------------------------------------------------------
    # JSON representation
    # ------------------------------------------------------------------

    step_json: list[dict] = []

    for step_index, (level, node_ids) in enumerate(
        sorted(step_groups.items()),
        start=1,
    ):

        actions = []

        for nid in node_ids:

            node = graph.nodes[nid]
            text, partner = _line(graph, nid)

            actions.append(
                {
                    "id": nid,
                    "agent": node.agent,
                    "label": _robot_label(node.agent),
                    "type": node.type,
                    "kind": node.kind,
                    "action": node.action,
                    "item": node.item,
                    "partner": partner,
                    "text": text,
                }
            )

        step_json.append(
            {
                "step": step_index,
                "level": level,
                "actions": actions,
            }
        )

    # ------------------------------------------------------------------
    # unresolved requests
    # ------------------------------------------------------------------

    unresolved_requests = []

    for nid in report.unresolved_needs:

        if nid not in graph.nodes:
            continue

        node = graph.nodes[nid]

        unresolved_requests.append(
            {
                "id": nid,
                "agent": node.agent,
                "label": _robot_label(node.agent),
                "type": node.type,
                "kind": node.kind,
                "action": node.action,
                "item": node.item,
            }
        )

    # ------------------------------------------------------------------
    # final JSON
    # ------------------------------------------------------------------

    joint = {
        "format": "dependency_ordered",
        "steps": step_json,
        "unresolved_requests": unresolved_requests,
        "warnings": list(report.warnings),
    }

    # ------------------------------------------------------------------
    # natural-language Joint Plan
    # ------------------------------------------------------------------

    lines = [
        "### Joint Plan",
        "",
    ]

    for step in step_json:

        lines.append(f"[Step {step['step']}]")

        for action in step["actions"]:

            lines.append(
                f"- [{action['label']}] {action['text']}"
            )

        lines.append("")

    # ------------------------------------------------------------------
    # unresolved requests
    # ------------------------------------------------------------------

    if unresolved_requests:

        lines.append("### Unresolved Requests")

        for request in unresolved_requests:

            what = (
                request["item"]
                if request["item"]
                else request["action"]
            )

            lines.append(
                f"- [{request['label']}] "
                f"[{request['type']}] {what}"
            )

        lines.append("")

    # ------------------------------------------------------------------
    # warnings
    # ------------------------------------------------------------------

    if joint["warnings"]:

        lines.append("### Warnings")

        for warning in joint["warnings"]:
            lines.append(f"- {warning}")

        lines.append("")

    return joint, "\n".join(lines).rstrip()
