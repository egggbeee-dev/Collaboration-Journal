"""Run: python -m pytest -q   (or: python tests/test_cda.py)"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cda.graph import PlanGraph  # noqa: E402
from cda.llm import MockLLM  # noqa: E402
from cda.pipeline import run_pipeline  # noqa: E402
from cda.schemas import (CONFIRMED, HELP_EDGE, PROPOSED, TRANSFER, AgentInput, Edge, Node,  # noqa: E402
                         Profile, TaskConfig)

TASK = ROOT / "examples/task_bath/task.json"
MOCK = ROOT / "examples/task_bath/mock_script.json"


def _run(**kw):
    return asyncio.run(run_pipeline(TaskConfig.load(TASK), MockLLM(MOCK), verbose=False, **kw))


def test_full_pipeline():
    r = _run()
    m = r["metrics"]
    assert m["n_requests_served"] == 4 and m["n_blocked"] == 0
    assert m["selected_by_graph"] == 1 and m["deadline_ok"]
    assert "t= 0" in r["joint_plan"]


def test_rules_only_ablation_leaves_ambiguity_unresolved():
    r = _run(use_graph_llm=False)
    g = r["_graph_obj"]
    assert g.nodes["r4_s2"].status == "blocked"            # two volunteers, nobody chose
    assert r["metrics"]["selected_by_graph"] == 0


def test_no_proposal_ablation_blocks_requests():
    r = _run(use_edge_proposal=False)
    assert r["metrics"]["n_requests_served"] == 0


def _tiny_cfg():
    p = Profile("room", True, 5, "x")
    return TaskConfig("t", "t", 30, [AgentInput("R1", p), AgentInput("R2", p)])


def test_confirmed_edge_cannot_be_removed():
    cfg = _tiny_cfg()
    plans = {"R1": [Node("r1_s1", "R1", "RECEIVE", item="cup", target="R2")],
             "R2": [Node("r2_s1", "R2", "PASS", item="cup", target="R1", answers="r1_s1")]}
    g = PlanGraph(cfg, plans, [Edge("r2_s1", "r1_s1", TRANSFER, CONFIRMED, "stage3")])
    issues = [{"id": "I1", "issue": "X", "edges": [{"src": "r2_s1", "dst": "r1_s1"}]}]
    assert g.apply_op({"op": "disconnect", "src": "r2_s1", "dst": "r1_s1"}, issues) is not None
    assert g.apply_op({"op": "drop", "node": "r2_s1"}, issues) is not None
    g.check_invariants()


def test_cycle_detected_and_fixed_by_move():
    # R1: ASK_HELP(r1_s1) -> HELP for R2 (r1_s2);  R2: ASK_HELP(r2_s1) -> HELP for R1 (r2_s2)
    # each robot helps only after its own request is served -> deadlock
    cfg = _tiny_cfg()
    plans = {"R1": [Node("r1_s1", "R1", "ASK_HELP", action="a", target="R2"),
                    Node("r1_s2", "R1", "HELP", action="b", target="R2", answers="r2_s1")],
             "R2": [Node("r2_s1", "R2", "ASK_HELP", action="b", target="R1"),
                    Node("r2_s2", "R2", "HELP", action="a", target="R1", answers="r1_s1")]}
    edges = [Edge("r2_s2", "r1_s1", HELP_EDGE, CONFIRMED), Edge("r1_s2", "r2_s1", HELP_EDGE, PROPOSED)]
    g = PlanGraph(cfg, plans, edges)
    issues = g.detect_issues()
    assert any(i["issue"] == "CYCLE" for i in issues)
    assert g.apply_op({"op": "move", "node": "r1_s2", "after": "START"}, issues) is None
    assert not any(i["issue"] == "CYCLE" for i in g.detect_issues())
    g.finalize()
    g.check_invariants()
    assert all(n.status == "active" for n in g.nodes.values())


def test_unanswered_request_blocks_descendants():
    cfg = _tiny_cfg()
    plans = {"R1": [Node("r1_s1", "R1", "RECEIVE", item="cup", target="R2"),
                    Node("r1_s2", "R1", "LOCAL", action="drink", uses=0)],
             "R2": [Node("r2_s1", "R2", "LOCAL", action="idle", uses=0)]}
    g = PlanGraph(cfg, plans, [])
    g.finalize()
    assert g.nodes["r1_s1"].status == "blocked" and g.nodes["r1_s2"].status == "blocked"
    assert g.nodes["r2_s1"].t_end == 2


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
