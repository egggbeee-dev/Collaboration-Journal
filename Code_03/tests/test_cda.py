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
                         Offer, Profile, TaskConfig)
from cda.stage2_local_plan import validate  # noqa: E402

TASK = ROOT / "examples/task_bath/task.json"
MOCK = ROOT / "examples/task_bath/mock_script.json"


def _run(**kw):
    return asyncio.run(run_pipeline(TaskConfig.load(TASK), MockLLM(MOCK), verbose=False, **kw))


def test_full_pipeline():
    r = _run()
    m = r["metrics"]
    assert m["n_requests_served"] == 5 and m["n_blocked"] == 0
    assert m["selected_by_graph"] == 1 and m["deadline_ok"]
    assert m["n_duplicates_merged"] == 1                    # R3 accepted work it already did
    assert m["plan_fix_rounds"]["R1"] == 1                  # R1 dropped its outsourcing request
    g = r["_graph_obj"]
    assert g.nodes["r3_s2"].travel == 2                     # R3 walks to the kitchen for the thermometer
    assert g.nodes["r3_s3"].travel == 2                     # ...and from there to the bathroom (not twice)
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


# ---------------------------------------------------------------- regression: movie run (2026-10-05)
# Observed with gpt-4o: every robot planned the whole task and "outsourced" the parts it could not do,
# targets accepted work they already did -> each sub-task executed three times, makespan 24 > 20,
# while all self-reported metrics looked perfect.

def _movie_cfg():
    return TaskConfig("movie", "prepare movie night", 20, [
        AgentInput("R1", Profile("kitchen", False, 2, "fixed arm")),
        AgentInput("R2", Profile("living room", True, 20, "heavy arm")),
        AgentInput("R3", Profile("bedroom", True, 3, "light arm")),
        AgentInput("R4", Profile("bathroom", True, 3, "light arm"))])


def test_movie_outsourcing_requests_are_rejected_by_stage2_check():
    cfg = _movie_cfg()
    offers = {a: Offer(a, "x", ["move", "carry light objects"], [], []) for a in cfg.ids}
    observed_r3 = [  # R3's observed plan: no own step depends on its two requests
        {"type": "LOCAL", "action": "Move to the living room", "uses": 0, "location": "living room", "duration": 3},
        {"type": "ASK_HELP", "action": "Rearrange the seating to face the TV", "target": "R2", "duration": 5},
        {"type": "ASK_HELP", "action": "Pick up snacks and drinks", "target": "R1", "duration": 3}]
    steps, errors = validate(observed_r3, "R3", offers, cfg)
    reqs = [s for s in steps if s["type"] == "ASK_HELP"]
    assert all(s["enables"] is None and s["violations"] for s in reqs)
    assert any("enables" in e for e in errors)


def test_movie_duplicate_work_is_merged_and_agreements_kept():
    cfg = _movie_cfg()
    seat = "Rearrange the seating to face the TV"
    plans = {
        "R2": [Node("r2_s1", "R2", "LOCAL", action=seat, uses=0, location="living room", duration=5),
               Node("r2_s4", "R2", "HELP", action=seat, target="R3", answers="r3_s2", location="living room", duration=5),
               Node("r2_s5", "R2", "HELP", action=seat, target="R4", answers="r4_s2", location="living room", duration=5)],
        "R3": [Node("r3_s1", "R3", "LOCAL", action="Move to the living room", uses=0, location="living room", duration=3),
               Node("r3_s2", "R3", "ASK_HELP", action=seat, target="R2", enables="r3_s3", location="living room"),
               Node("r3_s3", "R3", "LOCAL", action="Put cushions on the sofa", uses=0, location="living room")],
        "R4": [Node("r4_s2", "R4", "ASK_HELP", action=seat, target="R2", enables="r4_s3", location="living room"),
               Node("r4_s3", "R4", "LOCAL", action="Place drinks on the coffee table", uses=0, location="living room")],
        "R1": [Node("r1_s1", "R1", "LOCAL", action="Pick up snacks", uses=0, location="kitchen")]}
    edges = [Edge("r2_s4", "r3_s2", HELP_EDGE, CONFIRMED), Edge("r2_s5", "r4_s2", HELP_EDGE, CONFIRMED)]
    g = PlanGraph(cfg, plans, edges)
    dups = [i for i in g.detect_issues() if i["issue"] == "DUPLICATE_WORK"]
    assert len(dups) == 3                                     # s1~s4, s1~s5, s4~s5
    assert g.apply_op({"op": "drop", "node": "r2_s4"}, dups) is not None   # CONFIRMED -> cannot drop
    assert g.apply_op({"op": "merge", "node": "r2_s4", "into": "r2_s1"}, dups) is None
    assert g.apply_op({"op": "merge", "node": "r2_s5", "into": "r2_s1"}, dups) is None
    assert not [i for i in g.detect_issues() if i["issue"] == "DUPLICATE_WORK"]
    g.finalize()
    g.check_invariants()                                      # both agreements with R2 still hold
    assert g.makespan() == 7                                  # seating once (0-5), cushions/drinks after


def test_merge_across_robots_is_refused():
    cfg = _movie_cfg()
    plans = {"R1": [], "R2": [Node("r2_s1", "R2", "LOCAL", action="put pillow on sofa", uses=0)],
             "R3": [Node("r3_s1", "R3", "LOCAL", action="put pillow on sofa", uses=0)], "R4": []}
    g = PlanGraph(cfg, plans, [])
    issues = g.detect_issues()
    assert g.apply_op({"op": "merge", "node": "r3_s1", "into": "r2_s1"}, issues) is not None
    assert g.apply_op({"op": "drop", "node": "r3_s1"}, issues) is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
