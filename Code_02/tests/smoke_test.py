"""Offline tests (scripted LLM, no API key).

    python tests/smoke_test.py        # or: python -m pytest tests/
"""
import asyncio
import copy
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from demo_script import DEMO_SCRIPT                                        # noqa: E402
from graph_reasoning import GraphOp, PlanGraph, apply_op, rule_verify      # noqa: E402
from llm import ScriptedClient                                             # noqa: E402
from pipeline import run_pipeline                                          # noqa: E402
from schemas import LocalPlan, Offer, RawLocalPlan, RawStep, TaskConfig    # noqa: E402

CFG = TaskConfig.model_validate_json((ROOT / "examples/home_training.json").read_text())


def _run(script, **kw):
    return asyncio.run(run_pipeline(CFG, ScriptedClient(copy.deepcopy(script)), verbose=False, **kw))


def _graph(plans: dict[str, list[dict]], items: dict[str, list[str]] | None = None) -> PlanGraph:
    known = set(plans)
    lp = {a: LocalPlan.from_raw(a, RawLocalPlan(steps=[RawStep(**s) for s in st]), known) for a, st in plans.items()}
    offers = {a: Offer(agent=a, capability="x", has_items=[{"object": o} for o in (items or {}).get(a, [])])
              for a in plans}
    return PlanGraph("t", lp, offers)


L = lambda a: {"type": "LOCAL", "action": a}


# ------------------------------------------------------------------ 1. demo, WITH Stage 3
def test_demo_with_proposal():
    res = _run(DEMO_SCRIPT)
    st, calls = res["metrics"]["graph"], res["metrics"]["llm"]["calls_by_stage"]
    assert res["joint_plan"]["ok"] and st["unresolved_requests"] == 0
    assert calls["propose"] == 3                                   # agent_4 has nothing to pair -> skipped
    assert st["proposals"] == 2 and st["proposals_mutual"] == 1   # the invalid proposal was dropped
    assert st["edges_by_mutual"] == 1 and st["mutual_kept"] == 1
    assert st["one_sided_accepted"] == 0                           # the wrong one-sided proposal is not used
    assert st["edges_by_completion"] == 2 and st["graph_added_steps"] == 2


# ------------------------------------------------------------------ 2. demo, WITHOUT Stage 3 (ablation)
def test_demo_without_proposal():
    res = _run(DEMO_SCRIPT, use_edge_proposal=False)
    st, calls = res["metrics"]["graph"], res["metrics"]["llm"]["calls_by_stage"]
    assert "propose" not in calls and st["proposals"] == 0
    assert res["joint_plan"]["ok"] and st["unresolved_requests"] == 0
    assert st["edges_by_connect"] == 1 and st["ops_rejected"] == 1


# ------------------------------------------------------------------ 2b. rules only
def test_rules_only():
    res = _run(DEMO_SCRIPT, use_edge_proposal=False, use_llm_reasoner=False)
    assert res["metrics"]["llm"]["calls_by_stage"].get("graph", 0) == 0
    assert res["metrics"]["graph"]["unresolved_requests"] == 2


# ------------------------------------------------------------------ 3. connect rules
def test_connect_rules():
    g = _graph({
        "agent_1": [{"type": "ASK_HELP", "action": "move sofa"}, {"type": "RECEIVE", "action": "r", "item": "mat"}, L("x")],
        "agent_2": [{"type": "HELP", "action": "move sofa"}, {"type": "HELP", "action": "self"}],
    })
    assert not apply_op(g, GraphOp(op="connect", request="1-2", provider="2-1"))[0]   # RECEIVE <- HELP
    assert not apply_op(g, GraphOp(op="connect", request="2-1", provider="1-1"))[0]   # not a request
    assert apply_op(g, GraphOp(op="connect", request="1-1", provider="2-1"))[0]
    assert not apply_op(g, GraphOp(op="connect", request="1-1", provider="2-2"))[0]   # already served


# ------------------------------------------------------------------ 4. complete_transfer needs a declaration
def test_complete_transfer_needs_declaration():
    g = _graph({"agent_1": [{"type": "RECEIVE", "action": "r", "item": "mat"}, L("lay the mat")],
                "agent_2": [L("a")], "agent_3": [L("b")]},
               items={"agent_2": ["bath mat"]})
    assert not apply_op(g, GraphOp(op="complete_transfer", request="1-1", robot="agent_3", item="mat"))[0]
    assert apply_op(g, GraphOp(op="complete_transfer", request="1-1", robot="agent_2", item="bath mat"))[0]
    assert g.nodes["2-2"].type == "PASS" and g.nodes["2-2"].origin == "graph"


# ------------------------------------------------------------------ 5. an item is passed once
def test_item_passed_once():
    g = _graph({"agent_1": [{"type": "RECEIVE", "action": "r", "item": "mat"}, L("lay the mat")],
                "agent_3": [{"type": "RECEIVE", "action": "r", "item": "mat"}, L("lay the mat")],
                "agent_2": [L("a")]},
               items={"agent_2": ["mat"]})
    assert apply_op(g, GraphOp(op="complete_transfer", request="1-1", robot="agent_2"))[0]
    ok, why = apply_op(g, GraphOp(op="complete_transfer", request="3-1", robot="agent_2"))
    assert not ok and "already passes" in why


# ------------------------------------------------------------------ 6. RECEIVE is only added where the item is used
def test_receive_needs_user_step():
    g = _graph({"agent_1": [L("pick bottle"), {"type": "PASS", "action": "give", "item": "water bottle"}],
                "agent_2": [L("sweep the floor")],
                "agent_3": [L("put the water bottle next to the mat")]})
    assert not apply_op(g, GraphOp(op="complete_transfer", provider="1-2", robot="agent_2"))[0]
    assert apply_op(g, GraphOp(op="complete_transfer", provider="1-2", robot="agent_3"))[0]
    rep = rule_verify(g, final=True)
    seq = [n.id for n in g.sequence("agent_3")]
    assert seq.index("3-2") < seq.index("3-1") and rep.ok                    # RECEIVE before its use


# ------------------------------------------------------------------ 7. a link that would deadlock is rejected
def test_cycle_rejected():
    g = _graph({
        "agent_1": [{"type": "ASK_HELP", "action": "x"}, L("a"), {"type": "HELP", "action": "y"}],
        "agent_2": [{"type": "ASK_HELP", "action": "y"}, L("b"), {"type": "HELP", "action": "x"}],
    })
    assert apply_op(g, GraphOp(op="connect", request="1-1", provider="2-3"))[0]
    ok, why = apply_op(g, GraphOp(op="connect", request="2-1", provider="1-3"))
    assert not ok and "cycle" in why
    assert rule_verify(g, final=True).ok


# ------------------------------------------------------------------ 8. drop only LOCAL, with a reason
def test_drop_rules():
    g = _graph({"agent_1": [{"type": "ASK_HELP", "action": "x"}, L("a")], "agent_2": [L("b")]})
    assert not apply_op(g, GraphOp(op="drop", step="1-1", reason="r"))[0]
    assert not apply_op(g, GraphOp(op="drop", step="1-2"))[0]
    assert apply_op(g, GraphOp(op="drop", step="1-2", reason="unrelated"))[0]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS  {name}")
    print("\nall tests passed")
