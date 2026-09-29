"""Pipeline: Offer -> Local Planning -> [Edge Proposal] -> Graph Reasoning -> Joint Plan.

Offer, Local Planning and Edge Proposal run on every robot concurrently (asyncio.gather);
each phase ends with a barrier. Graph Reasoning is central (rules + one LLM call).
`use_edge_proposal=False` skips Stage 3 (ablation: the graph connects everything alone).

    python pipeline.py --task examples/home_training.json --mock
    python pipeline.py --task examples/home_training.json --model gpt-4o
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from edge_proposal import collect_proposals, propose_edges
from graph_reasoning import graph_reasoning
from llm import BaseLLM
from local_planning import make_local_plan
from offer import make_offer
from render import render_joint_plan
from runtime import Agent, Bus, EventLog
from schemas import Offer, TaskConfig, describe_config, public_offer


async def run_pipeline(
    config: TaskConfig,
    llm: BaseLLM,
    *,
    use_edge_proposal: bool = True,    # False = no Stage 3 (ablation)
    use_llm_reasoner: bool = True,     # False = rules-only ablation
    reasoner_scope: str = "full",      # "full" | "collab" (graph LLM sees only collaboration steps)
    max_graph_ops: int = 20,
    max_validation_retries: int = 0,   # 0 = no retry on invalid JSON
    verbose: bool = True,
    out_dir: str | Path | None = None,
) -> dict:
    log = EventLog()
    bus = Bus(log)
    inputs = config.agent_inputs()
    known = set(inputs)
    agents = [Agent(aid, inp, llm, bus, log, max_validation_retries, verbose=verbose)
              for aid, inp in inputs.items()]

    # ---- 1. Offer (parallel, barrier)
    log.log("offer", "-", "phase_start")
    await asyncio.gather(*[make_offer(a) for a in agents])
    log.log("offer", "-", "phase_end")

    # ---- 2. Local Planning (parallel, barrier)
    log.log("plan", "-", "phase_start")
    await asyncio.gather(*[make_local_plan(a, known) for a in agents])
    log.log("plan", "-", "phase_end")

    offers = {a.id: a.offer for a in agents}
    plans = {a.id: a.plan for a in agents}
    public_offers = {aid: Offer(**public_offer(o)) for aid, o in offers.items()}   # graph sees public only

    # ---- 3. Edge Proposal (parallel, barrier) - optional
    proposals: list[dict] = []
    if use_edge_proposal:
        log.log("propose", "-", "phase_start")
        per_robot = await asyncio.gather(*[propose_edges(a) for a in agents])
        proposals = collect_proposals(per_robot, log)
        log.log("propose", "-", "phase_end")

    # ---- 4. Graph Reasoning (central)
    log.log("graph", "-", "phase_start")
    g = await graph_reasoning(config.task, plans, public_offers, llm, log, proposals=proposals,
                              use_llm=use_llm_reasoner, scope=reasoner_scope,
                              max_ops=max_graph_ops, max_retries=max_validation_retries,
                              verbose=verbose)
    log.log("graph", "-", "phase_end")

    # ---- 5. Render (template)
    joint, joint_text = render_joint_plan(g.graph, g.report)

    result = {
        "task": config.task,
        "offers": {k: v.model_dump() for k, v in offers.items()},          # full (for analysis)
        "local_plans": {k: v.model_dump() for k, v in plans.items()},
        "edge_proposals": proposals,
        "graph": g.graph.to_dict(),
        "graph_ops": g.ops,
        "rule_report_before_llm": g.first_report.to_dict(),
        "rule_report_final": g.report.to_dict(),
        "joint_plan": joint,
        "joint_plan_text": joint_text,
        "metrics": {
            "llm": llm.stats(),
            "graph": g.stats,
            "config": {"use_edge_proposal": use_edge_proposal, "use_llm_reasoner": use_llm_reasoner,
                       "reasoner_scope": reasoner_scope},
            "coordination": {
                "task_allocation": "none (robots volunteer in their own plans)",
                "edges": ("robots propose (Stage 3), mutual ones connected, graph decides the rest"
                          if use_edge_proposal else "built by graph reasoning alone"),
                "private_offer_fields_not_broadcast": ["reasoning", "obs_scope", "cannot_do"],
                "graph_llm_sees_images": False,
            },
        },
    }

    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name, key in [("offers.json", "offers"), ("local_plans.json", "local_plans"),
                          ("edge_proposals.json", "edge_proposals"), ("graph.json", "graph"),
                          ("graph_ops.json", "graph_ops"), ("joint_plan.json", "joint_plan"), ("metrics.json", "metrics")]:
            (out / name).write_text(json.dumps(result[key], ensure_ascii=False, indent=2), encoding="utf-8")
        (out / "joint_plan.txt").write_text(joint_text, encoding="utf-8")
        log.dump(out / "events.jsonl")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="scenario JSON (see examples/)")
    ap.add_argument("--mock", action="store_true", help="scripted LLM answers, no API call")
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--no-proposal", action="store_true", help="skip Stage 3 (Edge Proposal)")
    ap.add_argument("--no-llm-reasoner", action="store_true")
    ap.add_argument("--scope", default="full", choices=["full", "collab"])
    ap.add_argument("--retries", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--out", default="outputs/run_001")
    a = ap.parse_args()

    config = TaskConfig.model_validate_json(Path(a.task).read_text(encoding="utf-8"))
    print(describe_config(config), "\n")

    if a.mock:
        from demo_script import DEMO_SCRIPT
        from llm import ScriptedClient
        llm: BaseLLM = ScriptedClient(DEMO_SCRIPT)
    else:
        from llm import OpenAIClient
        llm = OpenAIClient(model=a.model, max_concurrency=a.concurrency)

    res = asyncio.run(run_pipeline(config, llm, use_edge_proposal=not a.no_proposal,
                                   use_llm_reasoner=not a.no_llm_reasoner,
                                   reasoner_scope=a.scope, max_validation_retries=a.retries,
                                   verbose=not a.quiet, out_dir=a.out))
    print("\n" + res["joint_plan_text"])
    print("\nmetrics:", json.dumps(res["metrics"], ensure_ascii=False))
    print(f"\nsaved to {a.out}/")


if __name__ == "__main__":
    main()
