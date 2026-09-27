"""Whole pipeline: Offer -> Local Planning -> Auction -> Graph Reasoning -> template Joint Plan.

Inside the Offer and Local Planning phases all robots run concurrently (asyncio.gather);
each phase ends with a barrier (everybody finishes before the next phase starts), which keeps
runs reproducible while every robot still behaves like an independent process.

    python pipeline.py --task examples/home_training.json --mock --embedder hash
    python pipeline.py --task examples/home_training.json --model gpt-4o
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from auction import run_auction
from graph_reasoning import graph_reasoning
from llm import BaseLLM
from local_planning import make_local_plan
from offer import make_offer
from render import render_joint_plan
from runtime import Agent, Bus, EventLog
from schemas import TaskConfig, describe_config


async def run_pipeline(
    config: TaskConfig,
    llm: BaseLLM,
    embedder,
    *,
    auction_method: str = "consensus",       # "consensus" (consensus-style) | "greedy"
    use_llm_reasoner: bool = True,           # False = ablation "rules only"
    reasoner_scope: str = "full",            # "full" | "collab" (what text the graph LLM may read)
    auction_kwargs: dict | None = None,      # hint_bonus, min_score, cannot_do_threshold, ...
    max_validation_retries: int = 2,
    verbose: bool = True,                    # print banners + raw LLM responses as the pipeline runs
    out_dir: str | Path | None = None,
) -> dict:
    log = EventLog()
    bus = Bus(log)
    inputs = config.agent_inputs()
    known = set(inputs)
    agents = [Agent(aid, inp, llm, bus, log, max_validation_retries, verbose=verbose) for aid, inp in inputs.items()]

    # ---- 1. Offer (parallel, then barrier)
    log.log("offer", "-", "phase_start")
    await asyncio.gather(*[make_offer(a) for a in agents])
    log.log("offer", "-", "phase_end")

    # ---- 2. Local planning (parallel, then barrier)
    log.log("plan", "-", "phase_start")
    await asyncio.gather(*[make_local_plan(a, known) for a in agents])
    log.log("plan", "-", "phase_end")

    offers = {a.id: a.offer for a in agents}
    plans = {a.id: a.plan for a in agents}

    # ---- 3. Auction (edge generation, no LLM)
    log.log("auction", "-", "phase_start")
    auction = run_auction(plans, offers, embedder, method=auction_method, log=log, **(auction_kwargs or {}))
    log.log("auction", "-", "phase_end")

    # ---- 4. Graph reasoning (rules + constrained LLM repair + rules again)
    log.log("graph", "-", "phase_start")
    g = await graph_reasoning(config.task, plans, auction, llm, log, use_llm=use_llm_reasoner, scope=reasoner_scope)
    log.log("graph", "-", "phase_end")

    # ---- 5. Template rendering (rule-based)
    joint, joint_text = render_joint_plan(g.graph, g.report)

    result = {
        "task": config.task,
        "offers": {k: v.model_dump() for k, v in offers.items()},
        "local_plans": {k: v.model_dump() for k, v in plans.items()},
        "auction": auction.to_dict(),
        "graph": g.graph.to_dict(),
        "graph_ops": g.ops,
        "rule_report_before_llm": g.first_report.to_dict(),
        "rule_report_final": g.report.to_dict(),
        "joint_plan": joint,
        "joint_plan_text": joint_text,
        "metrics": {
            "llm": llm.stats(),
            "auction": {"method": auction.method, "rounds": auction.rounds, "matched": len(auction.matches),
                        "unmatched_needs": len(auction.unmatched_needs), "withdrawn_passes": len(auction.unmatched_passes)},
            "graph": g.stats,
            "coordination": {
                "local_plan_broadcasted": True,
                "matching_is_centralized_in_this_reference_implementation": True,
                "raw_private_inputs_sent_to_graph_reasoner": False,
            },
        },
    }

    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        files = {
            "offers.json": result["offers"], "local_plans.json": result["local_plans"], "auction.json": result["auction"],
            "graph.json": result["graph"], "graph_ops.json": result["graph_ops"], "joint_plan.json": joint, "metrics.json": result["metrics"],
        }
        for name, obj in files.items():
            (out / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        (out / "joint_plan.txt").write_text(joint_text, encoding="utf-8")
        log.dump(out / "events.jsonl")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="scenario JSON (see examples/)")
    ap.add_argument("--mock", action="store_true", help="canned LLM answers, no API call")
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--embedder", default="sbert", choices=["sbert", "hash"])
    ap.add_argument("--auction", default="consensus", choices=["consensus", "greedy"])
    ap.add_argument("--no-llm-reasoner", action="store_true")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--quiet", action="store_true", help="suppress per-step banners/raw responses")
    ap.add_argument("--out", default="outputs/run_001")
    a = ap.parse_args()

    from embedding import get_embedder

    config = TaskConfig.model_validate_json(Path(a.task).read_text(encoding="utf-8"))
    print(describe_config(config))
    print()
    if a.mock:
        from demo_script import DEMO_SCRIPT
        from llm import ScriptedClient

        llm: BaseLLM = ScriptedClient(DEMO_SCRIPT)
    else:
        from llm import OpenAIClient

        llm = OpenAIClient(model=a.model, max_concurrency=a.concurrency)

    res = asyncio.run(run_pipeline(config, llm, get_embedder(a.embedder), auction_method=a.auction,
                                   use_llm_reasoner=not a.no_llm_reasoner, verbose=not a.quiet, out_dir=a.out))
    print(res["joint_plan_text"])
    print("\nmetrics:", json.dumps(res["metrics"], ensure_ascii=False))
    print(f"\nsaved to {a.out}/")


if __name__ == "__main__":
    main()
