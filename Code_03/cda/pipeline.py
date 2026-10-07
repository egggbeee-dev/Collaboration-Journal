"""CDA pipeline: Offer -> Local Plan -> Edge Proposal -> Graph Reasoning -> Joint Plan.

Stages 1-3 run on every robot concurrently (asyncio.gather) with a barrier between stages.
Stage 4 is central and structure-only.

CLI:
    python -m cda.pipeline --task examples/task_bath/task.json --mock examples/task_bath/mock_script.json
    python -m cda.pipeline --task examples/task_bath/task.json --model gpt-4o
Notebook / Colab (event loop already running):
    res = await run_pipeline(cfg, llm)
"""
from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path

from .llm import BaseLLM, MockLLM, OpenAILLM
from .log import EventLog
from .render import metrics, render
from .schemas import TaskConfig
from .stage1_offer import make_offer
from .stage2_local_plan import make_local_plan
from .stage3_edge_proposal import edge_proposal
from .stage4_graph_reasoning import graph_reasoning


async def run_pipeline(cfg: TaskConfig, llm: BaseLLM, *, use_edge_proposal: bool = True,
                       use_graph_llm: bool = True, max_graph_rounds: int = 2,
                       out_dir: str | Path | None = None, verbose: bool = True) -> dict:
    log = EventLog(verbose)
    ids = cfg.ids

    # 1. Offer (parallel)
    offers_l = await asyncio.gather(*[make_offer(cfg, cfg.agent(a), llm, log) for a in ids])
    offers = dict(zip(ids, offers_l))

    # 2. Local Plan (parallel)
    res = await asyncio.gather(*[make_local_plan(cfg, a, offers, llm, log) for a in ids])
    plans = {a: r[0] for a, r in zip(ids, res)}
    plan_meta = {a: r[1] for a, r in zip(ids, res)}
    local_snapshot = {a: [asdict(n) for n in plans[a]] for a in ids}

    # 3. Edge Proposal (parallel judgments, deterministic insertion)
    collab, judgments = ([], [])
    if use_edge_proposal:
        collab, judgments = await edge_proposal(cfg, plans, offers, llm, log)

    # 4. Graph Reasoning (central)
    g = await graph_reasoning(cfg, plans, collab, offers, judgments,
                              llm if use_graph_llm else None, log, max_graph_rounds)

    text = render(g)
    uncovered = [(a, u) for a in ids for u in plan_meta[a].get("uncovered_checklist", [])]
    if uncovered:     # the robots' own checklists name work that no step does: make it visible
        text += "\n\n### Not covered (from the robots' own checklists)\n" + "\n".join(
            f"- {a}: {u}" for a, u in uncovered)
    m = metrics(g, plan_meta, llm.usage())
    m["config"] = {"use_edge_proposal": use_edge_proposal, "use_graph_llm": use_graph_llm,
                   "max_graph_rounds": max_graph_rounds}
    out = {"offers": {a: o.to_dict() for a, o in offers.items()},
           "local_plans": local_snapshot, "plan_meta": plan_meta,
           "judgments": judgments, "graph": g.to_dict(), "joint_plan": text, "joint_plan_text": text,
           "metrics": m, "events": log.events, "_graph_obj": g}

    if out_dir:
        od = Path(out_dir)
        od.mkdir(parents=True, exist_ok=True)
        for k in ("offers", "local_plans", "plan_meta", "judgments", "graph", "metrics", "events"):
            (od / f"{k}.json").write_text(json.dumps(out[k], ensure_ascii=False, indent=1), encoding="utf-8")
        (od / "joint_plan.md").write_text(text, encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--mock", help="scripted MockLLM json (no API key needed)")
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--no-proposal", action="store_true", help="ablation: skip Stage 3")
    ap.add_argument("--no-graph-llm", action="store_true", help="ablation: rules-only Stage 4")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--out", default=None)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    cfg = TaskConfig.load(a.task)
    llm = MockLLM(a.mock) if a.mock else OpenAILLM(a.model)
    out_dir = a.out or f"runs/{cfg.task_id}{'_mock' if a.mock else ''}"
    res = asyncio.run(run_pipeline(cfg, llm, use_edge_proposal=not a.no_proposal,
                                   use_graph_llm=not a.no_graph_llm, max_graph_rounds=a.rounds,
                                   out_dir=out_dir, verbose=not a.quiet))
    print("\n" + res["joint_plan"])
    print("\nmetrics:", json.dumps({k: v for k, v in res["metrics"].items() if k != "config"},
                                   ensure_ascii=False))
    print(f"\nsaved to {out_dir}/")


if __name__ == "__main__":
    main()
