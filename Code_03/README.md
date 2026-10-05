# CDA — Collaborate, Don't Assign

Allocation-free long-horizon multi-agent planning for heterogeneous robots under partial observability.

```
Stage 1 Offer          (each robot, parallel)  capability / can_do / has_items / need_from_others
Stage 2 Local Plan     (each robot, parallel)  LOCAL / ASK_HELP / RECEIVE, every step cites its can_do
Stage 3 Edge Proposal  (each robot, parallel)  accept | reject | volunteer on broadcast requests
Stage 4 Graph Reasoning (central, structure only)
        4a build dependency graph   4b rules + issues   4c LLM chooses among candidates
        4d validate + apply         4e block unresolvable parts, schedule t_start / t_end
```

## Core rules (enforced in code)

- Step ids (`r2_s3`) are created by code only and never renumbered. LLMs only reference ids.
- Every edge means **src must finish before dst can finish** (SEQ / TRANSFER / HELP).
- HELP / PASS steps exist only as answers to a request (`answers=<request id>`); pairing is by id.
- accept → CONFIRMED edge, volunteer → PROPOSED edge.
- Stage 4 can drop a step or move a provider step (WHEN), but **never creates a step** and
  **never removes a CONFIRMED edge** (`PlanGraph.check_invariants`). Every step in the Joint Plan
  was put there by the robot that executes it.
- Absolute times are computed by the scheduler from `duration`, not written by an LLM.
  ASK_HELP ends when its HELP ends; RECEIVE ends `duration` after its PASS ends;
  HELP outside the helper's room adds `travel_min`.

## Layout

```
cda/
  schemas.py              Node, Edge, Offer, TaskConfig
  llm.py                  OpenAILLM (gpt-4o, JSON, images), MockLLM (scripted)
  prompts.py              all prompts
  stage1_offer.py
  stage2_local_plan.py    + capability validator, 1 self-fix round
  stage3_edge_proposal.py
  graph.py                PlanGraph: rules, issues, ops, scheduling, invariants
  stage4_graph_reasoning.py
  render.py               template Joint Plan + metrics
  pipeline.py             orchestration + CLI
  viz.py                  timeline / dependency figure
examples/task_bath/       task.json, mock_script.json, images/
notebooks/cda_colab.ipynb
tests/test_cda.py
```

## Run

```bash
pip install -r requirements.txt
python tests/test_cda.py                                  # no API key needed
python -m cda.pipeline --task examples/task_bath/task.json --mock examples/task_bath/mock_script.json
export OPENAI_API_KEY=...
python -m cda.pipeline --task examples/task_bath/task.json --model gpt-4o
# ablations
python -m cda.pipeline --task ... --no-graph-llm          # rules-only Stage 4
python -m cda.pipeline --task ... --no-proposal           # no Stage 3
```

Outputs go to `runs/<task_id>/`: offers, local_plans, judgments, graph, metrics, events, joint_plan.md.

## Adding a task

Copy `examples/task_bath/task.json`, put camera images in `images/` and list them under
`agents[].images` (paths relative to task.json). `instruction` is the text about areas the robot
cannot see.

## Metrics (metrics.json)

`n_requests_served`, `n_blocked`, `confirmed_ratio` (share of final collaboration edges agreed
robot-to-robot), `selected_by_graph` (requests whose provider the graph chose among volunteers),
`capability_violations_after_fix`, `plan_fix_rounds`, `graph_ops_applied/rejected`, `makespan`,
`deadline_ok`, LLM calls and tokens.
