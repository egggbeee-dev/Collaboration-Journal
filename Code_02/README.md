# Volunteer-Based Multi-Robot Planning

Heterogeneous robots in separate rooms plan one shared task **without a central task allocator**.
Robots write their own steps (the nodes), including what they volunteer to do for others.
Robots propose edges from their own point of view; graph reasoning decides the edges and
repairs the joint plan.

```
Offer ──► Local Plan ──► Edge Proposal ──► Graph Reasoning ──► Joint Plan
N LLM      N LLM          ≤N LLM (text)     rules + 1 LLM       template
parallel   parallel       parallel          central
```

LLM calls per run: `N + N + (≤N) + 1`. Stage 3 can be switched off (`use_edge_proposal=False`,
`--no-proposal`) for the ablation where the graph connects everything alone.

## Stages

| Stage | Who | Output |
|---|---|---|
| 1. Offer | each robot | capability, can_do, has_items, need_from_others (public) · obs_scope, cannot_do, reasoning (private) |
| 2. Local Plan | each robot | own steps: LOCAL / ASK_HELP / HELP / RECEIVE / PASS, with chain-of-thought |
| 3. Edge Proposal | each robot | edges proposed from its own view (one end must be its own step), text only, private knowledge used |
| 4. Graph Reasoning | central | mutual proposals connected first; LLM decides the rest + joint-plan repair (ops below), rule-checked |
| 5. Render | template | `[Step k]` schedule + unresolved / unused / dropped / gaps / warnings |

### Step types

| Type | Meaning | Served by |
|---|---|---|
| LOCAL | I do it myself | — |
| ASK_HELP | my body cannot do a task my part needs | HELP |
| HELP | I do a task another robot needs | — |
| RECEIVE | my part needs an item not in my room | PASS |
| PASS | I give one of my items | — |

### Edges

All edges are directed and mean "must finish before": robot's own order, provider → requester
(HELP → ASK_HELP, PASS → RECEIVE), and cross-robot orders. The graph must stay acyclic.

### Stage 3 proposals

Each robot with a collaboration step proposes at most one partner per step. A proposal is
`mutual` when both robots of the pair proposed it independently: it is connected before the
graph LLM runs and can only be disconnected with a reason. One-sided proposals are evidence
for the graph LLM. Robots with nothing to pair make no LLM call.

### Graph operations (LLM proposes, rules check, violations are rolled back)

| op | effect | rule check |
|---|---|---|
| `connect(request, provider)` | link existing steps | type pair, different robots, one provider per request, item passed once, no cycle |
| `disconnect(request)` | remove a link | — |
| `complete_transfer(request, robot, item)` | add a PASS to `robot` | `robot` declared the item in `has_items` |
| `complete_transfer(provider, robot)` | add a RECEIVE to `robot` | `robot` has an own step that uses the item |
| `add_order(before, after)` | cross-robot order | different robots, no cycle |
| `drop(step, reason)` | remove a LOCAL step | robot's own LOCAL, reason given |
| `flag_gap(description)` | report missing work | — |

The graph never invents work: it links steps robots wrote and completes item transfers robots declared.
Graph-added steps are marked `⟨added by graph⟩` and placed by a rule at the slot that avoids cycles
and shortens the schedule.

## Run

```bash
pip install -r requirements.txt
python pipeline.py --task examples/home_training.json --mock          # offline, scripted LLM
python tests/smoke_test.py                                             # tests
python pipeline.py --task examples/home_training.json --model gpt-4o  # real run (OPENAI_API_KEY, images)
```

Options: `--no-proposal` (skip Stage 3), `--no-llm-reasoner` (rules-only), `--scope collab` (graph LLM sees only collaboration steps),
`--retries N` (retry invalid JSON, default 0), `--quiet`, `--out`.

In Python / Colab:

```python
result = await run_pipeline(config, llm, use_edge_proposal=True, use_llm_reasoner=True,
                            reasoner_scope="full", out_dir="outputs/run_001")
```

## Outputs

`offers.json`, `local_plans.json`, `edge_proposals.json`, `graph.json`, `graph_ops.json` (every op, applied / rejected and why),
`joint_plan.json`, `joint_plan.txt`, `metrics.json`, `events.jsonl`.

## Known limitations

- A robot's own HELP / PASS keeps the position the robot chose; a link that would deadlock is rejected.
- An item is passed at most once. Travel time is not modeled (unit-time steps).
- Item-name checks (`has_items`, "uses the item") are word-based.
