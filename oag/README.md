# Offer–Auction–Graph: Multi-Robot Planning in Spatially Separated Environments

Heterogeneous robots, each in its own room and seeing only its own images, jointly
plan for one shared task **without a central task allocator**. Each robot decides
*what* it does; collaboration emerges from the dependencies in the robots' own plans.

```
Offer ──► Local Plan ──► Auction (proposal) ──► Graph Reasoning ──► Joint Plan
 N LLM      N LLM          no LLM, parallel       rules + 1 LLM       template
 parallel   parallel
```

## Pipeline

| Stage | Where | What |
|---|---|---|
| 1. Offer | each robot, parallel | Declares capability, can_do, cannot_do, can_provide, needs. Only the **public part** is broadcast (`obs_scope`, `cannot_do` stay private). |
| 2. Local Plan | each robot, parallel | Writes its own plan with `LOCAL` / `ASK_HELP` / `RECEIVE`. A request must be placed right before the own step it enables (`purpose`). |
| 3. Auction | each robot, parallel | Reads others' requests and **volunteers** HELP / PASS steps from its own `can_provide` (skipping its private `cannot_do`). Produces candidate edges only — **no selection**. |
| 4. Graph Reasoning | central | Layer 1 (rules): initial 1:1 selection, one PASS per item, placement of accepted HELP/PASS in the provider's sequence, cycle breaking, topological schedule. Layer 2 (LLM): `reassign` / `unmatch` / `add_order` **only among proposed candidates**. Rules run again as a guard. |
| 5. Render | central | Deterministic, template-based Joint Plan (no LLM). |

LLM calls per run: `N (offer) + N (plan) + 1 (graph)`.

### Step types

| Type | Written by | Meaning |
|---|---|---|
| `LOCAL` | Local Plan | I do this myself. |
| `ASK_HELP` | Local Plan | My embodiment cannot do this task, and my later step depends on it. |
| `RECEIVE` | Local Plan | My later step needs this item, which is not in my room. |
| `HELP` | Auction (self-proposed) | I will do the requested task. |
| `PASS` | Auction (self-proposed) | I will give my item. An item is passed at most once. |

Pairs: `ASK_HELP ↔ HELP`, `RECEIVE ↔ PASS`.

### Graph

- **Nodes**: steps (proposed but unselected HELP/PASS stay inactive).
- **Edges**: `sequence` (a robot's own order), `collaboration` (provider → requester), `order` (added by the graph LLM). Candidate edges are visible to the LLM but not scheduled.
- The schedule is the topological level order; `[Step k]` groups steps that can run in parallel.

## Quick start

```bash
pip install -r requirements.txt

# offline: scripted LLM + hashing embedder (no API key)
python pipeline.py --task examples/home_training.json --mock --embedder hash

# tests
python tests/smoke_test.py

# real run (needs OPENAI_API_KEY and images in the scenario file)
python pipeline.py --task examples/home_training.json --model gpt-4o --embedder sbert
```

Options: `--no-llm-reasoner` (rules-only ablation), `--quiet`, `--concurrency`, `--out`.

### Scenario format

```json
{
  "task": "Prepare the living room for exercise.",
  "agents": [
    {"capability": "...", "images": ["path/a.jpg", "path/b.jpg"], "hidden_info": ["..."]}
  ]
}
```

Robots are named `agent_1 … agent_N` in order (rendered as `R1 … RN`).

### Outputs (`--out`, default `outputs/run_001/`)

`offers.json`, `local_plans.json`, `auction.json` (proposals / candidates), `graph.json`,
`graph_ops.json`, `joint_plan.json`, `joint_plan.txt`, `metrics.json`, `events.jsonl`
(every broadcast, retry, fallback and graph op).

## Robustness

Every LLM stage retries on invalid JSON / rule violations (`max_validation_retries=2`).
If all attempts fail, the run continues with a logged fallback (`fallback_used` in `events.jsonl`):

- Offer → minimal offer (capability only)
- Local Plan → keep valid steps, drop only rule-breaking ones (`fallback_salvaged`)
- Graph LLM → keep only well-formed ops (otherwise rules-only result)

## Parameters to tune

`auction_kwargs` in `run_pipeline`: `min_score` (0.40), `cannot_do_threshold` (0.75), `hint_bonus` (0.05).
Defaults were checked with the hashing embedder; recalibrate for SBERT.

## Known limitations

- An item is passed at most once (no circulation of the same item between robots).
- Spatial conflicts between robots in the same room are left to the graph LLM's `add_order`.
- Scheduling assumes unit-time steps (no travel time).
