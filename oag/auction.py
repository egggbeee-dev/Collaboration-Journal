"""Stage 3 - AUCTION: decentralized candidate proposal (no LLM, no selection).

The Auction does NOT allocate tasks and does NOT select edges.

After every Local Plan has been broadcast, EACH robot, in parallel and on
its own, reads the other robots' requests and asks:

    "Which of these requests can I answer with something I declared in my
     own CAN_PROVIDE?"

    ASK_HELP  <-  my CAN_PROVIDE(type=task)   ->  I propose a HELP step
    RECEIVE   <-  my CAN_PROVIDE(type=item)   ->  I propose a PASS step

A proposal is only made if
    - the request and my offering are semantically similar (embedding),
    - for HELP: the request does not hit my own (private) CANNOT_DO.

Every proposal appends a HELP / PASS step to the END of the proposing
robot's own plan (inactive until Graph Reasoning selects it) and becomes a
candidate edge  provider_step -> request_step.

Graph Reasoning later selects among these self-proposed candidates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from runtime import Agent, EventLog
from schemas import CanProvide, LocalPlan, Step, agent_of


REQUEST_TYPES = {"ASK_HELP", "RECEIVE"}

# request type -> (CAN_PROVIDE type, proposed step type)
CHANNEL = {
    "ASK_HELP": ("task", "HELP"),
    "RECEIVE": ("item", "PASS"),
}


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class AuctionResult:
    method: str

    # request_id -> [{"pass": provider_step_id, "score": float}, ...]
    # (key name "pass" kept for compatibility with Graph Reasoning)
    candidates: dict[str, list[dict]]

    # flat list of all proposals
    proposals: list[dict] = field(default_factory=list)

    # requests nobody proposed to answer
    requests_without_candidates: list[str] = field(default_factory=list)

    rounds: int = 1

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Text used for embedding
# ---------------------------------------------------------------------------

def request_text(step: Step) -> str:
    if step.type == "RECEIVE" and step.item:
        return step.item
    return step.action


def offering_text(p: CanProvide) -> str:
    if p.type == "item":
        return p.object or ""
    return " ".join(x for x in (p.action, p.object) if x)


# ---------------------------------------------------------------------------
# One robot's proposal (runs inside that robot, in parallel with others)
# ---------------------------------------------------------------------------

async def propose_responses(
    agent: Agent,
    embedder,
    *,
    min_score: float = 0.40,
    cannot_do_threshold: float = 0.75,
    hint_bonus: float = 0.05,
) -> list[dict]:
    """Let one robot propose HELP / PASS steps for other robots' requests.

    Uses only:
        - other robots' broadcast Local Plans (requests)
        - this robot's own Offer (CAN_PROVIDE, private CANNOT_DO)
    """

    assert agent.plan is not None and agent.offer is not None

    agent.receive()  # other robots' Local Plans

    # deterministic order (agent number, step order), independent of
    # which plan happened to arrive first
    requests = sorted(
        (
            s
            for plan in agent.others_plans.values()
            for s in plan.steps
            if s.type in REQUEST_TYPES
        ),
        key=lambda s: (int(s.id.split("-")[0]), s.order),
    )

    offerings = [p for p in agent.offer.can_provide if offering_text(p)]

    proposals: list[dict] = []

    if requests and offerings:

        cannot_texts = [
            " ".join(x for x in (c.action, c.object) if x)
            for c in agent.offer.cannot_do
        ]

        emb = embedder.embed(
            [request_text(r) for r in requests]
            + [offering_text(p) for p in offerings]
            + cannot_texts
        )

        n_r, n_o = len(requests), len(offerings)
        e_req = emb[:n_r]
        e_off = emb[n_r:n_r + n_o]
        e_cannot = emb[n_r + n_o:]

        for i, req in enumerate(requests):

            provide_type, step_type = CHANNEL[req.type]

            # private capability guard: never volunteer for a task
            # that matches my own CANNOT_DO
            if step_type == "HELP" and any(
                float(e_req[i] @ c) >= cannot_do_threshold for c in e_cannot
            ):
                continue

            best_j, best = None, -1.0
            for j, off in enumerate(offerings):
                if off.type != provide_type:
                    continue
                s = float(e_req[i] @ e_off[j])
                if s > best:
                    best_j, best = j, s

            if best_j is None or best < min_score:
                continue

            requester = agent_of(req.id)
            score = best + (hint_bonus if req.target == agent.id else 0.0)

            if step_type == "HELP":
                step = agent.plan.add_proposed_step(
                    type="HELP",
                    action=req.action,
                    serves=req.id,
                    target=requester,
                )
            else:
                item = offerings[best_j].object
                step = agent.plan.add_proposed_step(
                    type="PASS",
                    action=f"Pass the {item}",
                    item=item,
                    serves=req.id,
                    target=requester,
                )

            proposals.append({
                "request": req.id,
                "provider": step.id,
                "type": step_type,
                "score": round(score, 4),
                "offering": offering_text(offerings[best_j]),
            })

    agent.log.log("auction", agent.id, "proposed", n=len(proposals), proposals=proposals)

    if agent.verbose:
        for p in proposals:
            print(
                f"  [PROPOSE] {agent.id}: {p['type']} {p['provider']} -> "
                f"request {p['request']} (score={p['score']:.3f}, via '{p['offering']}')"
            )
        if not proposals:
            print(f"  [PROPOSE] {agent.id}: no proposal")

    agent.bus.broadcast(agent.id, "proposal", {"proposals": proposals}, phase="auction")

    return proposals


# ---------------------------------------------------------------------------
# Collect proposals into candidate edges (bookkeeping only, no decision)
# ---------------------------------------------------------------------------

def collect_candidates(
    plans: dict[str, LocalPlan],
    all_proposals: list[list[dict]],
    log: EventLog | None = None,
) -> AuctionResult:

    request_ids = [
        s.id
        for plan in plans.values()
        for s in plan.steps
        if s.type in REQUEST_TYPES
    ]

    candidates: dict[str, list[dict]] = {rid: [] for rid in request_ids}
    flat: list[dict] = []

    for props in all_proposals:
        for p in props:
            candidates.setdefault(p["request"], []).append(
                {"pass": p["provider"], "score": p["score"]}
            )
            flat.append(p)

    for rid in candidates:
        candidates[rid].sort(key=lambda c: (-c["score"], c["pass"]))

    result = AuctionResult(
        method="decentralized_proposal",
        candidates=candidates,
        proposals=flat,
        requests_without_candidates=[r for r, c in candidates.items() if not c],
    )

    if log:
        log.log(
            "auction", "-", "done",
            method=result.method,
            n_requests=len(request_ids),
            n_proposals=len(flat),
            requests_without_candidates=len(result.requests_without_candidates),
        )

    return result
