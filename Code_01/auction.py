"""Stage 3 - AUCTION: P2P candidate coordination, no LLM.

The Auction does not allocate the global task.
Local Planning has already produced independent collaboration intents.

Step-type matching:
    ASK_HELP <-> HELP
    RECEIVE  <-> PASS

The Auction generates candidate coordination edges.
It does NOT perform final global coordination.
Graph Reasoning receives both selected edges and unresolved candidate edges
and determines the final dependency structure.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from runtime import EventLog
from schemas import LocalPlan, Offer, Step, agent_of


# ---------------------------------------------------------------------------
# Result structures
# ---------------------------------------------------------------------------

@dataclass
class Match:
    """Provisional high-confidence coordination edge.

    `need` / `passed` names are kept for backward compatibility with the
    existing Graph interface.

    Semantics:
        ASK_HELP  -> HELP
        RECEIVE   -> PASS

    The Auction only proposes this edge.
    Graph Reasoning decides whether the edge should finally remain.
    """

    need: str
    passed: str
    score: float


@dataclass
class AuctionResult:
    method: str

    # High-confidence provisional edges
    matches: list[Match]

    # Requests/providers that were not selected in the provisional 1:1 matching
    unmatched_needs: list[str]
    unmatched_passes: list[str]

    # ALL candidate edges remain available to Graph Reasoning
    candidates: dict[str, list[dict]]

    # Flat candidate score list
    scores: list[dict]

    rounds: int

    def to_dict(self) -> dict:
        d = asdict(self)
        d["matches"] = [asdict(m) for m in self.matches]
        return d


# ---------------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------------

def step_text(step: Step) -> str:
    """Return the semantic text used for embedding."""

    if step.kind == "item" and step.item:
        return step.item

    return step.action


def _is_request(step: Step) -> bool:
    """Whether a step asks another robot for something."""

    return step.type in {"ASK_HELP", "RECEIVE"}


def _is_provider(step: Step) -> bool:
    """Whether a step offers to provide something to another robot."""

    return step.type in {"HELP", "PASS"}


def _compatible(request: Step, provider: Step) -> bool:
    """Check whether two steps belong to the same collaboration channel.

    Task:
        ASK_HELP <-> HELP

    Item:
        RECEIVE <-> PASS
    """

    if request.type == "ASK_HELP" and provider.type == "HELP":
        return request.kind == "task" and provider.kind == "task"

    if request.type == "RECEIVE" and provider.type == "PASS":
        return request.kind == "item" and provider.kind == "item"

    return False


def _target_hint(
    request: Step,
    provider: Step,
) -> bool:
    """Return True if either side explicitly points to the other robot.

    `target` is only a soft hint.
    It never removes a candidate.
    """

    request_agent = agent_of(request.id)
    provider_agent = agent_of(provider.id)

    # Requester explicitly asks for this provider
    if request.target == provider_agent:
        return True

    # Provider explicitly offers to this requester
    if provider.target == request_agent:
        return True

    return False


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

def _embed_candidate_scores(
    provider_id: str,
    provider_plan: LocalPlan,
    requests: list[Step],
    offers: dict[str, Offer],
    embedder,
    *,
    hint_bonus: float,
    min_score: float,
    cannot_do_threshold: float,
) -> dict[tuple[str, str], float]:
    """Generate candidate edges from one provider's point of view.

    This function is intentionally provider-local.

    A robot only evaluates:
        its own HELP / PASS steps
        against broadcast ASK_HELP / RECEIVE requests.

    No LLM is used.
    """

    providers = [
        s
        for s in provider_plan.steps
        if _is_provider(s)
    ]

    if not providers or not requests:
        return {}

    # A robot cannot collaborate with itself.
    relevant_requests = [
        r
        for r in requests
        if agent_of(r.id) != provider_id
    ]

    if not relevant_requests:
        return {}

    # ------------------------------------------------------------
    # Separate semantic channels
    # ------------------------------------------------------------
    task_providers = [
        s for s in providers
        if s.type == "HELP"
    ]

    item_providers = [
        s for s in providers
        if s.type == "PASS"
    ]

    task_requests = [
        r for r in relevant_requests
        if r.type == "ASK_HELP"
    ]

    item_requests = [
        r for r in relevant_requests
        if r.type == "RECEIVE"
    ]

    # ------------------------------------------------------------
    # Candidate pairs
    # ------------------------------------------------------------
    pairs: list[tuple[Step, Step]] = []

    # ASK_HELP <-> HELP
    for request in task_requests:
        for provider in task_providers:
            if _compatible(request, provider):
                pairs.append((request, provider))

    # RECEIVE <-> PASS
    for request in item_requests:
        for provider in item_providers:
            if _compatible(request, provider):
                pairs.append((request, provider))

    if not pairs:
        return {}

    # ------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------
    request_steps = [request for request, _ in pairs]
    provider_steps = [provider for _, provider in pairs]

    request_texts = [step_text(s) for s in request_steps]
    provider_texts = [step_text(s) for s in provider_steps]

    # Keep embedding calls simple and compatible with the existing embedder.
    all_texts = provider_texts + request_texts

    emb = embedder.embed(all_texts)

    p_count = len(provider_texts)

    e_provider = emb[:p_count]
    e_request = emb[p_count:]

    sim = e_provider @ e_request.T

    scores: dict[tuple[str, str], float] = {}

    # ------------------------------------------------------------
    # Score candidate edges
    # ------------------------------------------------------------
    for i, provider in enumerate(provider_steps):

        for j, request in enumerate(request_steps):

            # Pair list may contain only valid type combinations,
            # but keep this check for safety.
            if not _compatible(request, provider):
                continue

            base = float(sim[i, j])

            if base < min_score:
                continue

            # ----------------------------------------------------
            # cannot_do check
            #
            # Only HELP is subject to task capability checking.
            # PASS is an item-transfer relation, so comparing the
            # item name against cannot_do descriptions is not useful.
            # ----------------------------------------------------
            if provider.type == "HELP":
                offer = offers.get(provider_id)

                if offer:
                    blocked = False

                    for cannot_text in offer.cannot_do:

                        cannot_emb = embedder.embed(
                            [step_text(provider), cannot_text]
                        )

                        cannot_sim = float(
                            cannot_emb[0] @ cannot_emb[1]
                        )

                        if cannot_sim >= cannot_do_threshold:
                            blocked = True
                            break

                    if blocked:
                        continue

            # ----------------------------------------------------
            # Target hint
            # ----------------------------------------------------
            hint = 1.0 if _target_hint(request, provider) else 0.0

            score = base + hint_bonus * hint

            scores[(provider.id, request.id)] = score

    return scores


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------

def compute_scores(
    plans: dict[str, LocalPlan],
    offers: dict[str, Offer],
    embedder,
    *,
    hint_bonus: float = 0.05,
    min_score: float = 0.40,
    cannot_do_threshold: float = 0.75,
):
    """Generate all candidate coordination edges.

    Returns:
        scores:
            (provider_step_id, request_step_id) -> score

        requests:
            ASK_HELP / RECEIVE steps

        providers:
            HELP / PASS steps
    """

    requests = [
        step
        for plan in plans.values()
        for step in plan.steps
        if _is_request(step)
    ]

    providers = [
        step
        for plan in plans.values()
        for step in plan.steps
        if _is_provider(step)
    ]

    scores: dict[tuple[str, str], float] = {}

    # Each provider independently evaluates possible requesters.
    for provider_id, provider_plan in plans.items():

        local_scores = _embed_candidate_scores(
            provider_id,
            provider_plan,
            requests,
            offers,
            embedder,
            hint_bonus=hint_bonus,
            min_score=min_score,
            cannot_do_threshold=cannot_do_threshold,
        )

        scores.update(local_scores)

    return scores, requests, providers


# ---------------------------------------------------------------------------
# Candidate representation
# ---------------------------------------------------------------------------

def _build_candidates(
    scores: dict[tuple[str, str], float],
    request_ids: list[str],
) -> dict[str, list[dict]]:
    """Group candidate providers by request.

    Example:

    {
        "3-5": [
            {
                "pass": "2-7",
                "score": 0.979,
                "provider_type": "HELP",
                "request_type": "ASK_HELP"
            }
        ]
    }

    ALL candidates are preserved.
    """

    candidates = {
        request_id: []
        for request_id in request_ids
    }

    for (provider_id, request_id), score in scores.items():

        candidates[request_id].append(
            {
                "pass": provider_id,
                "score": round(float(score), 4),
            }
        )

    for request_id in candidates:

        candidates[request_id].sort(
            key=lambda x: (
                -x["score"],
                x["pass"],
            )
        )

    return candidates


# ---------------------------------------------------------------------------
# Provisional 1:1 selection
# ---------------------------------------------------------------------------

def _select_edges(
    scores: dict[tuple[str, str], float],
) -> list[Match]:
    """Select provisional high-confidence 1:1 coordination edges.

    IMPORTANT:
        This is NOT the final allocation.

    The purpose of this step is only to provide Graph Reasoning with
    a small set of strong candidate edges.

    Every non-selected candidate remains in `AuctionResult.candidates`
    and `AuctionResult.scores`.

    One request  <-> one provider
    """

    # Highest score first
    ranked = sorted(
        scores.items(),
        key=lambda kv: (
            -float(kv[1]),
            kv[0][1],   # request id
            kv[0][0],   # provider id
        ),
    )

    used_requests: set[str] = set()
    used_providers: set[str] = set()

    matches: list[Match] = []

    for (provider_id, request_id), score in ranked:

        # One-to-one provisional matching
        if request_id in used_requests:
            continue

        if provider_id in used_providers:
            continue

        used_requests.add(request_id)
        used_providers.add(provider_id)

        matches.append(
            Match(
                need=request_id,
                passed=provider_id,
                score=float(score),
            )
        )

    # Deterministic output
    matches.sort(
        key=lambda m: (
            m.need,
            m.passed,
        )
    )

    return matches


# ---------------------------------------------------------------------------
# Main Auction
# ---------------------------------------------------------------------------

def run_auction(
    plans: dict[str, LocalPlan],
    offers: dict[str, Offer],
    embedder,
    *,
    method: str = "p2p_candidates",
    hint_bonus: float = 0.05,
    min_score: float = 0.40,
    cannot_do_threshold: float = 0.75,
    max_rounds: int = 1,
    log: EventLog | None = None,
) -> AuctionResult:
    """Generate P2P collaboration candidate edges.

    Pipeline position:

        Local Planning
              ↓
        Auction
              ↓
        Graph Reasoning

    Auction:
        - no LLM
        - no global task decomposition
        - no final allocation
        - no final dependency resolution
        - generates semantically compatible candidate edges

    Matching channels:

        ASK_HELP  <-> HELP
        RECEIVE   <-> PASS

    Graph Reasoning receives:
        1. provisional high-confidence matches
        2. all candidate edges
        3. unresolved requests/providers
    """

    scores, requests, providers = compute_scores(
        plans,
        offers,
        embedder,
        hint_bonus=hint_bonus,
        min_score=min_score,
        cannot_do_threshold=cannot_do_threshold,
    )

    request_ids = [
        request.id
        for request in requests
    ]

    provider_ids = [
        provider.id
        for provider in providers
    ]

    # ------------------------------------------------------------
    # ALL candidate edges
    # ------------------------------------------------------------
    candidates = _build_candidates(
        scores,
        request_ids,
    )

    # ------------------------------------------------------------
    # Provisional high-confidence 1:1 edges
    # ------------------------------------------------------------
    matches = _select_edges(scores)

    matched_requests = {
        match.need
        for match in matches
    }

    matched_providers = {
        match.passed
        for match in matches
    }

    # ------------------------------------------------------------
    # Unmatched entities
    # ------------------------------------------------------------
    unmatched_requests = [
        request_id
        for request_id in request_ids
        if request_id not in matched_requests
    ]

    unmatched_providers = [
        provider_id
        for provider_id in provider_ids
        if provider_id not in matched_providers
    ]

    # ------------------------------------------------------------
    # Flat score list
    # ------------------------------------------------------------
    score_list = [
        {
            "pass": provider_id,
            "need": request_id,
            "score": round(float(score), 4),
        }
        for (provider_id, request_id), score in sorted(
            scores.items(),
            key=lambda kv: (
                -float(kv[1]),
                kv[0][1],
                kv[0][0],
            ),
        )
    ]

    result = AuctionResult(
        method="p2p_candidates",
        matches=matches,
        unmatched_needs=unmatched_requests,
        unmatched_passes=unmatched_providers,
        candidates=candidates,
        scores=score_list,
        rounds=1,
    )

    # ------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------
    if log:

        log.log(
            "auction",
            "-",
            "done",
            method="p2p_candidates",
            matched=len(result.matches),
            candidate_edges=len(score_list),
            unmatched_needs=len(result.unmatched_needs),
            unmatched_passes=len(result.unmatched_passes),
            rounds=1,
        )

    return result
