"""Stage 3 - AUCTION (one-shot edge generation, no LLM).

The Auction runs after all Local Plans have been broadcast.

It performs pairwise coordination only:
  NEED = request for another robot to perform a task
  PASS = offer to perform another robot's requested task

The Auction forms high-confidence 1:1 coordination edges in one shot.
Ambiguous/unselected candidate edges remain available to Global Graph Reasoning.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from runtime import EventLog
from schemas import LocalPlan, Offer, Step, agent_of


@dataclass
class Match:
    """Confirmed coordination edge: PASS owner -> NEED owner."""
    need: str
    passed: str
    score: float


@dataclass
class AuctionResult:
    method: str
    matches: list[Match]
    unmatched_needs: list[str]
    unmatched_passes: list[str]
    candidates: dict[str, list[dict]]
    scores: list[dict]
    rounds: int

    def to_dict(self) -> dict:
        d = asdict(self)
        d["matches"] = [asdict(m) for m in self.matches]
        return d


def step_text(step: Step) -> str:
    """Canonical text used for semantic matching."""
    return step.item if (step.kind == "item" and step.item) else step.action


def compute_scores(
    plans: dict[str, LocalPlan],
    offers: dict[str, Offer],
    embedder,
    *,
    hint_bonus: float = 0.05,
    min_score: float = 0.40,
    cannot_do_threshold: float = 0.75,
):
    """Generate feasible NEED-PASS candidate edges."""
    needs = [s for p in plans.values() for s in p.steps if s.type == "NEED"]
    passes = [s for p in plans.values() for s in p.steps if s.type == "PASS"]

    if not needs or not passes:
        return {}, needs, passes

    cannot = [
        (aid, text)
        for aid, offer in offers.items()
        for text in offer.cannot_do
    ]

    texts = (
        [step_text(s) for s in needs + passes]
        + [text for _, text in cannot]
    )

    emb = embedder.embed(texts)

    e_need = emb[: len(needs)]
    e_pass = emb[len(needs): len(needs) + len(passes)]
    e_cannot = emb[len(needs) + len(passes):]

    sim = e_pass @ e_need.T

    scores: dict[tuple[str, str], float] = {}

    for i, passed in enumerate(passes):
        owner = agent_of(passed.id)

        for j, need in enumerate(needs):

            # Do not match a robot with its own NEED.
            if agent_of(need.id) == owner:
                continue

            # NEED and PASS must have the same semantic kind.
            if need.kind != passed.kind:
                continue

            base = float(sim[i, j])

            if base < min_score:
                continue

            # Capability safety guard:
            # do not accept a PASS that contradicts the owner's
            # explicit cannot_do declaration.
            blocked = False

            for k, (aid, _) in enumerate(cannot):
                if aid == owner and e_pass[i] @ e_cannot[k] >= cannot_do_threshold:
                    blocked = True
                    break

            if blocked:
                continue

            # Target is only a weak hint.
            # It cannot create a candidate below min_score.
            hint = 1.0 if passed.target == agent_of(need.id) else 0.0

            scores[(passed.id, need.id)] = base + hint_bonus * hint

    return scores, needs, passes


def _select_edges(scores: dict[tuple[str, str], float]) -> list[Match]:
    """Select a mutually consistent set of high-confidence 1:1 edges in one shot."""
    used_pass: set[str] = set()
    used_need: set[str] = set()
    matches: list[Match] = []

    ordered = sorted(
        scores.items(),
        key=lambda kv: (-kv[1], kv[0][1], kv[0][0]),
    )

    for (passed, need), score in ordered:
        if passed in used_pass or need in used_need:
            continue

        used_pass.add(passed)
        used_need.add(need)

        matches.append(
            Match(
                need=need,
                passed=passed,
                score=float(score),
            )
        )

    return sorted(matches, key=lambda m: m.need)


def _build_candidates(scores, need_ids):
    """Keep all candidate edges so Graph Reasoning can inspect unresolved alternatives."""
    candidates = {need_id: [] for need_id in need_ids}

    for (passed, need), score in scores.items():
        candidates[need].append(
            {
                "pass": passed,
                "score": round(float(score), 4),
            }
        )

    for need_id in candidates:
        candidates[need_id].sort(
            key=lambda c: (-c["score"], c["pass"])
        )

    return candidates


def run_auction(
    plans: dict[str, LocalPlan],
    offers: dict[str, Offer],
    embedder,
    *,
    method: str = "one_shot",
    hint_bonus: float = 0.05,
    min_score: float = 0.40,
    cannot_do_threshold: float = 0.75,
    max_rounds: int = 1,
    log: EventLog | None = None,
) -> AuctionResult:
    """Run one-shot pairwise coordination after Local Plan broadcast.

    method/max_rounds are retained for call-site compatibility; no consensus or
    re-bidding loop is performed.
    """
    if method not in {"one_shot", "greedy", "consensus"}:
        raise ValueError(f"unknown auction method {method!r}")

    scores, needs, passes = compute_scores(
        plans,
        offers,
        embedder,
        hint_bonus=hint_bonus,
        min_score=min_score,
        cannot_do_threshold=cannot_do_threshold,
    )

    need_ids = [need.id for need in needs]
    pass_ids = [passed.id for passed in passes]

    matches = _select_edges(scores)
    candidates = _build_candidates(scores, need_ids)

    matched_needs = {m.need for m in matches}
    matched_passes = {m.passed for m in matches}

    # ================================================================
    # AUCTION DEBUG
    # ================================================================

    print("\n" + "=" * 70)
    print("[AUCTION DEBUG]")
    print("=" * 70)

    # ------------------------------------------------
    # 1. All NEED steps
    # ------------------------------------------------
    print("\n[NEEDS]")

    if needs:
        for n in needs:
            print(
                f"  {n.id} "
                f"| owner={agent_of(n.id)} "
                f"| kind={n.kind} "
                f"| target={n.target} "
                f"| {step_text(n)}"
            )
    else:
        print("  None")

    # ------------------------------------------------
    # 2. All PASS steps
    # ------------------------------------------------
    print("\n[PASSES]")

    if passes:
        for p in passes:
            print(
                f"  {p.id} "
                f"| owner={agent_of(p.id)} "
                f"| kind={p.kind} "
                f"| target={p.target} "
                f"| {step_text(p)}"
            )
    else:
        print("  None")

    # ------------------------------------------------
    # 3. All candidate scores
    # ------------------------------------------------
    print("\n[CANDIDATE SCORES]")

    if scores:
        for (passed, need), score in sorted(
            scores.items(),
            key=lambda kv: (-kv[1], kv[0][1], kv[0][0]),
        ):
            p = next(
                (x for x in passes if x.id == passed),
                None,
            )
            n = next(
                (x for x in needs if x.id == need),
                None,
            )

            target_hint = (
                "YES"
                if p is not None
                and n is not None
                and p.target == agent_of(n.id)
                else "NO"
            )

            print(
                f"  PASS={passed} "
                f"({agent_of(passed)}) "
                f"-> NEED={need} "
                f"({agent_of(need)}) "
                f"| score={score:.4f} "
                f"| target_hint={target_hint}"
            )
    else:
        print("  No candidate edges.")

    # ------------------------------------------------
    # 4. Candidate edges grouped by NEED
    # ------------------------------------------------
    print("\n[CANDIDATES BY NEED]")

    if candidates:
        for need_id, candidate_list in candidates.items():
            print(f"  NEED {need_id}:")

            if not candidate_list:
                print("    - None")
            else:
                for c in candidate_list:
                    print(
                        f"    PASS={c['pass']} "
                        f"| score={c['score']:.4f}"
                    )
    else:
        print("  None")

    # ------------------------------------------------
    # 5. Final selected matches
    # ------------------------------------------------
    print("\n[SELECTED MATCHES]")

    if matches:
        for m in matches:
            print(
                f"  PASS={m.passed} "
                f"({agent_of(m.passed)}) "
                f"-> NEED={m.need} "
                f"({agent_of(m.need)}) "
                f"| score={m.score:.4f}"
            )
    else:
        print("  None")

    # ------------------------------------------------
    # 6. Unmatched NEEDs
    # ------------------------------------------------
    print("\n[UNMATCHED NEEDS]")

    unmatched_needs = [
        n for n in need_ids
        if n not in matched_needs
    ]

    if unmatched_needs:
        for n in unmatched_needs:
            print(f"  {n}")
    else:
        print("  None")

    # ------------------------------------------------
    # 7. Unmatched PASSes
    # ------------------------------------------------
    print("\n[UNMATCHED PASSES]")

    unmatched_passes = [
        p for p in pass_ids
        if p not in matched_passes
    ]

    if unmatched_passes:
        for p in unmatched_passes:
            print(f"  {p}")
    else:
        print("  None")

    print("=" * 70 + "\n")

    # ================================================================
    # RESULT
    # ================================================================

    result = AuctionResult(
        method="one_shot",
        matches=matches,
        unmatched_needs=unmatched_needs,
        unmatched_passes=unmatched_passes,
        candidates=candidates,
        scores=[
            {
                "pass": p,
                "need": n,
                "score": round(float(s), 4),
            }
            for (p, n), s in sorted(
                scores.items(),
                key=lambda kv: (-kv[1], kv[0][1], kv[0][0]),
            )
        ],
        rounds=1,
    )

    if log:
        log.log(
            "auction",
            "-",
            "done",
            method="one_shot",
            matched=len(result.matches),
            unmatched_needs=len(result.unmatched_needs),
            rounds=1,
        )

    return result
