"""Stage 3 - AUCTION (edge generation, no LLM).

The Auction does NOT allocate tasks. Every robot has already planned for itself; the Auction
only decides which PASS step serves which NEED step (a "handoff" edge), 1:1.

  requests   = NEED steps (item or task)
  volunteers = PASS steps of other robots with the same `kind`
  score      = cosine( embed(request), embed(offer) )            (+ hint_bonus if the PASS hint
               already points at the requester)

A candidate must reach `min_score` on similarity alone (the hint can never justify a wrong
match) and must not look like something its owner declared in `cannot_do`.

Methods (same interface, swap with `method=`):
  "consensus": CBAA-style rounds. Every free PASS step bids on the request it can win, bids are
               merged (highest wins, ties -> lower step id), a step that is outbid becomes free
               and bids again next round, until nothing changes. No central chooser.
  "greedy"   : highest score first with the 1:1 constraint (simple baseline / volunteer-style).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from runtime import EventLog
from schemas import LocalPlan, Offer, Step, agent_of


@dataclass
class Match:
    need: str     # NEED step id (the receiver side)
    passed: str   # PASS step id (the provider side)
    score: float


@dataclass
class AuctionResult:
    method: str
    matches: list[Match]
    unmatched_needs: list[str]
    unmatched_passes: list[str]          # their owners withdraw these steps
    candidates: dict[str, list[dict]]    # need id -> [{"pass": id, "score": x}] (self-nominated volunteers)
    scores: list[dict]
    rounds: int

    def to_dict(self) -> dict:
        d = asdict(self)
        d["matches"] = [asdict(m) for m in self.matches]
        return d


def step_text(step: Step) -> str:
    """Text that gets embedded: the canonical item name for items, the sentence for tasks."""
    return step.item if (step.kind == "item" and step.item) else step.action


def compute_scores(
    plans: dict[str, LocalPlan],
    offers: dict[str, Offer],
    embedder,
    *,
    hint_bonus: float = 0.5,
    min_score: float = 0.4,
    cannot_do_threshold: float = 0.75,
):
    needs = [s for p in plans.values() for s in p.steps if s.type == "NEED"]
    passes = [s for p in plans.values() for s in p.steps if s.type == "PASS"]
    if not needs or not passes:
        return {}, needs, passes

    cannot = [(aid, t) for aid, o in offers.items() for t in o.cannot_do]
    emb = embedder.embed([step_text(s) for s in needs + passes] + [t for _, t in cannot])
    e_need = emb[: len(needs)]
    e_pass = emb[len(needs): len(needs) + len(passes)]
    e_cannot = emb[len(needs) + len(passes):]
    sim = e_pass @ e_need.T

    scores: dict[tuple[str, str], float] = {}
    for i, p in enumerate(passes):
        owner = agent_of(p.id)
        # infeasible: the owner declared it cannot do something that looks like this
        if any(aid == owner and float(e_pass[i] @ e_cannot[j]) >= cannot_do_threshold for j, (aid, _) in enumerate(cannot)):
            continue
        for j, n in enumerate(needs):
            if agent_of(n.id) == owner or n.kind != p.kind:
                continue
            base = float(sim[i, j])
            if base < min_score:
                continue
            scores[(p.id, n.id)] = base + (hint_bonus if p.target == agent_of(n.id) else 0.0)
    return scores, needs, passes


# --------------------------------------------------------------------------- methods
def _consensus(scores, need_ids, pass_ids, max_rounds, log):
    rank = {p: i for i, p in enumerate(sorted(pass_ids))}

    def key(p, s):  # higher is better; ties go to the lower step id
        return (round(s, 9), -rank[p])

    y: dict[str, tuple] = {}              # need -> (bid key, holder pass)  (the shared bid table)
    holds: dict[str, str | None] = {p: None for p in pass_ids}
    rounds = 0
    for _ in range(max_rounds):
        # local step: every free PASS step bids on its best request that it can still win
        proposals: dict[str, tuple[str, tuple]] = {}
        for p in sorted(pass_ids):
            if holds[p] is not None:
                continue
            best = None
            for n in sorted(need_ids):
                s = scores.get((p, n))
                if s is None:
                    continue
                k = key(p, s)
                if n in y and k <= y[n][0]:
                    continue  # cannot outbid the current holder
                if best is None or k > best[1]:
                    best = (n, k)
            if best:
                proposals[p] = best
        if not proposals:
            break
        rounds += 1
        # consensus step: per request, the highest proposal wins; the old holder is released
        changed = 0
        for n in sorted({n for n, _ in proposals.values()}):
            p_best, (_, k_best) = max(((p, v) for p, v in proposals.items() if v[0] == n), key=lambda t: t[1][1])
            if n in y and k_best <= y[n][0]:
                continue
            if n in y:
                holds[y[n][1]] = None
            y[n] = (k_best, p_best)
            holds[p_best] = n
            changed += 1
        if log:
            log.log("auction", "-", "round", round=rounds, proposals=len(proposals), changes=changed)
    matches = [Match(need=n, passed=p, score=scores[(p, n)]) for n, (_, p) in sorted(y.items())]
    return matches, rounds


def _greedy(scores):
    used_p, used_n, matches = set(), set(), []
    for (p, n), s in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0][1], kv[0][0])):
        if p in used_p or n in used_n:
            continue
        used_p.add(p)
        used_n.add(n)
        matches.append(Match(need=n, passed=p, score=s))
    return sorted(matches, key=lambda m: m.need), 1


# --------------------------------------------------------------------------- entry point
def run_auction(
    plans: dict[str, LocalPlan],
    offers: dict[str, Offer],
    embedder,
    *,
    method: str = "consensus",
    hint_bonus: float = 0.5,
    min_score: float = 0.4,
    cannot_do_threshold: float = 0.75,
    max_rounds: int = 50,
    log: EventLog | None = None,
) -> AuctionResult:
    scores, needs, passes = compute_scores(
        plans, offers, embedder, hint_bonus=hint_bonus, min_score=min_score, cannot_do_threshold=cannot_do_threshold
    )
    need_ids = [n.id for n in needs]
    pass_ids = [p.id for p in passes]
    if method == "consensus":
        matches, rounds = _consensus(scores, need_ids, pass_ids, max_rounds, log)
    elif method == "greedy":
        matches, rounds = _greedy(scores)
    else:
        raise ValueError(f"unknown auction method {method!r}")

    cands: dict[str, list[dict]] = {n: [] for n in need_ids}
    for (p, n), s in scores.items():
        cands[n].append({"pass": p, "score": round(s, 4)})
    for n in cands:
        cands[n].sort(key=lambda c: (-c["score"], c["pass"]))

    matched_n = {m.need for m in matches}
    matched_p = {m.passed for m in matches}
    result = AuctionResult(
        method=method,
        matches=matches,
        unmatched_needs=[n for n in need_ids if n not in matched_n],
        unmatched_passes=[p for p in pass_ids if p not in matched_p],
        candidates=cands,
        scores=[{"pass": p, "need": n, "score": round(s, 4)} for (p, n), s in sorted(scores.items())],
        rounds=rounds,
    )
    if log:
        log.log("auction", "-", "done", method=method, matched=len(matches), unmatched_needs=len(result.unmatched_needs),
                withdrawn_passes=len(result.unmatched_passes), rounds=rounds)
    return result
