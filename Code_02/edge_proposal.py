"""Stage 3 - EDGE PROPOSAL (each robot, in parallel, one text-only LLM call).

After every Local Plan is broadcast, each robot looks at the other robots' collaboration
steps and proposes edges FROM ITS OWN POINT OF VIEW:

    "my HELP 2-2 serves agent_3's ASK_HELP 3-1"      (one end must be my own step)

It judges with its PRIVATE knowledge (observations, cannot_do, hidden info), which only it has.
Proposals are not decisions: Graph Reasoning decides. When BOTH robots of a pair propose the same
edge independently, it is a mutual edge and is connected before the graph LLM runs.

Edges are directed provider -> requester (HELP -> ASK_HELP, PASS -> RECEIVE).
"""

from __future__ import annotations

from runtime import Agent, EventLog
from schemas import PAIR, PROVIDER_TYPES, REQUEST_TYPES, LocalPlan, Step, agent_of

COLLAB = REQUEST_TYPES | PROVIDER_TYPES
PARTNER_TYPE = {"ASK_HELP": "HELP", "HELP": "ASK_HELP", "RECEIVE": "PASS", "PASS": "RECEIVE"}


# =============================================================================
# Prompt
# =============================================================================

PROPOSE_SYSTEM = """You are one robot in a team of heterogeneous robots in separate rooms.
Every robot has shared its plan. Now you propose which of YOUR collaboration steps should be
linked to which step of ANOTHER robot. You only propose; a later stage decides.

PAIRS (the only valid links)
- your HELP     <-> another robot's ASK_HELP   (you do the task it cannot do)
- your ASK_HELP <-> another robot's HELP       (it does the task you cannot do)
- your PASS     <-> another robot's RECEIVE    (you give the item it needs)
- your RECEIVE  <-> another robot's PASS       (it gives the item you need)

THINK FIRST (in "reasoning"):
1) For each of your collaboration steps, which listed steps could be its partner?
2) Using your private knowledge (what you see, what your body cannot do, hidden info):
   can this link really work? (Can you really do that task? Is the item really usable?)
3) Pick at most ONE partner per step of yours. Skip a step if nothing fits.

RULES
- "mine" must be one of YOUR step ids, "other" one of the LISTED step ids of other robots.
- Match by meaning (a "bath mat" can serve a request for an "exercise mat").
- Never invent ids. Proposing nothing is fine.

Return ONE JSON object:
{"reasoning": "...", "edges": [{"mine": "<your id>", "other": "<their id>", "why": "..."}]}
"""


def _fmt(s: Step) -> str:
    return f"{s.id} [{s.type}] {s.action}" + (f" (item: {s.item})" if s.item else "") + \
           (f" (target: {s.target})" if s.target else "")


def build_propose_user(agent: Agent, candidates: list[Step]) -> str:
    o = agent.offer
    own_steps = "\n".join(f"- {_fmt(s)}" for s in agent.plan.steps) or "- (none)"
    others = "\n".join(f"- {agent_of(s.id)}: {_fmt(s)}" for s in candidates)
    obs = ", ".join(x.object for x in o.obs_scope) or "(none)"
    cannot = ", ".join(f"{c.action} {c.object or ''}".strip() for c in o.cannot_do) or "(none)"
    hidden = "; ".join(agent.inp.hidden_info) or "(none)"
    items = ", ".join(h.object for h in o.has_items) or "(none)"
    return (
        f"TASK:\n{agent.inp.task}\n\nYOU ARE: {agent.id}\nCAPABILITY: {agent.inp.capability}\n\n"
        f"YOUR PRIVATE KNOWLEDGE:\n- you observe: {obs}\n- you cannot: {cannot}\n"
        f"- hidden info: {hidden}\n- your items: {items}\n\n"
        f"YOUR PLAN:\n{own_steps}\n\n"
        f"OTHER ROBOTS' COLLABORATION STEPS (possible partners):\n{others}\n\nReturn JSON only."
    )


# =============================================================================
# One robot
# =============================================================================

def _parse(raw: dict) -> dict:
    if not isinstance(raw.get("edges", []), list):
        raise ValueError("`edges` must be a list")
    return raw


async def propose_edges(agent: Agent) -> list[dict]:
    """Returns validated proposals as {"request", "provider", "by", "why"} (provider -> request)."""
    assert agent.plan is not None and agent.offer is not None
    agent.receive()   # other robots' plans

    mine = {s.id: s for s in agent.plan.steps if s.type in COLLAB}
    others = {s.id: s for p in agent.others_plans.values() for s in p.steps if s.type in COLLAB}
    wanted = {PARTNER_TYPE[s.type] for s in mine.values()}
    candidates = sorted((s for s in others.values() if s.type in wanted),
                        key=lambda s: (int(s.id.split("-")[0]), s.order))

    proposals: list[dict] = []
    if not mine or not candidates:
        agent.log.log("propose", agent.id, "skipped", reason="nothing to pair")
        if agent.verbose:
            print(f"  [PROPOSE] {agent.id}: skipped (nothing to pair)")
    else:
        raw = await agent.ask_text("propose", PROPOSE_SYSTEM, build_propose_user(agent, candidates),
                                   parse=_parse, banner_label="EDGE PROPOSAL RAW")
        used: set[str] = set()
        for e in raw.get("edges", []):
            m, o = mine.get(str(e.get("mine", ""))), others.get(str(e.get("other", "")))
            why = str(e.get("why", ""))
            if m is None or o is None or PARTNER_TYPE[m.type] != o.type or m.id in used:
                agent.log.log("propose", agent.id, "proposal_rejected", proposal=e)
                if agent.verbose:
                    print(f"  [PROPOSE] {agent.id}: rejected {e}")
                continue
            used.add(m.id)
            req, prov = (m, o) if m.type in REQUEST_TYPES else (o, m)
            proposals.append({"request": req.id, "provider": prov.id, "by": agent.id, "why": why})

        if agent.verbose:
            for p in proposals:
                print(f"  [PROPOSE] {agent.id}: {p['provider']} -> {p['request']}  ({p['why']})")
            if not proposals:
                print(f"  [PROPOSE] {agent.id}: no proposal")

    agent.log.log("propose", agent.id, "proposed", n=len(proposals), proposals=proposals)
    agent.bus.broadcast(agent.id, "proposal", {"proposals": proposals}, phase="propose")
    return proposals


# =============================================================================
# Collect (bookkeeping only, no decision)
# =============================================================================

def collect_proposals(all_lists: list[list[dict]], log: EventLog | None = None) -> list[dict]:
    """Merge per-robot proposals. `mutual` = both robots of the pair proposed it."""
    merged: dict[tuple[str, str], dict] = {}
    for props in all_lists:
        for p in props:
            key = (p["request"], p["provider"])
            m = merged.setdefault(key, {"request": p["request"], "provider": p["provider"],
                                        "proposed_by": [], "why": {}})
            if p["by"] not in m["proposed_by"]:
                m["proposed_by"].append(p["by"])
                m["why"][p["by"]] = p["why"]
    out = []
    for m in merged.values():
        pair = {agent_of(m["request"]), agent_of(m["provider"])}
        m["mutual"] = pair <= set(m["proposed_by"])
        out.append(m)
    out.sort(key=lambda m: (not m["mutual"], m["request"], m["provider"]))
    if log:
        log.log("propose", "-", "collected", n=len(out), mutual=sum(m["mutual"] for m in out))
    return out
