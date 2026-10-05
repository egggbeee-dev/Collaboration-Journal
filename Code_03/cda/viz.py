"""Draw the scheduled dependency graph: x = time, y = robot. Works in Colab."""
from __future__ import annotations

import matplotlib.pyplot as plt

from .graph import PlanGraph
from .schemas import PROPOSED, SEQ

COLORS = {"LOCAL": "#9aa5b1", "ASK_HELP": "#e8a33d", "HELP": "#3d9be8",
          "RECEIVE": "#e8603d", "PASS": "#46b37a"}
EDGE_COLORS = {"SEQ": "#c0c0c0", "TRANSFER": "#46b37a", "HELP": "#3d9be8"}


def draw(g: PlanGraph, path: str | None = None, show: bool = True):
    ids = g.cfg.ids
    row = {a: len(ids) - 1 - i for i, a in enumerate(ids)}
    fig, ax = plt.subplots(figsize=(max(8, g.makespan() * 0.6 + 3), 1.3 * len(ids) + 1))
    pos = {}
    for nid, n in g.nodes.items():
        if n.status != "active":
            continue
        y = row[n.agent]
        if n.travel:                                   # travel part, hatched
            ax.barh(y, n.travel, left=n.t_start, height=0.5, color="white", hatch="///",
                    edgecolor="#888888", linewidth=0.4)
        x0 = n.t_start + n.travel
        w = n.t_end - x0
        if w > 0:
            ax.barh(y, w, left=x0, height=0.5, color=COLORS[n.type], edgecolor="black", linewidth=0.4,
                    alpha=0.55 if n.type == "ASK_HELP" else 1.0)
        else:                                          # instant (help already done): thin tick
            ax.plot([x0, x0], [y - 0.25, y + 0.25], color=COLORS[n.type], lw=2)
        if max(w, 0) >= 0.9:
            ax.text(x0 + 0.1, y, f"{n.type}\n{nid}", va="center", fontsize=6)
        pos[nid] = (n.t_start, n.t_end, y)
    for e in g.edges():
        if e.kind == SEQ or e.src not in pos or e.dst not in pos:
            continue
        _, x0, y0 = pos[e.src]
        _, x1, y1 = pos[e.dst]
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color=EDGE_COLORS[e.kind], lw=1.4,
                                    linestyle="--" if e.status == PROPOSED else "-"))
    ax.axvline(g.cfg.deadline_min, color="red", lw=1, ls=":")
    ax.set_yticks([row[a] for a in ids], ids)
    ax.set_xlabel("time (min)")
    ax.set_title(f"{g.cfg.task_id} — makespan {g.makespan()} / deadline {g.cfg.deadline_min}")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS.values()]
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="white", hatch="///", edgecolor="#888888"))
    ax.legend(handles, list(COLORS.keys()) + ["travel"], loc="upper center", fontsize=7, ncol=6,
              bbox_to_anchor=(0.5, -0.12), frameon=False)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    if show:
        plt.show()
    return fig
