from __future__ import annotations

import time


class EventLog:
    """Ordered log of what each robot / stage did. Saved as events.json."""

    def __init__(self, verbose: bool = True):
        self.events: list[dict] = []
        self.verbose = verbose
        self.t0 = time.time()

    def log(self, stage: str, who: str, what: str, **data) -> None:
        e = {"t": round(time.time() - self.t0, 2), "stage": stage, "who": who, "what": what, **data}
        self.events.append(e)
        if self.verbose:
            extra = " ".join(f"{k}={v}" for k, v in data.items() if k not in ("detail",))
            print(f"[{e['t']:6.2f}s] {stage:<8} {who:<4} {what} {extra}".rstrip())
