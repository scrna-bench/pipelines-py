"""Phase timings as obkit structured events (<dir>/obkit-events.jsonl).

    init(args.outputs["methods_events"])
    with phase("pca"):
        ...
    with phase("leiden") as attrs:
        ...
        attrs.update(num_runs=n, resolution=res)   # goes on the end event
"""

import os
from contextlib import contextmanager

from obkit.logger import emit, init_logger


def init(events_path):
    if os.path.basename(events_path) != "obkit-events.jsonl":
        raise ValueError(f"the events output must be named obkit-events.jsonl, got {events_path}")
    init_logger(os.path.dirname(os.path.abspath(events_path)))


@contextmanager
def phase(name, **attrs):
    emit(name, "start")
    try:
        yield attrs
    except BaseException:
        attrs["error"] = True
        raise
    finally:
        emit(name, "end", attrs or None)
