import inspect
from pathlib import Path

import numpy as np


def _preview_sequence(values, limit):
    if len(values) <= limit:
        return repr(values)

    head_count = max(1, limit // 2)
    tail_count = max(1, limit - head_count)
    head = repr(values[:head_count])
    tail = repr(values[-tail_count:])
    return f"{head} ... {tail}"


def _summarize_value(value, limit):
    if isinstance(value, np.ndarray):
        return (
            f"ndarray(shape={value.shape}, "
            f"min={value.min():.6f}, max={value.max():.6f}, "
            f"preview={np.array2string(value.reshape(-1)[:limit], precision=6, threshold=limit)})"
        )

    if isinstance(value, bytes):
        return (
            f"bytes(len={len(value)}, "
            f"preview={_preview_sequence(value[:limit], limit)})"
        )

    if isinstance(value, (list, tuple)):
        return (
            f"{type(value).__name__}(len={len(value)}, "
            f"preview={_preview_sequence(value, limit)})"
        )

    return repr(value)


def debug_value(name, value, *, limit=12):
    caller = inspect.currentframe().f_back
    file_name = Path(caller.f_code.co_filename).name
    line_number = caller.f_lineno
    summary = _summarize_value(value, limit)
    print(f"[DEBUG {file_name} {name} = {summary}")
