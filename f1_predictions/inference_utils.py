import re
from typing import Any

import numpy as np


def normalize_event_name(event: str) -> str:
    if not isinstance(event, str):
        return str(event)
    return event.replace("S\u00e3o", "Sao")


def canonical_event_name(event: str) -> str:
    ev = normalize_event_name(str(event or "").strip().replace("_", " "))
    return re.sub(r"\s+", " ", ev).strip()


def normalize_driver_code(val: Any) -> str:
    code = str(val or "").strip().upper()
    if len(code) == 3 and code.isalpha():
        return code
    return ""


def safe_float(v, default=0.0) -> float:
    try:
        x = float(v)
        if np.isfinite(x):
            return x
    except (TypeError, ValueError):
        pass
    return float(default)

