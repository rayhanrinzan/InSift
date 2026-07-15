"""Track the application runtime version without mutating active imports."""

from __future__ import annotations

import builtins


RUNTIME_VERSION = 12


def ensure_runtime_current() -> None:
    """Record the running version; deployments and dev scripts restart cleanly."""

    setattr(builtins, "_flowsift_runtime_version", RUNTIME_VERSION)
