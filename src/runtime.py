"""Refresh safe application modules after Streamlit hot deployments."""

from __future__ import annotations

import builtins
import importlib


RUNTIME_VERSION = 8


def ensure_runtime_current() -> None:
    """Reload service modules once when a newer app runtime reaches the process."""

    active_version = int(getattr(builtins, "_flowsift_runtime_version", 0))
    if active_version >= RUNTIME_VERSION:
        return

    module_names = (
        "src.database.repositories",
        "src.research.public_discussion_search",
        "src.ingestion.web",
        "src.services.discovery_service",
        "src.services.problem_scout_service",
        "src.services.opportunity_service",
        "src.ui.data",
    )
    for module_name in module_names:
        module = importlib.import_module(module_name)
        importlib.reload(module)
    setattr(builtins, "_flowsift_runtime_version", RUNTIME_VERSION)
