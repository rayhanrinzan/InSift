"""Reusable Streamlit components and layout helpers."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from math import ceil
from typing import Generic, Optional, Sequence, TypeVar

import streamlit as st

from src.config import Settings, redacted_database_url
from src.ui.formatting import format_score


T = TypeVar("T")


GLOBAL_STYLES = """
<style>
    :root {
        --insift-ink: #172126;
        --insift-muted: #5e6b70;
        --insift-line: #dce2df;
        --insift-panel: #ffffff;
        --insift-canvas: #f6f8f6;
        --insift-teal: #0f766e;
        --insift-amber: #a16207;
        --insift-red: #b42318;
    }
    [data-testid="stAppViewContainer"] {
        background: var(--insift-canvas);
    }
    [data-testid="stHeader"] {
        background: rgba(246, 248, 246, 0.92);
    }
    [data-testid="stSidebarNav"] {
        display: none;
    }
    [data-testid="stSidebar"] {
        border-right: 1px solid var(--insift-line);
    }
    .block-container {
        max-width: 1480px;
        padding-top: 1.75rem;
        padding-bottom: 3rem;
    }
    h1, h2, h3, p, label, button, input, textarea {
        letter-spacing: 0 !important;
    }
    h1 {
        color: var(--insift-ink);
        font-size: 2.6rem !important;
        line-height: 1.08 !important;
    }
    h2 {
        color: var(--insift-ink);
    }
    .insift-brand {
        color: var(--insift-ink);
        font-size: 1.35rem;
        font-weight: 750;
        margin-bottom: 0.1rem;
    }
    .insift-brand-note, .insift-eyebrow {
        color: var(--insift-muted);
        font-size: 0.78rem;
        font-weight: 650;
        text-transform: uppercase;
        letter-spacing: 0.08rem !important;
    }
    .insift-page-note {
        color: var(--insift-muted);
        font-size: 1rem;
        margin-top: -0.5rem;
        margin-bottom: 1.35rem;
        max-width: 56rem;
    }
    .insift-badge {
        border: 1px solid var(--insift-line);
        border-radius: 999px;
        display: inline-block;
        font-size: 0.75rem;
        font-weight: 700;
        line-height: 1;
        padding: 0.34rem 0.52rem;
        white-space: nowrap;
    }
    .insift-badge[data-tone="good"] {
        background: #e8f5f1;
        border-color: #9bcfc3;
        color: #075f57;
    }
    .insift-badge[data-tone="warn"] {
        background: #fff7df;
        border-color: #e8cb78;
        color: #7a4b00;
    }
    .insift-badge[data-tone="risk"] {
        background: #fff0ed;
        border-color: #e8aaa1;
        color: #912018;
    }
    .insift-badge[data-tone="neutral"] {
        background: #eef1ef;
        color: #455157;
    }
    [data-testid="stMetric"] {
        background: var(--insift-panel);
        border: 1px solid var(--insift-line);
        border-radius: 6px;
        min-height: 7rem;
        padding: 1rem 1.1rem;
    }
    [data-testid="stMetricValue"] {
        color: var(--insift-ink);
        font-size: 1.8rem;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--insift-panel);
        border-color: var(--insift-line) !important;
        border-radius: 6px !important;
    }
    [data-testid="stForm"], [data-testid="stExpander"] {
        background: var(--insift-panel);
        border-color: var(--insift-line) !important;
        border-radius: 6px !important;
    }
    .stButton > button, .stLinkButton > a, [data-testid="stFormSubmitButton"] button {
        border-radius: 6px !important;
        min-height: 2.5rem;
    }
    [data-baseweb="tab-list"] {
        gap: 0.25rem;
    }
    [data-baseweb="tab"] {
        border-radius: 4px 4px 0 0;
    }
    @media (max-width: 760px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
            padding-top: 1rem;
        }
        [data-testid="stMetric"] {
            min-height: 5.8rem;
        }
        h1 {
            font-size: 2.15rem !important;
        }
    }
</style>
"""


@dataclass(frozen=True)
class PageSlice(Generic[T]):
    """A bounded slice of a larger result set."""

    items: tuple[T, ...]
    page: int
    page_size: int
    total_items: int
    total_pages: int
    first_item: int
    last_item: int


def configure_page(title: str, settings: Settings) -> None:
    """Apply shared page metadata, styling, navigation, and runtime context."""

    page_title = "InSift" if title == "Overview" else f"{title} | InSift"
    st.set_page_config(
        page_title=page_title,
        page_icon="IS",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(GLOBAL_STYLES, unsafe_allow_html=True)
    with st.sidebar:
        st.markdown('<div class="insift-brand">InSift</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="insift-brand-note">Opportunity intelligence</div>',
            unsafe_allow_html=True,
        )
        st.divider()
        _render_navigation()
        st.divider()
        if settings.demo_mode:
            mode, tone = "Demo data", "neutral"
        elif settings.live_ready:
            mode, tone = "Live providers", "good"
        else:
            mode, tone = "Setup required", "warn"
        st.markdown(status_badge_html(mode, tone), unsafe_allow_html=True)
        st.caption(redacted_database_url(settings.database_url))


def _render_navigation() -> None:
    """Render multipage links with a direct-page test fallback."""

    pages = [
        ("streamlit_app.py", "Overview", "/"),
        ("pages/1_Discover.py", "Discover", "/Discover"),
        ("pages/2_Opportunities.py", "Opportunities", "/Opportunities"),
        (
            "pages/3_Opportunity_Details.py",
            "Opportunity details",
            "/Opportunity_Details",
        ),
        ("pages/4_Settings.py", "Settings", "/Settings"),
    ]
    try:
        for page, label, _ in pages:
            st.page_link(page, label=label, use_container_width=True)
    except KeyError:
        for _, label, route in pages:
            st.markdown(f"[{label}]({route})")


def render_page_link(
    page: str,
    *,
    label: str,
    route: str,
    use_container_width: bool = False,
) -> None:
    """Render an internal page link with a direct-page execution fallback."""

    try:
        st.page_link(
            page,
            label=label,
            use_container_width=use_container_width,
        )
    except KeyError:
        st.markdown(f"[{escape(label)}]({route})")


def page_header(title: str, note: str, *, eyebrow: str = "InSift") -> None:
    """Render a compact page heading with stable spacing."""

    st.markdown(
        f'<div class="insift-eyebrow">{escape(eyebrow)}</div>',
        unsafe_allow_html=True,
    )
    st.title(title)
    st.markdown(
        f'<div class="insift-page-note">{escape(note)}</div>',
        unsafe_allow_html=True,
    )


def score_metric(label: str, score: Optional[float]) -> None:
    """Render a score metric with a consistent empty state."""

    st.metric(label, format_score(score))


def status_badge_html(label: str, tone: str = "neutral") -> str:
    """Return a small escaped status badge for use in Streamlit markdown."""

    valid_tone = tone if tone in {"good", "warn", "risk", "neutral"} else "neutral"
    return (
        f'<span class="insift-badge" data-tone="{valid_tone}">'
        f"{escape(label)}</span>"
    )


def score_tone(score: Optional[float]) -> str:
    """Map a score to a restrained visual tone."""

    if score is None:
        return "neutral"
    if score >= 70:
        return "good"
    if score >= 45:
        return "warn"
    return "risk"


def paginate_items(
    items: Sequence[T],
    *,
    page: int,
    page_size: int,
) -> PageSlice[T]:
    """Return a clamped page without mutating the source collection."""

    if page_size < 1:
        raise ValueError("page_size must be at least 1")
    total_items = len(items)
    total_pages = max(1, ceil(total_items / page_size))
    safe_page = min(max(1, page), total_pages)
    start = (safe_page - 1) * page_size
    end = min(start + page_size, total_items)
    return PageSlice(
        items=tuple(items[start:end]),
        page=safe_page,
        page_size=page_size,
        total_items=total_items,
        total_pages=total_pages,
        first_item=start + 1 if total_items else 0,
        last_item=end,
    )


def page_size_control(key: str, *, default: int = 10) -> int:
    """Render a compact page-size selector."""

    options = [5, 10, 25, 50]
    default_index = options.index(default) if default in options else 1
    return int(
        st.selectbox(
            "Rows per page",
            options,
            index=default_index,
            key=f"{key}-page-size",
        )
    )


def render_pagination(page_slice: PageSlice[object], key: str) -> None:
    """Render previous/next controls for an already calculated page."""

    state_key = f"{key}-page"
    st.session_state[state_key] = page_slice.page
    previous, summary, next_column = st.columns([1, 3, 1])
    if previous.button(
        "Previous",
        key=f"{key}-previous",
        disabled=page_slice.page <= 1,
        use_container_width=True,
    ):
        st.session_state[state_key] = page_slice.page - 1
        st.rerun()
    summary.caption(
        f"{page_slice.first_item}-{page_slice.last_item} of {page_slice.total_items} "
        f"| Page {page_slice.page} of {page_slice.total_pages}"
    )
    if next_column.button(
        "Next",
        key=f"{key}-next",
        disabled=page_slice.page >= page_slice.total_pages,
        use_container_width=True,
    ):
        st.session_state[state_key] = page_slice.page + 1
        st.rerun()


def render_database_error(context: str, settings: Settings) -> None:
    """Show a safe, actionable database error without exposing credentials."""

    st.error(f"{context} is unavailable because InSift could not reach its database.")
    command = "python scripts/initialize_database.py"
    if settings.demo_mode:
        command += "\npython scripts/seed_demo_data.py"
    st.code(command, language="bash")
    st.caption(f"Configured database: {redacted_database_url(settings.database_url)}")


def set_flash(message: str, tone: str = "success") -> None:
    """Store one message that survives a Streamlit rerun."""

    st.session_state["insift-flash"] = {"message": message, "tone": tone}


def render_flash() -> None:
    """Render and clear a previously stored message."""

    payload = st.session_state.pop("insift-flash", None)
    if not payload:
        return
    renderer = {
        "success": st.success,
        "warning": st.warning,
        "error": st.error,
        "info": st.info,
    }.get(payload.get("tone"), st.info)
    renderer(payload.get("message", "Update complete."))
