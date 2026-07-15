"""Tests for Phase 8 UI helpers."""

import pytest

from src.ui.components import fact_block_html, paginate_items, score_bar_html
from src.ui.components import status_badge_html
from src.ui.data import get_ui_session_factory


def test_pagination_returns_requested_slice() -> None:
    page = paginate_items(list(range(23)), page=3, page_size=10)

    assert page.items == (20, 21, 22)
    assert page.page == 3
    assert page.total_pages == 3
    assert page.first_item == 21
    assert page.last_item == 23


def test_pagination_clamps_out_of_range_page() -> None:
    page = paginate_items(["a", "b", "c"], page=99, page_size=2)

    assert page.page == 2
    assert page.items == ("c",)


def test_pagination_handles_empty_collection() -> None:
    page = paginate_items([], page=1, page_size=10)

    assert page.items == ()
    assert page.total_pages == 1
    assert page.first_item == 0
    assert page.last_item == 0


def test_pagination_rejects_invalid_page_size() -> None:
    with pytest.raises(ValueError, match="page_size"):
        paginate_items([1], page=1, page_size=0)


def test_status_badge_escapes_user_controlled_text() -> None:
    badge = status_badge_html("<script>alert(1)</script>", "good")

    assert "<script>" not in badge
    assert "&lt;script&gt;" in badge


def test_score_bar_clamps_width_and_escapes_label() -> None:
    score = score_bar_html("<Problem>", 140)

    assert "width:100.0%" in score
    assert "&lt;Problem&gt;" in score
    assert "<Problem>" not in score


def test_fact_block_escapes_values_and_uses_fallback() -> None:
    fact = fact_block_html("Target", None, "<Unknown>")

    assert "&lt;Unknown&gt;" in fact
    assert "<Unknown>" not in fact


def test_ui_session_factory_initializes_a_fresh_database(tmp_path) -> None:
    get_ui_session_factory.clear()
    database_url = f"sqlite:///{tmp_path / 'fresh.db'}"

    SessionFactory = get_ui_session_factory(database_url)
    with SessionFactory() as session:
        table_names = set(
            session.get_bind().dialect.get_table_names(session.connection())
        )

    assert "evidence_items" in table_names
    assert "opportunity_clusters" in table_names
    get_ui_session_factory.clear()
