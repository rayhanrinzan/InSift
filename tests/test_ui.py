"""Tests for Phase 8 UI helpers."""

import pytest

from src.ui.components import paginate_items, status_badge_html


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
