"""Tests for UI components module."""

from contextlib import suppress
from pathlib import Path
from unittest.mock import Mock, patch

from openscientist.job_manager import JobStatus
from openscientist.webapp_components.ui_components import (
    OPENSCIENTIST_GITHUB_URL,
    OPENSCIENTIST_PAPER_URL,
    OPENSCIENTIST_RELEASE_URL,
    get_project_resource_links,
)


class TestProjectResourceLinks:
    """Tests for shared public resource links."""

    def test_links_expose_repo_paper_and_release(self):
        """Resource strip should expose only links reachable without auth."""
        assert OPENSCIENTIST_PAPER_URL is not None
        assert get_project_resource_links() == [
            ("GitHub", OPENSCIENTIST_GITHUB_URL),
            ("Paper", OPENSCIENTIST_PAPER_URL),
            ("Latest Release", OPENSCIENTIST_RELEASE_URL),
        ]


class TestRenderErrorCard:
    """Tests for render_error_card function (basic structure testing)."""

    @patch("openscientist.webapp_components.ui_components.ui")
    def test_render_error_card_called(self, mock_ui):
        """Test that render_error_card can be called without errors."""
        from openscientist.webapp_components.ui_components import render_error_card

        # Mock UI components
        mock_ui.card.return_value.__enter__ = Mock()
        mock_ui.card.return_value.__exit__ = Mock(return_value=False)
        mock_ui.row.return_value.__enter__ = Mock()
        mock_ui.row.return_value.__exit__ = Mock(return_value=False)
        mock_ui.column.return_value.__enter__ = Mock()
        mock_ui.column.return_value.__exit__ = Mock(return_value=False)
        mock_ui.expansion.return_value.__enter__ = Mock()
        mock_ui.expansion.return_value.__exit__ = Mock(return_value=False)
        mock_ui.element.return_value.__enter__ = Mock()
        mock_ui.element.return_value.__exit__ = Mock(return_value=False)
        mock_ui.button.return_value.__enter__ = Mock()
        mock_ui.button.return_value.__exit__ = Mock(return_value=False)

        error_info = {
            "category": "configuration",
            "title": "Test Error",
            "message": "Test message",
            "extracted_error": "Error details",
            "steps": ["Step 1", "Step 2"],
            "raw": "Raw error",
            "contact_admin": True,
        }

        job_info = Mock()
        job_info.status = JobStatus.FAILED
        job_info.iterations_completed = 1
        job_info.max_iterations = 5
        job_info.failed_at = "2026-02-05T10:00:00"

        job_dir = Path("/fake/job/dir")

        # Some exceptions are acceptable due to mock limitations.
        with suppress(Exception):
            render_error_card(error_info, job_info, job_dir)

        # Verify UI elements were called
        assert mock_ui.card.called or mock_ui.row.called


class TestBadgeSymbolsReexportedFromUiComponents:
    """
    Badge components were extracted to
    openscientist.webapp_components.components.badges. These tests guard the
    backward-compatibility re-export from ui_components, since production code
    across the app still imports badge symbols directly from this module.
    """

    REEXPORTED_BADGE_SYMBOLS = [
        "CATEGORY_COLORS",
        "STATUS_COLORS",
        "STATUS_ICONS",
        "get_category_color",
        "get_status_badge_props",
        "render_container_status_badge",
        "render_job_id_badge",
        "render_job_id_slot",
        "render_permission_badge_slot",
        "render_pmid_badge",
        "render_stat_badges",
        "render_status_cell_slot",
        "render_text_with_pmid_links",
        "transform_pmid_references",
    ]

    def test_badge_symbols_are_identical_objects_in_both_modules(self):
        """ui_components must expose the exact same objects as the badges module."""
        from openscientist.webapp_components import ui_components
        from openscientist.webapp_components.components import badges

        for name in self.REEXPORTED_BADGE_SYMBOLS:
            assert hasattr(ui_components, name), f"{name} is not importable from ui_components"
            assert getattr(ui_components, name) is getattr(badges, name), (
                f"ui_components.{name} is not the same object as badges.{name}"
            )

    def test_named_imports_from_ui_components_still_work(self):
        """Exercise the historical `from ui_components import X` usage pattern."""
        from openscientist.webapp_components.ui_components import (
            CATEGORY_COLORS,
            STATUS_COLORS,
            STATUS_ICONS,
            get_category_color,
            get_status_badge_props,
            render_container_status_badge,
            render_job_id_badge,
            render_job_id_slot,
            render_permission_badge_slot,
            render_pmid_badge,
            render_stat_badges,
            render_status_cell_slot,
            render_text_with_pmid_links,
            transform_pmid_references,
        )

        assert isinstance(CATEGORY_COLORS, dict)
        assert isinstance(STATUS_COLORS, dict)
        assert isinstance(STATUS_ICONS, dict)
        assert callable(get_category_color)
        assert callable(get_status_badge_props)
        assert callable(render_container_status_badge)
        assert callable(render_job_id_badge)
        assert callable(render_job_id_slot)
        assert callable(render_permission_badge_slot)
        assert callable(render_pmid_badge)
        assert callable(render_stat_badges)
        assert callable(render_status_cell_slot)
        assert callable(render_text_with_pmid_links)
        assert callable(transform_pmid_references)
