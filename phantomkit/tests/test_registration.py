"""Unit tests for phantomkit.registration."""

from phantomkit.registration import ParseMrStatsStdout

import pytest


# ── ParseMrStatsStdout ────────────────────────────────────────────────────────


def test_parse_mrstats_stdout_multiple_values() -> None:
    out = ParseMrStatsStdout(stdout="1.5 2.3 0.8")()
    assert out.out == pytest.approx([1.5, 2.3, 0.8])


def test_parse_mrstats_stdout_single_value() -> None:
    out = ParseMrStatsStdout(stdout="42.0")()
    assert out.out == pytest.approx([42.0])
