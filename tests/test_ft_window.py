"""Tests for ft_window. Run with: python -m pytest tests/ -q   (or: python tests/test_ft_window.py)"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from ft_window import (  # noqa: E402
    DEFAULT_CTX_PLUS_QUERY,
    HARM_THRESHOLD,
    WARN_THRESHOLD,
    advise,
    check,
    episode_context_rows,
    inference_context_rows,
    ratio,
    regime,
    safe_deployment_window,
)


# --------------------------------------------------------------- arithmetic
def test_episode_context_is_the_non_query_share():
    # 128 total with a 20% query share is the 102/26 episode used in the grid.
    assert episode_context_rows(10_000, 128, 0.2) == 102
    assert episode_context_rows(10_000, 1280, 0.2) == 1024


def test_episode_cannot_exceed_the_training_set():
    assert episode_context_rows(1_000, DEFAULT_CTX_PLUS_QUERY, 0.2) == 800


def test_inference_context_defaults_to_every_row():
    assert inference_context_rows(7_000, None) == 7_000
    assert inference_context_rows(7_000, 1_024) == 1_024
    assert inference_context_rows(500, 1_024) == 500  # capped by the data


@pytest.mark.parametrize("bad", [0, -1, 2.5e-3, None, "x"])
def test_bad_sizes_are_rejected(bad):
    with pytest.raises((ValueError, TypeError)):
        episode_context_rows(bad)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_bad_split_ratio_is_rejected(bad):
    with pytest.raises(ValueError):
        episode_context_rows(10_000, 1_280, bad)


# ------------------------------------------------------- the library default
def test_library_defaults_are_locked_at_1_25_below_the_cap():
    # The documented finding: for n <= 50,000 the defaults couple both lengths.
    # Tolerance is 1e-4, not 0: the episode is an integer number of rows, so
    # n * (1 - split) is rounded and the ratio lands on 1.25 only up to that.
    for n in (1_000, 10_000, 49_999):
        assert ratio(n) == pytest.approx(1.25, abs=1e-4)


def test_the_default_coupling_breaks_on_large_data():
    # Above the 50,000 cap the episode stops growing while the context keeps going.
    assert ratio(100_000) == pytest.approx(2.5, abs=1e-6)
    assert ratio(400_000) == pytest.approx(10.0, abs=1e-6)
    assert check(100_000).ok is True      # exactly at the conservative guard
    assert check(120_000).ok is False


def test_lowering_the_budget_for_memory_is_the_other_failure_mode():
    v = check(20_000, n_finetune_ctx_plus_query_samples=1_280)
    assert v.ratio == pytest.approx(20_000 / 1_024, abs=1e-6)
    assert v.regime == "harmful"
    assert v.ok is False
    assert "HARMFUL" in v.message


# --------------------------------------------------------------- the regimes
def test_regimes_follow_the_measured_bands():
    assert regime(0.05)[0] == "diminishing"
    assert regime(0.4)[0] == "optimal"
    assert regime(0.8)[0] == "optimal"
    assert regime(1.5)[0] == "good"
    assert regime(3.0)[0] == "marginal"
    assert regime(6.0)[0] == "harmful"
    assert regime(20.0)[0] == "harmful"


def test_the_sign_flips_where_the_measurements_say_it_does():
    assert regime(HARM_THRESHOLD - 0.1)[1] > 0
    assert regime(HARM_THRESHOLD + 0.1)[1] < 0


def test_regime_covers_every_positive_ratio():
    for r in (1e-6, 1e-3, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 1e3, 1e9):
        name, rel, dauc = regime(r)
        assert name in {"diminishing", "optimal", "good", "marginal", "harmful"}
        assert isinstance(rel, float) and isinstance(dauc, float)
    with pytest.raises(ValueError):
        regime(0.0)


# ---------------------------------------------------------------- the advice
def test_advice_targets_the_measured_optimum_when_data_allows():
    a = advise(n_train=100_000, deployment_context=1_024)
    assert a.ratio == pytest.approx(0.5, abs=0.02)
    assert a.memory_limited is False
    # The returned kwargs are meant to be splatted into the real classifier.
    assert set(a.kwargs) == {
        "n_finetune_ctx_plus_query_samples",
        "finetune_ctx_query_split_ratio",
        "n_inference_subsample_samples",
    }


def test_advice_is_clipped_by_the_training_set_and_says_so():
    # Deploying on the whole training set: the episode cannot be longer than it,
    # so 1.25 is a floor, not a choice.
    a = advise(n_train=6_000)
    assert a.ratio == pytest.approx(1.25, abs=1e-6)
    assert a.memory_limited is True
    assert a.n_finetune_ctx_plus_query_samples == 6_000


def test_advice_under_a_memory_limit_proposes_capping_the_context():
    a = advise(n_train=200_000, max_episode_rows=2_000)
    assert a.ratio > HARM_THRESHOLD
    assert "n_inference_subsample_samples" in a.message
    assert a.memory_limited is True


def test_advice_never_proposes_a_harmful_ratio_when_it_has_the_room():
    for n in (2_000, 20_000, 200_000):
        for ctx in (256, 1_024, 4_096):
            if ctx > n:
                continue
            a = advise(n_train=n, deployment_context=ctx)
            if not a.memory_limited:
                assert a.ratio <= WARN_THRESHOLD, (n, ctx, a.ratio)


def test_advice_round_trips_through_check():
    a = advise(n_train=50_000, deployment_context=2_048)
    v = check(50_000, **a.kwargs)
    assert v.ratio == pytest.approx(a.ratio, rel=1e-9)
    assert v.ok is True


# ---------------------------------------------------------------- the window
def test_safe_window_brackets_the_episode_length():
    low, high = safe_deployment_window(1_024)
    assert low < 1_024 < high
    assert high == int(round(1_024 * WARN_THRESHOLD))
    assert check(10_000, n_finetune_ctx_plus_query_samples=1_280,
                 n_inference_subsample_samples=high).ok is True
    assert check(10_000, n_finetune_ctx_plus_query_samples=1_280,
                 n_inference_subsample_samples=high * 3).ok is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
