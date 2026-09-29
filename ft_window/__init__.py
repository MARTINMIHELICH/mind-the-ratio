"""ft_window - pick a TabPFN fine-tuning episode length that matches how you deploy.

Fine-tuning a tabular in-context model requires choosing *two* lengths:

  * how many rows go into one training episode, and
  * how many rows go into the context at prediction time.

Their **ratio** decides whether fine-tuning helps. Measured on 24 binary OpenML
tasks across three independent runs (see ../data and ../notebooks/02_analysis.ipynb):

    ratio = inference_context_rows / episode_context_rows

    ratio 0.5 - 2    fine-tuning helps         65/70 dataset-runs improve, +3.6% log-loss
    ratio > 4        fine-tuning hurts         11/60 improve,              -4.1% log-loss

This module has no dependencies outside the standard library, and it does not
import or call tabpfn. It reads the same arguments you already pass to
``FinetunedTabPFNClassifier`` and tells you which regime they put you in.

    >>> from ft_window import check
    >>> print(check(n_train=200_000).message)      # doctest: +SKIP

Parameter names follow tabpfn 9.0.0 exactly, so a call can be copied verbatim
from your own code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

__all__ = [
    "DEFAULT_CTX_PLUS_QUERY",
    "DEFAULT_SPLIT_RATIO",
    "BANDS",
    "Verdict",
    "Advice",
    "episode_context_rows",
    "inference_context_rows",
    "ratio",
    "regime",
    "check",
    "advise",
    "safe_deployment_window",
]

# --- tabpfn 9.0.0 defaults, read off the installed package -------------------
# Reproduce with notebooks/01_inspect_api.ipynb, which prints file and line.
DEFAULT_CTX_PLUS_QUERY = 50_000     # n_finetune_ctx_plus_query_samples (a CAP)
DEFAULT_SPLIT_RATIO = 0.2           # finetune_ctx_query_split_ratio
DEFAULT_INFERENCE_SUBSAMPLE = None  # n_inference_subsample_samples (None = all rows)

# --- measured effect per ratio band ------------------------------------------
# Mean relative log-loss gain (%) and mean AUC delta, averaged per dataset then
# across datasets, at 200 fine-tuning steps, pooled over three runs.
# Regenerate with notebooks/02_analysis.ipynb.
BANDS = (
    #  low,  high,  name,          rel. log-loss %, delta AUC
    (0.00, 0.25, "diminishing", +2.1, +0.0067),
    (0.25, 0.50, "optimal", +4.7, +0.0114),
    (0.50, 1.00, "optimal", +4.4, +0.0108),
    (1.00, 2.00, "good", +2.2, +0.0062),
    (2.00, 4.00, "marginal", +1.0, +0.0037),
    (4.00, 8.00, "harmful", -1.9, +0.0004),
    (8.00, float("inf"), "harmful", -5.9, -0.0024),
)

#: Beyond this ratio the measured mean effect on log-loss is negative.
HARM_THRESHOLD = 4.0

#: Conservative guard. Accuracy of the "fine-tune iff ratio <= t" rule is flat
#: between t = 2.5 and t = 5 (75-78% out of sample, against 68% for
#: "always fine-tune"), so the threshold is a plateau, not a point. We take the
#: low end: inside the plateau it is the value that wrongly recommends
#: fine-tuning least often.
WARN_THRESHOLD = 2.5

#: Target ratio for advise(). The measured optimum is broad - log-loss peaks
#: near 0.3-0.5 and AUC plateaus from 0.08 to 0.3 - so any value in [0.5, 1]
#: sits in well-measured territory. We aim at 0.5 and accept up to 1.
TARGET_RATIO = 0.5
TARGET_RATIO_MAX = 1.0


def _positive_int(value: object, name: str) -> int:
    try:
        out = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise TypeError(f"{name} must be an integer, got {value!r}") from None
    if out <= 0:
        raise ValueError(f"{name} must be positive, got {out}")
    return out


def _split_ratio(value: float) -> float:
    value = float(value)
    if not 0.0 < value < 1.0:
        raise ValueError(
            f"finetune_ctx_query_split_ratio must be in (0, 1), got {value}"
        )
    return value


def episode_context_rows(
    n_train: int,
    n_finetune_ctx_plus_query_samples: int = DEFAULT_CTX_PLUS_QUERY,
    finetune_ctx_query_split_ratio: float = DEFAULT_SPLIT_RATIO,
) -> int:
    """Rows the model actually sees as context in one fine-tuning episode.

    ``n_finetune_ctx_plus_query_samples`` is a *cap* on context plus query, and
    it cannot exceed the training set. The query share is carved out of it, so
    the context is the remainder.
    """
    n_train = _positive_int(n_train, "n_train")
    cap = _positive_int(
        n_finetune_ctx_plus_query_samples, "n_finetune_ctx_plus_query_samples"
    )
    split = _split_ratio(finetune_ctx_query_split_ratio)
    episode = min(cap, n_train)
    return max(1, int(round(episode * (1.0 - split))))


def inference_context_rows(
    n_train: int, n_inference_subsample_samples: Optional[int] = None
) -> int:
    """Rows given to the model as context at prediction time.

    ``None`` - the library default - means the whole training set is used.
    """
    n_train = _positive_int(n_train, "n_train")
    if n_inference_subsample_samples is None:
        return n_train
    return min(
        n_train,
        _positive_int(n_inference_subsample_samples, "n_inference_subsample_samples"),
    )


def ratio(
    n_train: int,
    n_finetune_ctx_plus_query_samples: int = DEFAULT_CTX_PLUS_QUERY,
    finetune_ctx_query_split_ratio: float = DEFAULT_SPLIT_RATIO,
    n_inference_subsample_samples: Optional[int] = None,
) -> float:
    """inference context rows divided by fine-tuning episode context rows."""
    return inference_context_rows(n_train, n_inference_subsample_samples) / (
        episode_context_rows(
            n_train,
            n_finetune_ctx_plus_query_samples,
            finetune_ctx_query_split_ratio,
        )
    )


def regime(r: float) -> tuple[str, float, float]:
    """Return (name, mean relative log-loss %, mean AUC delta) for a ratio."""
    r = float(r)
    if r <= 0:
        raise ValueError(f"ratio must be positive, got {r}")
    for low, high, name, rel, dauc in BANDS:
        if low < r <= high or (low == 0.0 and r <= high):
            return name, rel, dauc
    raise AssertionError("unreachable: BANDS must cover (0, inf)")


@dataclass(frozen=True)
class Verdict:
    """What a given configuration amounts to."""

    ratio: float
    regime: str
    episode_context_rows: int
    inference_context_rows: int
    expected_rel_logloss_pct: float
    expected_auc_delta: float
    ok: bool
    message: str

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.message


def check(
    n_train: int,
    n_finetune_ctx_plus_query_samples: int = DEFAULT_CTX_PLUS_QUERY,
    finetune_ctx_query_split_ratio: float = DEFAULT_SPLIT_RATIO,
    n_inference_subsample_samples: Optional[int] = None,
) -> Verdict:
    """Diagnose a fine-tuning configuration before you run it.

    Takes the same arguments you pass to ``FinetunedTabPFNClassifier``, plus the
    size of your training set.
    """
    ctx_ep = episode_context_rows(
        n_train, n_finetune_ctx_plus_query_samples, finetune_ctx_query_split_ratio
    )
    ctx_inf = inference_context_rows(n_train, n_inference_subsample_samples)
    r = ctx_inf / ctx_ep
    name, rel, dauc = regime(r)
    ok = r <= WARN_THRESHOLD

    head = (
        f"ratio {r:.2f}  (inference context {ctx_inf:,} rows / "
        f"episode context {ctx_ep:,} rows)"
    )
    if r > HARM_THRESHOLD:
        body = (
            f"HARMFUL. Episodes are {r:.0f}x shorter than the context you deploy "
            f"with. Measured mean effect in this band: {rel:+.1f}% relative "
            f"log-loss. Raise n_finetune_ctx_plus_query_samples, or cap the "
            f"inference context with n_inference_subsample_samples, or do not "
            f"fine-tune at all."
        )
    elif r > WARN_THRESHOLD:
        body = (
            f"MARGINAL. Measured mean effect: {rel:+.1f}% relative log-loss, and "
            f"the sign flips beyond ratio {HARM_THRESHOLD:g}. Prefer longer "
            f"episodes if memory allows."
        )
    elif r < 0.1:
        body = (
            f"WASTEFUL but safe. Episodes are much longer than the context you "
            f"deploy with. The benefit decays toward zero without turning "
            f"negative; you are paying for length you will not use. Shorter "
            f"episodes would be cheaper and measure slightly better."
        )
    else:
        body = (
            f"GOOD. This is the well-measured band: {rel:+.1f}% relative "
            f"log-loss, {dauc:+.4f} AUC on 24 OpenML binary tasks."
        )
    return Verdict(
        ratio=r,
        regime=name,
        episode_context_rows=ctx_ep,
        inference_context_rows=ctx_inf,
        expected_rel_logloss_pct=rel,
        expected_auc_delta=dauc,
        ok=ok,
        message=f"{head}\n{body}",
    )


@dataclass(frozen=True)
class Advice:
    """A configuration to pass to ``FinetunedTabPFNClassifier``."""

    n_finetune_ctx_plus_query_samples: int
    finetune_ctx_query_split_ratio: float
    n_inference_subsample_samples: Optional[int]
    episode_context_rows: int
    inference_context_rows: int
    ratio: float
    memory_limited: bool
    message: str
    kwargs: dict = field(default_factory=dict)

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.message


def advise(
    n_train: int,
    deployment_context: Optional[int] = None,
    max_episode_rows: Optional[int] = None,
    finetune_ctx_query_split_ratio: float = DEFAULT_SPLIT_RATIO,
) -> Advice:
    """Choose an episode length for the context length you will deploy with.

    Parameters
    ----------
    n_train:
        Rows in your training set.
    deployment_context:
        Rows you will give the model as context at prediction time. ``None``
        means the whole training set, which is the library default and usually
        the best choice for absolute accuracy.
    max_episode_rows:
        Largest episode (context + query) your GPU can hold, if you know it.
        This is the constraint that most often forces a bad ratio.
    """
    n_train = _positive_int(n_train, "n_train")
    split = _split_ratio(finetune_ctx_query_split_ratio)
    ctx_inf = inference_context_rows(n_train, deployment_context)

    # Aim at the measured optimum, then clip to what the data and the GPU allow.
    wanted_ctx = ctx_inf / TARGET_RATIO
    wanted_total = wanted_ctx / (1.0 - split)
    hard_cap = float(n_train if max_episode_rows is None else min(n_train, max_episode_rows))
    total = int(round(min(wanted_total, hard_cap)))
    total = max(total, 2)
    memory_limited = total < int(round(wanted_total))

    ctx_ep = episode_context_rows(n_train, total, split)
    r = ctx_inf / ctx_ep

    if r <= TARGET_RATIO_MAX:
        msg = (
            f"Use n_finetune_ctx_plus_query_samples={total:,}. Episode context "
            f"{ctx_ep:,} rows against a deployment context of {ctx_inf:,} rows "
            f"gives ratio {r:.2f}, inside the band where 65 of 70 dataset-runs "
            f"improved."
        )
    elif r <= WARN_THRESHOLD:
        msg = (
            f"Use n_finetune_ctx_plus_query_samples={total:,}, the largest your "
            f"limit allows. Ratio {r:.2f} is still acceptable, but below the "
            f"measured optimum: the whole training set is shorter than "
            f"{TARGET_RATIO_MAX:g}x your deployment context."
        )
    else:
        msg = (
            f"No safe configuration at this deployment context. The longest "
            f"episode you allow gives ratio {r:.2f}, past the point where "
            f"fine-tuning starts to hurt. Either raise max_episode_rows, or cap "
            f"the inference context: n_inference_subsample_samples="
            f"{int(ctx_ep * WARN_THRESHOLD):,} would bring the ratio back to "
            f"{WARN_THRESHOLD:g}. Otherwise do not fine-tune."
        )

    return Advice(
        n_finetune_ctx_plus_query_samples=total,
        finetune_ctx_query_split_ratio=split,
        n_inference_subsample_samples=deployment_context,
        episode_context_rows=ctx_ep,
        inference_context_rows=ctx_inf,
        ratio=r,
        memory_limited=memory_limited,
        message=msg,
        kwargs={
            "n_finetune_ctx_plus_query_samples": total,
            "finetune_ctx_query_split_ratio": split,
            "n_inference_subsample_samples": deployment_context,
        },
    )


def safe_deployment_window(
    episode_context: int, threshold: float = WARN_THRESHOLD
) -> tuple[int, int]:
    """Deployment context lengths that keep a given episode length in range.

    Returns ``(low, high)``. Below ``low`` fine-tuning still works but wastes
    compute; above ``high`` it degrades the model.
    """
    ctx = _positive_int(episode_context, "episode_context")
    return max(1, int(round(ctx * 0.1))), int(round(ctx * float(threshold)))
