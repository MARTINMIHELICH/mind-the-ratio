# Mind the ratio

**Fine-tuning TabPFN tunes it to a context length.** What decides whether it helps is
not how much data you fine-tune on, but how the length of a training episode compares
with the context the model will see when you deploy it. Fine-tune on episodes much
shorter than that context and the model comes out **worse than if you had not
fine-tuned at all**.

The compact way to write the comparison is a ratio:

```
ratio = rows in the inference context / rows in a fine-tuning episode
```

It is a summary, not a law, and §"What the ratio does and does not capture" below
says exactly where it holds and where it breaks.

Everything below comes from 24 binary OpenML tasks, three independent runs, 200
gradient steps. **Every number is reproducible in thirty seconds with no GPU** —
the raw measurements are in `data/`, and `notebooks/02_analysis.ipynb` recomputes
the lot:

```bash
pip install -r requirements.txt
jupyter nbconvert --execute --to notebook --inplace notebooks/02_analysis.ipynb
```

---

## The shape

![AUC against the ratio](figures/auc_vs_ratio.png)

Three fine-tuning episode lengths, swept across every deployment context. Every curve
**rises, peaks, then crosses into negative**. The three share that shape but neither
their height nor the exact position of their peak: in AUC the peaks sit at ratios of
0.63, 0.25 and 0.08, so they do not define one common optimum. What they do share is
the sign change, and the fact that it always happens on the same side — at large
contexts, never at small ones.

## The one table

AUC gained by fine-tuning, against the un-adapted model **at the same context**.
Same 37 dataset-runs in every cell — nothing is averaged across different
populations. `r` is the ratio.

| deploy with ↓ | episode 102 rows | episode 512 rows | episode 1638 rows |
|--:|--:|--:|--:|
| **64** | **+0.0264** `r=0.63` | +0.0142 `r=0.12` | +0.0068 `r=0.04` |
| **128** | **+0.0212** `r=1.25` | +0.0192 `r=0.25` | +0.0104 `r=0.08` |
| **256** | +0.0122 `r=2.51` | **+0.0157** `r=0.50` | +0.0103 `r=0.16` |
| **512** | +0.0054 `r=5.02` | **+0.0114** `r=1.00` | +0.0101 `r=0.31` |
| **1024** | −0.0019 `r=10.0` | +0.0043 `r=2.00` | **+0.0055** `r=0.63` |
| **2048** | −0.0094 `r=20.1` | −0.0042 `r=4.00` | **+0.0002** `r=1.25` |
| **full train** | −0.0088 `r=63.5` | −0.0048 `r=12.7` | **−0.0024** `r=3.96` |

Read it row by row. **The winning column moves right as the deployment context
grows**: the 102-row episode wins up to 128 rows of context, the 512-row episode at
256 and 512, and the 1638-row episode from 1024 upward. No single episode length wins
everywhere, and the longest never wins at small contexts.

The winners' ratios are 0.31, 0.63, 1.25, 0.50, 1.00, 0.63, 1.25 and 3.96 — six of the
eight between
0.5 and 1.3, but not a rule: at 128 and at 512 rows of context the winner is *not* the
episode closest to that band. The ratio locates the good region; it does not pick the
winner cell by cell.

## When this changes what you do

**1. You deploy with a short context for latency.** Fraud scoring, real-time
ranking, anything where attending over ten thousand rows per request is too slow.
Say you settle on 64 rows of context. Fine-tune with 102-row episodes and you gain
**+0.0264 AUC**; fine-tune with 1638-row episodes — the "more data must be better"
reflex — and you gain **+0.0068**. Same data, same steps, **four times less**,
purely because the episode no longer resembles what the model will see.

**2. Your training set is large.** `tabpfn` caps a fine-tuning episode at 50,000
context+query rows, while inference uses every row you give it. At 200,000 training
rows that is a ratio of 5, and at 400,000 a ratio of 10 — both in the range where
our measurements show fine-tuning doing nothing or actively hurting. Above roughly
100,000 rows the cap needs raising, or the inference context needs bounding. Which
of the two is a memory decision; leaving it unexamined is not.

**3. You hit an out-of-memory error while fine-tuning.** The reflex is to halve the
episode budget and keep everything else. That single change moves you rightwards
across this table, and the bottom-left cells are where the damage is.

## Bigger is not always better

"Use the longest episode you can afford" is not a law. It is what you observe when
every episode you *can* afford is still shorter than the context you deploy with.

![where the episodes fall](figures/where_the_episodes_fall.png)

* **Below 256 rows of deployment context the shortest episode wins.** At 64 rows the
  102-row episode gives **+0.0264** against **+0.0068** for the 1638-row one: four
  times more, for a sixteenth of the memory.
* **From 1024 rows upward the longest wins** — but that is a fact about this grid, not
  about length. 1638 context rows is simply the largest episode a T4 holds on tables of
  up to 110 features, and at those contexts all three episodes are still on the short
  side. We cannot see what a longer one would do, because we could not fit one.

So "longer is better" holds wherever GPU memory keeps every affordable episode shorter
than the deployment context — which is where the published work sits, and where most
practitioners are. It is a consequence of the memory ceiling as much as a property of
length. Lift the ceiling and the advice would need re-testing.

## What the ratio does and does not capture

The ratio is a compact summary. It earns that status on one axis and not on another,
and the difference is worth stating plainly rather than leaving a reader to find it.

**Where it holds.** At a matched ratio, changing the episode length sixteen-fold moves
the relative log-loss gain by 0.3 to 1.2 points and never significantly, while the
ratio itself moves it from +6% to −7%. The relative log-loss peaks sit at ratios of
0.63, 0.50 and 0.31 for the three episode lengths — within a factor of two of one
another.

**Where it breaks.** In AUC the peaks sit at 0.63, 0.25 and 0.08: a factor of eight,
which is no common optimum at all. And the sign change does not happen at a fixed
ratio either:

| episode | context where the AUC gain crosses zero | ratio there |
|--:|--:|--:|
| 102 rows | ~856 rows | 8.4 |
| 512 rows | ~1454 rows | 2.8 |
| 1638 rows | ~2251 rows | 1.4 |

**Read that table as the real finding.** A longer episode does push the crossing out —
856, then 1454, then 2251 rows — which is the useful, robust claim. But it pushes it
out **sub-proportionally**: a sixteen-fold longer episode buys only 2.6 times more
deployable context. Expressed as a ratio, the crossing therefore falls from 8.4 to 1.4
rather than staying put.

So: use the ratio to locate the regime you are in, and to see that the two lengths must
be chosen together. Do not use it as a constant you can solve for. The quantity that
behaves predictably is the direction of the effect, not its threshold.

## Too short is expensive

The two failures are not symmetric, and only one of them damages the model.

**Episode too short relative to the context — the model gets worse than if you had
not fine-tuned at all.** At a ratio of 20 the mean AUC change is **−0.0094** with
only **8 datasets out of 37 improving**; pooled across all runs, beyond a ratio of 8
the relative log-loss is **−5.9%**. You spend GPU time to degrade your model. This
is the failure that matters, and it is the one an out-of-memory workaround walks
straight into.

**Episode longer than needed — you leave gains on the table, but break nothing.**
At the smallest ratio we measured, 0.02, the AUC change is still **+0.0027** and
positive on 26 of 37 datasets. The benefit erodes toward zero; it does not turn
negative. The cost is memory and time, not accuracy.

## What the literature does with these two lengths

Every study we found sets both with a single number, so their ratio never varies and
the effect cannot appear.

| Study | How the two lengths are set | Ratio |
|---|---|---|
| den Breejen et al., [2024](https://arxiv.org/abs/2405.13396) | Support is `min(0.8 × n_train, 8192)`; evaluation uses the whole training set | **1.25**, rising to `n_train/8192` past 10,240 rows |
| den Breejen et al., 2023 (TRL @ NeurIPS) | 80 % split of the same training set used as context at evaluation | **1.25** |
| [LoCalPFN](https://arxiv.org/abs/2406.05207), NeurIPS 2024 | One neighbourhood size `k = min(10√N, 1000)`, shared by both phases | **exactly 1** |
| [Real-TabPFN](https://arxiv.org/abs/2507.03971), 2025 | Training context capped at 20,000 samples, 60 % of it context; evaluated on the AutoML Benchmark | 0.19 – 1.86 depending on the setting |

The 1.25 is not a coincidence: fine-tuning on an 80/20 split and then evaluating with
the *whole* training set as context gives 1/0.8 exactly. Verbatim quotes for each row
are in `notebooks/02_analysis.ipynb`.

## The library's defaults

`notebooks/01_inspect_api.ipynb` reads the installed package and prints file paths and
line numbers. For `tabpfn==9.0.0`:

* `n_finetune_ctx_plus_query_samples = 50000` — a cap on context + query
* `finetune_ctx_query_split_ratio = 0.2`
* `n_inference_subsample_samples = None` — the full training set is the context

Below the cap these give a ratio of exactly **1.25**, inside the band where 65 of 70
dataset-runs improve. That is a good choice, and our measurements support it.

The difficulty is that **the coupling is implicit**. Nothing in the signature says
these parameters must move together, so it survives only while the defaults are left
alone — and it is released in the two situations in the section above: past roughly
100,000 training rows, and whenever the episode budget is lowered to fit in memory.

```python
from ft_window import check, advise

check(n_train=200_000)
# ratio 5.00  (inference context 200,000 rows / episode context 40,000 rows)
# HARMFUL. Episodes are 5x shorter than the context you deploy with. …

advise(n_train=100_000, deployment_context=1_024).kwargs
# {'n_finetune_ctx_plus_query_samples': 2560, 'finetune_ctx_query_split_ratio': 0.2,
#  'n_inference_subsample_samples': 1024}
```

No dependencies, never imports `tabpfn`, uses the library's own parameter names.

## Limits

Binary classification, one model family, one benchmark suite, 200 gradient steps.

The threshold is **a plateau, not a point**: "fine-tune iff ratio ≤ t" scores 77.1 % ±
2.7 out of sample against 67.6 % for "always fine-tune", winning in 200 random splits
out of 200 — but its accuracy is flat for any t between 2.5 and 5, so a single number
would be false precision. `ft_window` takes the conservative end.

**Ratios below 0.1 are only reachable with deployment contexts of 128 rows or fewer**,
because the longest episode we could fit on a T4 is 1638 context rows. The far left of
the curve therefore describes a regime nobody deploys in, and "small ratio" is
confounded with "small context" there. The right-hand side carries no such confound.

**No mechanism.** Out of band, log-loss degrades 7.3× more than AUC does, against 2.2×
inside it, so most of the damage looks like miscalibration rather than lost knowledge.
A single temperature fitted at the deployment context would separate the two — in
binary classification it leaves AUC exactly unchanged. Designed, not run.

## Repository

```
data/          three runs, one CSV each, with a MANIFEST
ft_window/     advise() and check(), no dependencies
tests/         24 unit tests — python -m pytest tests/
notebooks/     01 audit (CPU) · 02 analysis (CPU, English) · 03, 03b, 04 experiments (GPU)
figures/
```

Every notebook is in English. `02_analysis.ipynb` is the one to run: it needs no GPU and
recomputes every number and both figures from the committed CSVs.

MIT licensed.
