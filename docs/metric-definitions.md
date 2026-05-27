# GO/NO-GO Threshold Methodology

This document defines the metrics referenced in `docs/GO_NO_GO.md` and
explains how each is computed, which downstream repo is responsible for
measuring it, and what would cause it to fail.

`xai-dissect` is a **structural analysis tool**. It does not execute
model inference and cannot compute these metrics directly. It produces the
structural artifacts (inventory, router-critical-tensors, coverage
manifest) that enable downstream repos to perform the measurements. The
metrics below are defined here so that both the structural gate (which
`xai-dissect` controls) and the runtime measurement (which downstream
repos control) are clearly documented.

---

## Router top-1 agreement

**Threshold**: `>= 99.0%`

### Definition

For a calibration dataset of input tokens, compute the router's top-1
expert selection (the expert with the highest logit) under the FP32
baseline and under the quantized pilot. The agreement is the fraction of
tokens for which both the FP32 router and the quantized router select the
same expert index.

```
agreement = (1 / N) * Σ_{i=1}^{N} [ argmax(router_FP32(token_i)) == argmax(router_Q(token_i)) ]
```

Where `[condition]` is 1 if the condition is true, 0 otherwise.

### Why 99.0%?

The router is the only path through which MoE expert selection flows.
A 1% disagreement rate means that 1 in 100 tokens reaches a different
expert, which cascades into different FFN computation and different
model outputs. At scale (billions of tokens), even a 0.5% router
disagreement can measurably shift output distributions. The 99.0%
threshold was chosen conservatively based on empirical sensitivity
analysis on comparable MoE models.

### What would cause failure

- Router weights are quantized to int8 with insufficient scale
  resolution, pushing the softmax output near the boundary between
  two experts
- Per-tensor quantization of the router changes the logit magnitudes
  enough to flip borderline decisions
- Calibration dataset is too small to be representative of the target
  distribution

### Responsible downstream repo

**grok-ozempic** — it executes the bounded pilot runs, captures router
logits for the calibration inputs, and computes the agreement fraction.
`xai-dissect` supplies `routing-critical-tensors.json` so grok-ozempic
knows exactly which tensors must be preserved as FP32.

---

## Router top-2 set agreement

**Threshold**: `>= 99.5%`

### Definition

The top-2 set is the unordered pair of the two highest-logit experts.
Even if the top-1 expert differs, the set agreement measures whether the
routing **context** is preserved:

```
set_agreement = (1 / N) * Σ_{i=1}^{N} [ top2_set_FP32(token_i) == top2_set_Q(token_i) ]
```

Where `top2_set` is the set `{expert_index_top1, expert_index_top2}`.

The top-2 threshold is higher than top-1 because the set agreement
matters for expert-load balancing — even if the first expert is the
same, a swapped second expert changes which FFN processes the token
and how the weighted sum is computed.

### Why 99.5%?

Top-2 agreement is a tighter bound. If the top-1 is correct 99.0% of
the time and the top-2 is correct 99.5% of the time, then the router
is mostly stable but the secondary expert selection is more sensitive.
The higher threshold reflects that secondary expert selection still
affects model quality.

---

## Block output cosine

**Threshold**: `>= 0.995`

### Definition

For each of the 64 transformer blocks, run a forward pass with the same
input activations through both the FP32 baseline and the quantized
pilot. Compare the block output activations (the residual stream after the
block's computation) using cosine similarity:

```
cosine(A, B) = (A · B) / (||A|| * ||B||)
```

Aggregate over all calibration tokens and report the mean cosine per
block, then report the minimum across all 64 blocks.

```
block_output_cosine = min_{block ∈ blocks} mean_{token ∈ calibration} cosine(output_FP32, output_Q)
```

### Why is this different from router agreement?

Router agreement measures the **input** to the MoE layer (which expert is
selected). Block output cosine measures the **output** of the entire
block — including how the selected experts processed the token, how the
residual stream was accumulated, and how RMSNorm transformed the
activations. A good router agreement score does not guarantee a good
block output cosine — quantization noise in the expert FFN layers can
distort the block output even if routing was preserved exactly.

### Why 0.995?

A cosine of 0.995 means the block outputs are very close — at the level
of numerical noise from FP16 accumulation differences. Lower values
indicate that quantization error is compounding across layers. This
threshold is conservative for Grok-1's 64-layer depth; accumulated error
across 64 layers at 0.99 per layer would be catastrophic.

### Responsible downstream repo

**grok-ozempic** — it captures intermediate activation tensors from the
bounded pilot runs and computes the per-block cosine similarity.

---

## All other metrics in route-preservation-report.json

The full `route-preservation-report.json` (from `tests/fixtures/exports/route-preservation.snap`)
declares 15 metrics across four scopes. All have `status: "unknown"` in
the structural report because they require downstream pilot evidence:

| Metric | Scope | Threshold | Requires |
|--------|-------|-----------|---------|
| `router_top1_agreement` | router_behavior | >= 99.0% | grok-ozempic pilot run |
| `router_top2_set_agreement` | router_behavior | >= 99.5% | grok-ozempic pilot run |
| `expert_load_distribution_delta` | router_behavior | — | routing traces from grok-ozempic |
| `expert_load_js_divergence` | router_behavior | — | routing traces from grok-ozempic |
| `router_logit_rank_correlation` | router_behavior | — | logit captures from grok-ozempic |
| `block_output_cosine` | block_behavior | >= 0.995 | activation captures from grok-ozempic |
| `block_output_rmse` | block_behavior | — | activation captures from grok-ozempic |
| `residual_stream_drift` | block_behavior | — | activation captures from grok-ozempic |
| `weight_reconstruction_mse` | weight_reconstruction | — | weight diff from grok-ozempic |
| `weight_cosine_similarity` | weight_reconstruction | — | weight diff from grok-ozempic |
| `weight_max_absolute_error` | weight_reconstruction | — | weight diff from grok-ozempic |
| `per_channel_scale_error_summary` | weight_reconstruction | — | quantization metadata from grok-ozempic |
| `logit_kl` | model_behavior | — | logit captures from grok-ozempic |
| `perplexity_delta` | model_behavior | — | perplexity evaluation from grok-ozempic |
| `generation_sanity_summary` | model_behavior | — | generation run from grok-ozempic |

`xai-dissect` defines the metric surface and the pass/fail thresholds.
It does not execute inference and cannot populate the observed values.

---

## What xai-dissect guarantees vs. what it requires

`xai-dissect` **guarantees** (through the structural inventory and
coverage manifest):
- The structural inventory is complete and deterministic
- Every tensor that should be FP32 (router, norms) is correctly classified
- Every expert family is present and named consistently
- The checksum is reproducible across reruns on the same checkpoint

`xai-dissect` **requires** (before GO is granted):
- `grok1-coverage.json` passes validation with `grok1-map-v1-clean`
- `routing-critical-tensors.json` identifies all 64 routers
- All planning artifacts (`quant-plan.json`, `pilot-selection-plan.json`,
  `route-preservation-report.json`) have been emitted
- The downstream runbook exists (linked in the PR checklist)

`xai-dissect` **cannot** guarantee:
- Router top-1 agreement >= 99.0% (requires inference)
- Block output cosine >= 0.995 (requires inference)
- Perplexity delta (requires inference)

These are the responsibility of `grok-ozempic` and are tracked as the
"observed" values in the route-preservation report.