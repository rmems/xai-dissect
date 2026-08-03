# Grok-1 campaign comparison: run2 (May) vs run3 (Aug 2026)

**Classification:** internal experimental comparison (checkpoint cartography re-run)  
**Depth:** structural + statistical + schema delta analysis  
**Sources:** local artifacts only (no external literature)

| Field | May run2 | Aug run3 |
|-------|----------|----------|
| Path | `~/rmems/grok-result/xai-dissect/grok1_run2_after_fixes_20260525T002904Z` | `~/rmems/grok-result/xai-dissect/grok1_run3_20260802T023050Z` |
| Generated | 2026-05-25T00:29:04Z | 2026-08-02T02:30:50Z → 02:51:47Z (~21 min) |
| Checkpoint | `.../Downloads/SNN_Quantization/grok-1-official/ckpt-0` (historical) | `~/.models/xai-grok-1/ckpt-0` |
| Slug | `grok-1-official__ckpt-0` | `xai-grok-1-ckpt-0` |
| Tooling baseline | pre planning-surface PRs | main after #32/#34/#36/#37/#48 |

---

## Executive summary

**Cartography is bit-stable on structure and sampled stats.** Re-running on the relocated Grok-1 ckpt-0 with the post-planning-surface binary reproduces:

- identical tensor identities (770 tensors; path-stripped inventory equal)
- identical experts atlas and routing report (path-stripped equal)
- identical coverage counts and checksum (`fnv1a64:de5a1c978121c62c`)
- identical offline stats aggregates (`avg_rms ≈ 19.76228`, `avg_var ≈ 631.45063` over 770 profiles at 65 536 samples/tensor)

**What changed is product surface, not the model map.** Run3 adds conversion/quant/pilot/route-preservation artifacts and expands SAAQ readiness to schema v2 with much richer candidate payloads. That is expected tool evolution, not checkpoint drift.

**Implication for GO/NO-GO:** structural prerequisites for a Grok-1 pilot remain green. Runtime gates (router agreement, block cosine) stay `unknown` by design until grok-ozempic (or similar) fills them.

---

## 1. Structural identity (unchanged)

| Metric | run2 | run3 | Match |
|--------|------|------|-------|
| model_family | grok-1 | grok-1 | yes |
| shards / tensors | 770 / 770 | 770 / 770 | yes |
| total payload bytes | 318 114 914 304 | 318 114 914 304 | yes |
| inferred (vocab, d_model, n_experts, d_ff, n_blocks) | 131072 / 6144 / 8 / 32768 / 64 | same | yes |
| dtype mix | 322 f32 + 448 int8 | same | yes |
| tensor identity sequence (ordinal, shape, offset, nbytes, kind, block) | — | — | **equal** |
| inventory tensors path-stripped | — | — | **equal** |
| experts path-stripped | — | — | **equal** |
| routing path-stripped | — | — | **equal** |

### Kind histogram (both runs)

| Kind | Count |
|------|------:|
| attn_proj_i8.model_width | 128 |
| attn_proj_i8.narrow | 128 |
| block_norm | 256 |
| final_norm | 1 |
| moe_expert.down / gate / up | 64 each |
| router | 64 |
| token_embedding | 1 |

### Coverage gate

| Field | run2 | run3 |
|-------|------|------|
| validation | pass | pass |
| coverage_schema_version | **1** | **2** |
| baseline_profile | *(absent)* | **`grok1-map-v1-clean`** |
| checksum | `fnv1a64:de5a1c978121c62c` | same |
| expected = discovered | blocks 64, tensors 770, routers 64, expert_families 192, unknown 0 | same |
| unknown_slots | [] | [] |

Coverage schema v2 only adds the explicit baseline label; the integrity hash is unchanged.

---

## 2. Statistical profile (unchanged at aggregate level)

Both runs used `--sample-values 65536` with the same near-zero thresholds (`f32_near_zero_abs=0.001`, `i8_near_zero_abs=1`).

| Aggregate | run2 | run3 |
|-----------|------|------|
| profiles | 770 | 770 |
| mean RMS (all tensors) | 19.76228195189274 | **identical** |
| mean variance | 631.4506342843616 | **identical** |

Shared JSON/MD bodies for inventory/experts/routing/stats differ by only ~24–30 bytes in most files — explained by shorter absolute `checkpoint_path` / `shard_path` strings on the new disk location, not by numerical drift.

---

## 3. SAAQ readiness: schema expansion (material delta)

| Artifact | run2 size | run3 size | Δ |
|----------|----------:|----------:|--:|
| `exports/saaq-readiness.json` | 77 445 | 1 416 674 | **+1.34 MiB** |
| `manifests/candidate-saaq-targets.json` | 926 | 371 133 | **+370 KiB** |
| `reports/saaq-readiness.md` | 22 917 | 140 966 | **+118 KiB** |

### Schema

| | run2 | run3 |
|--|------|------|
| `schema_version` (saaq export) | 1 | **2** |
| export keys | candidate_targets, inferred, layer_readiness, manifest, notes, risky_tensors, routing_critical_tensors, … | + **`deferred_tensors`**, **`precision_sensitive_tensors`**, **`quantization_candidates`** |

Run3 readiness report notes partition **770** tensors into:

- 448 quantization candidates  
- 64 routing-critical  
- 257 precision-sensitive  
- 1 deferred (`token_embedding`)

Top quant-plan readiness callout: `block_030.slot_01.moe_expert.down` (readiness ≈ 0.188).

This is **richer scouting output**, not a change in which tensors exist.

---

## 4. New planning artifacts (run3 only)

Absent from May run2; all require clean baseline and landed via #32/#34/#36:

| Artifact | Role | Highlights |
|----------|------|------------|
| `conversion-manifest.json` | Per-tensor conversion handoff (770 entries) | Policies: wrap_existing_int8_expert 192, wrap_existing_int8_unknown 256, passthrough_f32_norm 257, passthrough_f32_router 64, candidate_saaq_embedding 1 |
| `quant-plan.json` / `.md` | Family-level pilot policy | **keep_fp32:** router, block_norm, final_norm; **pilot_quantize:** attn + moe experts; **defer:** token_embedding |
| `pilot-selection-plan.json` / `.md` | Representative blocks | blocks **0, 8, 28, 60, 63**; modes attention_only / expert_only / attention_plus_expert |
| `route-preservation-report.json` / `.md` | Gate surface for runtime metrics | Router top-1/top-2, block cosine, etc. all **`status: unknown`** with thresholds reserved for downstream pilots |

These close the handoff gap to grok-ozempic / SAAQ without re-deriving the May map.

---

## 5. What did *not* regress

- No new unknown tensors  
- No router/expert family count drift  
- No expert unresolved entries  
- No stats sampling regression at default 65 536  
- No inventory kind reclassification  

---

## 6. Residual differences (non-semantic)

| Difference | Cause | Action |
|------------|--------|--------|
| Absolute paths in JSON | Checkpoint relocated under `~/.models/` | Normalize on consumer side; use ordinal + kind identity |
| Slug naming | Explicit `--checkpoint-slug` sanitized `__` → `-` | Prefer new slug for handoff |
| File size −24 B on many shared exports | Shorter path strings | Ignore |
| SAAQ size explosion | Schema v2 + full candidate expansion | Prefer run3 for scouting |
| Coverage schema 1 → 2 | Explicit `baseline_profile` | Consumers should accept v2 |

---

## 7. Actionable insights

1. **Trust run3 as the new structural baseline** for exports to grok-ozempic (`LATEST_CORRECT_GROK1_RUN`).  
2. **Do not re-litigate MoE geometry** — path-stripped experts/routing match May.  
3. **Start pilot work from** `pilot-selection-plan` blocks + `conversion-manifest` policies; protect routers/norms per quant-plan.  
4. **Fill route-preservation `unknown` metrics** in the quant runtime repo; xai-dissect cannot clear quant GO alone.  
5. **Optional:** store path-normalized golden fixtures from run3 for CI regression (checksum already stable).

---

## 8. Development plan

| Horizon | Work |
|---------|------|
| Short | Point grok-ozempic at `~/rmems/grok-result/xai-dissect/LATEST_CORRECT_GROK1_RUN/manifests/xai-grok-1-ckpt-0/` |
| Medium | Run attention/expert pilot modes on blocks 0/8/28/60/63; write observed router/cosine into route-preservation |
| Long | Automate run3-style campaign in CI smoke against a tiny fixture; keep real-weight campaign as manual release gate |

---

## Verdict

| Question | Answer |
|----------|--------|
| Did the checkpoint map change? | **No** (structure + stats aggregates identical) |
| Did tooling output improve? | **Yes** (planning + conversion + SAAQ v2) |
| Prefer run for handoff? | **run3** |
| Quant GO? | Structural inputs **pass**; runtime metrics still **missing** |
