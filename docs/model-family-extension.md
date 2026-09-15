# Model-family extension points

This is a maintainer design document. It describes what must become
pluggable before `xai-dissect` can support a second Grok-family checkpoint
layout. It does **not** implement that abstraction.

**Status:** design only. No production-code change is required to land this
document.

Related:

- GitHub [#42](https://github.com/rmems/xai-dissect/issues/42)
- Linear RM-153
- Grok-2 checklist: [`docs/grok2-future-support.md`](grok2-future-support.md)
- Grok-1 architecture: [`docs/grok1-architecture.md`](grok1-architecture.md)
- Coverage gate: [`docs/grok1-coverage-manifest.md`](grok1-coverage-manifest.md)
- Pipeline layers: [`docs/architecture.md`](architecture.md)

## Why this document exists

Today the CLI is a Grok-1 dissector. Family identity is a string header
(`--family`, default `grok-1`), but layout, slot order, expected counts, and
planning validators are hardcoded against official Grok-1 `ckpt-0`.

The long-term goal is more than Grok-1. The next consumer is Grok-2, not
"any model." Before writing a `FamilyProfile` trait or moving files under
`families/grok1/`, this document inventories the Grok-1-specific surface and
proposes the module boundaries that a later implementation epic should
follow.

Copy the Grok-1 pattern: one family, one layout table, one fail-closed
coverage profile, one set of planning artifacts. Do not collapse that into a
generic "support every transformer" layer.

## Non-goals

This document, and any first implementation epic it feeds, does **not**:

- Add an inference runtime, forward pass, decode loop, or sampler.
- Add a conversion or quantization runtime. Planning artifacts describe
  policy; `grok-ozempic` applies it.
- Promise "support every model," every Grok variant, or Hugging Face
  transformers-at-large.
- Change the pickle scanner into a plugin host, or add safetensors /
  GGUF writers, in the first family split.
- Implement Grok-2 parsing, classification, or coverage in this issue.
- Open a single GitHub issue titled "support any model" / "support Grok-2."
  Follow-on work belongs in a bounded epic with one family (or one
  checkpoint layout) per issue.

These match [`docs/non_goals.md`](non_goals.md) and the first Grok-2 pass
listed in [`docs/grok2-future-support.md`](grok2-future-support.md).

## Proposed boundaries

Keep three layers distinct. Grok-1 currently mixes the last two inside
`inventory/` and `planning/`.

```
  raw shard bytes
        |
        v
  [ parser / format ]          bytes → RawTensor
        |                      family-agnostic container grammar
        v
  [ schema ]                   stable on-wire types
        |
        v
  [ family profile / semantics ]   RawTensor + layout → TensorKind,
        |                          block_index, block_slot, projections
        v
  [ coverage validator ]       fail-closed completeness gate
        |
        +--> experts / routing / stats   consume classified inventory
        +--> planning                    requires a passing coverage profile
        +--> reports / exports
```

### 1. Parser (format)

**Job:** turn one shard file into `parser::RawTensor` records. No model
semantics.

Current Grok-1 implementation: `src/parser/mod.rs` (`dissect_shard`).

Format assumptions that happen to match official Grok-1 JAX/Pickle shards:

| Symbol | Meaning |
| ------ | ------- |
| `OP_PROTO` + `0x04` | Pickle Protocol 4 magic |
| `DTYPE_TAG_F32` / `DTYPE_TAG_I8` | only `f32` and `int8` ndarrays |
| `SIG_QW8_*` | `QuantizedWeight8bit` class tags |
| `assign_qw8_roles` | int8 body → `quant.weight`; first following f32 → `quant.scales` |

A second family that still ships PROTO-4 pickle + `QuantizedWeight8bit`
should reuse this parser unchanged. A family that does not (new dtypes,
safetensors, a different quant envelope) needs a **format** change, not a
family-profile tweak. That is a parser issue, filed separately from
"add family X."

The parser must stay ignorant of `d_model`, slot order, block count, and
`model_family`.

### 2. Family profile (semantics)

**Job:** given format-level tensors and a family id, produce a classified
`ModelInventory`: hyperparameters, `TensorKind`, `block_index` /
`block_slot`, and source-backed projection names.

Current Grok-1 implementation is inlined in `src/inventory/mod.rs`:

| Step | Function | Grok-1-specific piece |
| ---- | -------- | --------------------- |
| Shard glob | `InventoryConfig.prefix` (default `"tensor"`) | official leaf naming |
| Infer width / experts | `infer_hyperparams` | `vocab_size`, `d_model`, `n_experts`, `d_ff` from shapes; does **not** set `n_blocks` |
| Classify | `classify_tensor` | Grok-1 shape language parameterized by inferred `hp` |
| Block windows | `assign_block_indices` / `assign_block_indices_for_scan` | `candidates = [12]` (`K = 12`); this pass sets `inferred.n_blocks` |
| Layout pick | `choose_grok_block_layout` / `grok_layout_candidate` | embedding + edge-norm + equal block windows |
| Gate vs up | `disambiguate_grok1_moe_projection_slots` | Haiku restore order, slots 0/1/2 |

`model_family` on `ModelInventory` is currently a **label copied from
`--family`**. It does not select a profile. `build_inventory` only branches
on `cfg.model_family == "grok-1"` to run MoE slot disambiguation. Passing
`--family grok-2` today writes `"grok-2"` into the JSON header and still
applies the Grok-1 `K = 12` block layout.

That is the bug a family profile exists to fix: layout and classification
must be selected by family id, not inferred from a string stamped after the
fact.

Experts and routing are **consumers** of the classified inventory. They
already key off `TensorKind`, inferred `n_experts` / `d_model`, and
`block_slot`. They should not grow a second copy of `GROK1_SLOT_SPECS`. If
a new family needs different expert geometry, extend the profile (and
maybe `TensorKind`), then let `experts::build_expert_atlas` and
`routing::build_routing_report` follow the inventory.

### 3. Coverage validator

**Job:** fail-closed completeness proof for one named baseline profile.
Independent of the parser. Independent of stats sampling.

Current Grok-1 implementation: `src/inventory/grok1_coverage.rs`, emitted as
`manifests/<slug>/grok1-coverage.json` (`schema::Grok1CoverageManifest`).

The Grok-1 profile `grok1-map-v1-clean` encodes:

```text
blocks          = 64
tensors         = 770
routers         = 64
expert_families = 192   (3 projections × 64 layers)
unknown_tensors = 0
```

plus per-slot signatures in `GROK1_SLOT_SPECS` and metadata checks against
`GROK1_D_MODEL`, `GROK1_D_FF`, `GROK1_N_EXPERTS`, `GROK1_EXPECTED_VOCAB_SIZE`.

A second family needs **its own** baseline name, expected counts, slot
table, and manifest filename (or a family field inside a shared coverage
schema). Do not reuse `grok1-map-v1-clean` or treat a missing
`grok1-coverage.json` as a pass. Downstream handoff rules:
[`docs/export-contracts.md`](export-contracts.md).

Planning (`src/planning/mod.rs`) already depends on coverage:
`validate_grok1_clean_baseline` calls `validate_grok1_complete_manifest`
before building conversion / quant / pilot / route-preservation artifacts.
That dependency should remain: planning is downstream of a green coverage
gate, not a fourth way to hardcode `6144×8`.

## Inventory of Grok-1-hardcoded touchpoints

Production symbols only. Tests and docs that restate the same constants
are listed at the end as follow-the-fixture, not as extra abstraction
surface.

### Parser / format

| File | Symbol | What is Grok-1-shaped |
| ---- | ------ | --------------------- |
| `src/parser/mod.rs` | `dissect_shard` | PROTO 4 required |
| `src/parser/mod.rs` | `DTYPE_TAG_F32`, `DTYPE_TAG_I8` | dtype closed set |
| `src/parser/mod.rs` | `SIG_QW8_STRICT`, `SIG_QW8_LOOSE_MODULE`, `SIG_QW8_CLASS_TAG` | `QuantizedWeight8bit` envelope |
| `src/parser/mod.rs` | `assign_qw8_roles`, `find_qw8_sites` | quant pairing rule |

### Schema / export types

| File | Symbol | What is Grok-1-shaped |
| ---- | ------ | --------------------- |
| `src/schema/mod.rs` | `GROK1_BASELINE_PROFILE` | `"grok1-map-v1-clean"` |
| `src/schema/mod.rs` | `Grok1CoverageManifest` | family-named coverage document |
| `src/schema/mod.rs` | `Grok1CoverageCounts` | five Grok-1 dimensions |
| `src/schema/mod.rs` | `Grok1UnknownSlot` | coverage unknown-slot rows |
| `src/schema/mod.rs` | `default_grok1_baseline_profile` | serde default for that profile |
| `src/schema/mod.rs` | `ModelInventory.model_family` (and siblings) | string; only `"grok-1"` is supported |
| `src/schema/mod.rs` | `TensorKind` variants | Grok-1 role vocabulary (embedding, norms, router, MoE, attn widths) |
| `src/inventory/mod.rs` | `SCHEMA_VERSION` | inventory schema; bump on incompatible kind/layout changes |

### Family profile / inventory semantics

| File | Symbol | What is Grok-1-shaped |
| ---- | ------ | --------------------- |
| `src/inventory/mod.rs` | `InventoryConfig::default` | `prefix = "tensor"`, `model_family = "grok-1"` |
| `src/inventory/mod.rs` | `build_inventory` | calls Grok-1 MoE disambiguation when family is `"grok-1"` |
| `src/inventory/mod.rs` | `infer_hyperparams` | `vocab_size` / `d_model` / `n_experts` / `d_ff` from shapes; does not set `n_blocks` |
| `src/inventory/mod.rs` | `classify_tensor` | Grok-1 shape → `TensorKind` rules |
| `src/inventory/mod.rs` | `assign_block_indices` | `candidates = [12]`; sets `inferred.n_blocks` |
| `src/inventory/mod.rs` | `assign_block_indices_for_scan` | skips layout when `--limit` truncated the scan |
| `src/inventory/mod.rs` | `GrokBlockLayout`, `choose_grok_block_layout`, `grok_layout_candidate` | embedding + edge-norm + 64×K windows |
| `src/inventory/mod.rs` | `disambiguate_grok1_moe_projection_slots` | only runs for `"grok-1"` |
| `src/inventory/mod.rs` | `grok1_moe_projection_for_slot` | slot 0 gate, 1 down, 2 up |
| `src/inventory/mod.rs` | `grok1_moe_projection_shape_matches` | `(E, d_model, d_ff)` vs `(E, d_ff, d_model)` |
| `src/cli/mod.rs` | `ModelFamilyArg`, `PlanningFamilyArg` | `--family` default `"grok-1"` |
| `src/cli/mod.rs` | `validate_complete_inventory_scope` | planning commands require a complete Grok-1-style scan |
| `src/cli/mod.rs` | `run_pilot_plan`, `run_route_preservation`, `run_quant_plan` | always call `build_grok1_*` |
| `src/main.rs` | `Command::*` family fields | CLI wiring for the same default |

### Coverage validator

| File | Symbol | What is Grok-1-shaped |
| ---- | ------ | --------------------- |
| `src/inventory/grok1_coverage.rs` | `GROK1_COVERAGE_SCHEMA_VERSION` | `2` |
| `src/inventory/grok1_coverage.rs` | `GROK1_EXPECTED_BLOCKS` | `64` |
| `src/inventory/grok1_coverage.rs` | `GROK1_EXPECTED_TENSORS` | `770` |
| `src/inventory/grok1_coverage.rs` | `GROK1_EXPECTED_ROUTERS` | `64` |
| `src/inventory/grok1_coverage.rs` | `GROK1_EXPECTED_EXPERT_FAMILIES` | `192` |
| `src/inventory/grok1_coverage.rs` | `GROK1_EXPECTED_VOCAB_SIZE` | `131_072` |
| `src/inventory/grok1_coverage.rs` | `GROK1_D_MODEL` | `6_144` |
| `src/inventory/grok1_coverage.rs` | `GROK1_D_FF` | `32_768` |
| `src/inventory/grok1_coverage.rs` | `GROK1_N_EXPERTS` | `8` |
| `src/inventory/grok1_coverage.rs` | `GROK1_BLOCK_SLOTS` | `12` |
| `src/inventory/grok1_coverage.rs` | `GROK1_*_SHAPE` | per-kind expected dims |
| `src/inventory/grok1_coverage.rs` | `Grok1SlotSpec`, `Grok1ExpectedKind`, `GROK1_SLOT_SPECS` | 12-slot table |
| `src/inventory/grok1_coverage.rs` | `should_validate_grok1_coverage` | family id + 770-tensor / canonical-layout gate |
| `src/inventory/grok1_coverage.rs` | `is_canonical_grok1_shard_layout` | `(shard_count - 2) % 12 == 0` |
| `src/inventory/grok1_coverage.rs` | `validate_grok1_complete_manifest` | fail-closed entry |
| `src/inventory/grok1_coverage.rs` | `validate_grok1_metadata` | inferred hp must match constants |
| `src/inventory/grok1_coverage.rs` | `validate_grok1_blocks`, `validate_grok1_expected_slots`, `validate_grok1_slot_signature` | occupancy + signatures |
| `src/inventory/grok1_coverage.rs` | `grok1_structural_name`, `grok1_checksum` | path-independent FNV-1a view |
| `src/exports/mod.rs` | `write_inventory_bundle` | calls the Grok-1 coverage gate |
| `src/exports/mod.rs` | coverage path `"grok1-coverage.json"` | family-named artifact |
| `src/exports/mod.rs` | `remove_stale_coverage_manifest` | leftover Grok-1 gate file |
| `src/report/mod.rs` | `write_grok1_coverage_manifest_json` | JSON writer for that document |

### Planning (downstream of coverage)

| File | Symbol | What is Grok-1-shaped |
| ---- | ------ | --------------------- |
| `src/planning/mod.rs` | `validate_grok1_clean_baseline` | requires `GROK1_BASELINE_PROFILE` |
| `src/planning/mod.rs` | `build_grok1_planning_artifacts` | conversion-manifest + quant-plan |
| `src/planning/mod.rs` | `build_grok1_pilot_selection_plan` | selected block list |
| `src/planning/mod.rs` | `build_grok1_route_preservation_report` | router/block metric gates |
| `src/planning/mod.rs` | `GROK1_EXPECTED_EXPERTS_PER_BLOCK` | `8` |
| `src/planning/mod.rs` | `GROK1_EXPECTED_EXPERT_FAMILIES_PER_BLOCK` | `3` |
| `src/planning/mod.rs` | `GROK1_EXPECTED_ROUTER_SHAPE` | `[6144, 8]` (duplicated literals, not `GROK1_D_MODEL`) |
| `src/planning/mod.rs` | `GROK1_KEEP_FP32_ORDER`, `GROK1_PILOT_QUANTIZE_ORDER`, `GROK1_DEFER_ORDER` | policy buckets |
| `src/planning/mod.rs` | `GROK1_PILOT_BLOCKS` | blocks 0, 8, 28, 60, 63 |
| `src/planning/mod.rs` | `validate_expert_atlas`, `validate_routing_report` | Grok-1 atlas/router checks |

### Downstream consumers (follow inventory; do not duplicate the slot table)

| File | Symbol | Notes |
| ---- | ------ | ----- |
| `src/experts/mod.rs` | `build_expert_atlas`, `infer_expert_projection` | uses inventory `kind` / slot; module docs cite Grok-1 slot order |
| `src/routing/mod.rs` | `build_routing_report` | `(d_model, n_experts)` routers; Grok-1 example is `(6144, 8)` |

### Tests and docs that pin the same constants

These should move **with** the Grok-1 profile, not become a second source of
truth:

- `src/inventory/test_fixtures.rs` — `canonical_grok1_inventory`,
  `CANONICAL_BLOCK_SLOTS`
- `tests/support/mod.rs` — sample inventories tagged `"grok-1"`
- `docs/grok1-architecture.md`, `docs/grok1-coverage-manifest.md`,
  `docs/tensor-schema.md`, `docs/observed-grok1-ckpt0.md`

## Minimum interface sketch

Pseudocode for a later implementation. Not present in the tree. Names can
change; the **cuts** should not: format vs profile vs coverage.

```rust
/// Format layer. No d_model, no slots, no family id.
pub trait ShardFormat {
    fn probe(bytes: &[u8]) -> bool;
    fn dissect_shard(path: &Path) -> Result<Vec<RawTensor>>;
}

/// Semantic family. Selected by `--family` / `InventoryConfig.model_family`.
pub trait FamilyProfile {
    fn id(&self) -> &'static str;                 // "grok-1"
    fn shard_prefix(&self) -> &'static str;       // "tensor"
    fn block_layout(&self) -> BlockLayoutSpec;    // K, singleton placement rules
    fn classify(&self, t: &RawTensor, hp: &InferredHyperparams) -> TensorKind;
    fn assign_blocks(
        &self,
        tensors: &mut [TensorInfo],
        shard_count: usize,
    ) -> Option<u32>;
    fn disambiguate_projections(
        &self,
        tensors: &mut [TensorInfo],
        hp: &InferredHyperparams,
    );
}

pub struct BlockLayoutSpec {
    pub shards_per_block: u32,          // Grok-1: 12
    pub embedding_singleton: bool,      // Grok-1: shard 0
    pub norm_singleton: NormPlacement,  // tail or after embedding
}

/// Fail-closed gate. One named baseline per (family, layout).
pub trait CoverageProfile {
    fn family_id(&self) -> &'static str;
    fn baseline_name(&self) -> &'static str;      // "grok1-map-v1-clean"
    fn manifest_filename(&self) -> &'static str;  // "grok1-coverage.json"
    fn should_validate(&self, inv: &ModelInventory) -> bool;
    fn validate(&self, inv: &ModelInventory) -> Result<FamilyCoverageManifest>;
}

/// Planning stays behind a green coverage profile.
pub trait FamilyPlanning {
    fn build_conversion_and_quant(
        &self,
        inv: &ModelInventory,
        atlas: &ExpertAtlas,
        routing: &RoutingReport,
        readiness: &SaaqReadinessReport,
    ) -> Result<(ConversionManifest, QuantPlan)>;
}
```

Suggested module layout when code *does* move (existing issue #44, not this
PR):

```text
src/
  parser/                 # format; keep as-is until a second container appears
  families/
    mod.rs                # registry: id → &'static dyn FamilyProfile
    grok1/
      mod.rs              # InventoryConfig defaults, classify, slots
      coverage.rs         # today's grok1_coverage.rs
      planning.rs         # today's build_grok1_* + GROK1_* policy tables
  inventory/mod.rs        # walk shards, call parser + selected profile
```

`FamilyCoverageManifest` may start as a rename of `Grok1CoverageManifest`
with `model_family` already on the struct. Do not invent a parallel JSON
document until a second family proves the five Grok-1 count dimensions are
insufficient.

Registry policy for v1:

- Unknown `--family` is a hard CLI error, not a silent Grok-1 parse with a
  different header.
- Only registered profiles run `disambiguate_projections` and coverage.
- Grok-1 remains the default so existing commands stay stable.

## How to add a family (using the Grok-1 pattern)

Worked example: Grok-1 is the template. Replace the italic parts.

1. **Confirm format.** Point the `dissect` CLI at a **checkpoint
   directory** (`dissect /path/to/ckpt --limit 1`). The command
   `read_dir`s its positional path, so a shard file fails. To scan one
   file, call `parser::dissect_shard` from a test or REPL. If PROTO 4 /
   `f32`+`int8` / `QuantizedWeight8bit` still hold, keep the parser. If
   not, file a parser issue *first* and stop. Capture dtypes, a few
   shapes, and the exact failure in the Grok-2 issue template
   (`.github/ISSUE_TEMPLATE/grok2-support.md`).
2. **Name the family.** Stable id string (`grok-1`, later `grok-2`). This
   is the `--family` value and every export's `model_family`. Do not
   overload `grok-1` for a different layout.
3. **Write a layout table** equivalent to Grok-1's 12-slot block (see
   [`docs/grok1-architecture.md`](grok1-architecture.md)). Need:
   shards per block `K`, singleton placement, per-slot role/dtype/shape,
   source-backed projection order if shape collides (Grok-1 gate vs up).
4. **Infer hyperparameters without copying Grok-1 constants.** Follow
   `infer_hyperparams` for embedding → `vocab_size`/`d_model` and expert
   stack → `n_experts`/`d_ff`. `n_blocks` comes from the layout pass
   (`assign_block_indices` / `assign_block_indices_for_scan`), not from
   `infer_hyperparams`. Hardcoded widths belong in the **coverage**
   profile, not in classification.
5. **Classify, then disambiguate.** Shape rules first (`classify_tensor`);
   slot map second (`grok1_moe_projection_for_slot`). A new family that
   also cannot tell gate from up by shape needs its own restore-order
   table, not Grok-1's slots 0/1/2.
6. **Add a coverage baseline** modeled on `grok1-map-v1-clean`: expected
   block/tensor/router/expert/unknown counts, slot signatures, checksum
   over a path-independent canonical view. New filename or an explicit
   `model_family` check so `grok1-coverage.json` cannot green-wash a
   different layout.
7. **Keep planning behind that gate.** Copy the Grok-1 split: routers and
   norms stay FP32 in the first pilot; expert/attention families are
   compression candidates; embeddings deferred. Pilot *block indices* are
   family-specific (`GROK1_PILOT_BLOCKS` is not portable).
8. **Wire CLI last.** Register the id, reject unknown families, add a
   canonical fixture like `canonical_grok1_inventory()`, and extend
   export-contract docs. Golden snaps (`tests/fixtures/exports/*.snap`)
   change only when an export schema changes; a new family should add
   snaps, not rewrite Grok-1 ones.
9. **Docs.** Architecture notes, coverage algorithm, observed-checkpoint
   notes, README "supported now / not yet" lines. Grok-2's checklist
   lives in [`docs/grok2-future-support.md`](grok2-future-support.md).

### What not to copy from Grok-1

- Literal `6144`, `32768`, `770`, or `K = 12` into shared inventory code.
- `GROK1_EXPECTED_ROUTER_SHAPE` duplicated in planning; planning should
  read the coverage profile (or inferred hp) instead of restating dims.
- Filing one issue that both abstracts families **and** implements Grok-2.
- Treating `--family` as a cosmetic JSON stamp.

## Grok-2 as first consumer

Grok-2 is the first family that should exercise this split. Public weights
exist; this CLI still has **no** Grok-2 parser profile, inventory
guarantee, or coverage gate.

Preconditions, first questions, and the initial acceptance bar are owned
by [`docs/grok2-future-support.md`](grok2-future-support.md). Do not start
Grok-2 implementation until those preconditions are true (Grok-1 pilot /
slice-quant handoff stable, local Grok-2 checkpoint in hand, license in
scope).

When that work starts, map it onto the boundaries above:

| Grok-2 question (from the checklist) | Layer |
| ------------------------------------ | ----- |
| Shard/container still PROTO 4 pickle? New dtypes? | Parser / format |
| Expert/routing layout vs current heuristics | Family profile |
| Same `reports/` / `exports/` / `manifests/` tree | Schema + output conventions |
| Completeness proof for one real checkpoint | New coverage profile, not `grok1-map-v1-clean` |
| Pilot / route-preservation artifacts | Planning, after coverage passes |

Expected Grok-2 deltas to anticipate (not yet measured): different
`n_experts` / `d_model` / `K`, possibly different MoE restore order, possibly
new `TensorKind` variants. Schema guidance:
[`docs/tensor-schema.md`](tensor-schema.md) (forward compatibility).

Issue hygiene: use `.github/ISSUE_TEMPLATE/grok2-support.md` per checkpoint
layout. Split parser-audit vs profile vs coverage vs planning. Do not open
"support Grok-2" or "support any model" as a single issue.

## Follow-on implementation epic

This design feeds a later code epic. It does not open that epic.

Natural issue cuts (already tracked separately, not created here):

1. Isolate the Grok-1 profile (`families/grok1`) with **no** behavior
   change — mechanical move of today's functions and constants.
2. Introduce the registry / traits with Grok-1 as the only impl; unknown
   `--family` becomes an error.
3. Deduplicate planning literals (`[6144, 8]`) onto the coverage profile.
4. **Then** a Grok-2 family issue, gated on
   [`docs/grok2-future-support.md`](grok2-future-support.md).

Landing this document is the only acceptance criterion for [#42](https://github.com/rmems/xai-dissect/issues/42).
