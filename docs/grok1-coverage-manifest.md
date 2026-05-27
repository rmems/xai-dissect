# Grok-1 Coverage Manifest: grok1-map-v1-clean

This document explains the `grok1-coverage.json` manifest produced by
`xai-dissect`, the `grok1-map-v1-clean` baseline profile, and the
FNV-1a 64-bit checksum algorithm used for reproducibility verification.

## What is grok1-map-v1-clean?

`grok1-map-v1-clean` is the canonical structural baseline profile for
Grok-1 ckpt-0. It was derived from the first complete parse of the
official Grok-1 release and encodes the expected structural invariants
of a fully-discovered, fully-classified Grok-1 checkpoint:

```text
blocks            = 64
tensors           = 770
routers           = 64
expert_families   = 192   (3 projection families × 64 layers)
unknown_tensors   = 0
```

The name components:
- `grok1` — Grok-1 family
- `map` — structural inventory / tensor table
- `v1` — first stable schema version
- `clean` — zero unknown tensors, complete classification

When `xai-dissect` validates a checkpoint inventory against this
profile, it checks that every structural invariant matches before
allowing the coverage manifest to be written. If any invariant fails,
`validation` becomes `"fail"` and the manifest is rejected by downstream
consumers.

## The five validation dimensions

The `Grok1CoverageManifest` encodes two `Grok1CoverageCounts` snapshots:

```json
{
  "expected": {
    "blocks": 64, "tensors": 770, "routers": 64,
    "expert_families": 192, "unknown_tensors": 0
  },
  "discovered": {
    "blocks": 64, "tensors": 770, "routers": 64,
    "expert_families": 192, "unknown_tensors": 0
  }
}
```

Each dimension has a specific meaning for Grok-1 quantization readiness:

| Dimension | Value | Why it matters |
|----------|-------|----------------|
| `blocks` | 64 | Grok-1 has exactly 64 transformer layers. Any deviation means block assignment failed or the checkpoint is not a standard Grok-1 run. |
| `tensors` | 770 | 1 embedding + 64×12 block shards + 1 final norm = 770. The count is a strong integrity check — adding or removing shards changes this number. |
| `routers` | 64 | One router per block. A missing router means block assignment skipped that block. An extra router means shard ordering is ambiguous. |
| `expert_families` | 192 | 3 projection families (gate, down, up) × 64 layers = 192. All three families must be present per block for the pilot quant plan to be complete. |
| `unknown_tensors` | 0 | Any unknown tensor means the classifier encountered a shape/dtype combination it could not resolve. Quantization of unknown tensors is risky — they may be structural dependencies. |

## FNV-1a 64-bit checksum algorithm

The checksum in `grok1-coverage.json` is computed to enable reproducible
comparison of structural inventories without depending on machine-local
paths. Downstream tooling can recompute the checksum from a new inventory
run and compare it against the baseline to detect structural drift.

FNV-1a 64-bit is a non-cryptographic hash function chosen for:
- **Deterministic**: same input always produces the same output
- **Fast**: O(n) over the byte sequence, no cryptographic overhead
- **Good avalanche**: small input changes produce large output changes
- **Well-specified**: no implementation-specific behavior (unlike some
  non-cryptographic hashes)

### Algorithm

FNV-1a 64-bit processes the input as a sequence of bytes:

```
hash ← FNV_offset_basis_64   // 14695981039346656037
for each byte b in input:
    hash ← hash XOR b
    hash ← hash × FNV_prime_64 // 1099511628211
return hash
```

In hexadecimal, the constants are:
```
FNV_offset_basis_64 = 0xcbf29ce484222325
FNV_prime_64          = 0x100000001b3
```

### Canonical representation

The checksum is **not** computed over the full JSON file. It is computed
over a **canonical structural view** that excludes:

- `shard_path` (machine-local, not portable)
- `checkpoint_path` (machine-local, not portable)
- Field ordering (JSON field order must be canonical)
- `offset` and `nbytes` (may vary slightly across filesystem block sizes)

The canonical view includes only:
1. `shard_ordinal` (stable 0-based index)
2. `in_shard_index` (stable intra-shard index)
3. `kind.kind` (the classification discriminant)
4. `kind.detail` (the classification detail, if present)
5. `shape` (the tensor shape as a list of u64 dimensions)
6. `role` (parser-level role: tensor / quant_weight / quant_scales)
7. `dtype` (f32 / i8)

For each tensor, these fields are serialized in a fixed order and
concatenated into a byte sequence. The full inventory's canonical view
bytes are then hashed with FNV-1a 64-bit.

This means: the checksum is stable across different machine paths,
different checkpoint directory names, and minor filesystem differences,
but will change if the tensor classification, shape, or ordering changes.

### Example checksum from the real Grok-1 run

From `out/grok1_run2_after_fixes_20260525T002904Z/`:

```json
{
  "checksum": "fnv1a64:de5a1c978121c62c",
  "validation": "pass"
}
```

A downstream pipeline that reruns `xai-dissect inventory` on the same
checkpoint on a different machine should produce the same checksum if:
- The Grok-1 checkpoint is byte-identical
- The classification logic has not changed
- The schema version is the same

If the checksum changes, the structural inventory has drifted — the
grok-ozempic consumer should reject the bundle and re-run with a
validated baseline.

## How downstream consumers use the manifest

The `grok1-coverage.json` manifest is the **fail-closed gate** for the
grok-ozempic handoff. Consumer logic must verify all of:

```text
validation       = "pass"
expected.blocks = discovered.blocks = 64
expected.tensors = discovered.tensors = 770
expected.routers = discovered.routers = 64
expected.expert_families = discovered.expert_families = 192
expected.unknown_tensors = discovered.unknown_tensors = 0
unknown_slots = []   (empty array — no unclassified tensors)
```

If any check fails, the bundle is rejected. The `checksum` field is
advisory — it provides fast approximate comparison without re-running the
full inventory, but the explicit equality checks above are the normative
acceptance criteria.

## Repacked Grok-1

Some downstream pipelines may re-shard or repack the Grok-1 checkpoint into
a different storage format (e.g., safetensors or a memory-mapped layout).
When this happens, the shard-level structure changes, so the inventory
would be different. The `grok1-map-v1-clean` profile does not directly
handle repacked checkpoints — the consumer should regenerate the
coverage manifest from the repacked layout and treat the new checksum
as the baseline for that layout.

## The grok1-coverage.json from the real Grok-1 run

Complete 22-line file from `out/grok1_run2_after_fixes_20260525T002904Z/`:

```json
{
  "model_family": "grok-1",
  "schema_version": 2,
  "coverage_schema_version": 1,
  "validation": "pass",
  "checksum": "fnv1a64:de5a1c978121c62c",
  "expected": {
    "blocks": 64, "tensors": 770, "routers": 64,
    "expert_families": 192, "unknown_tensors": 0
  },
  "discovered": {
    "blocks": 64, "tensors": 770, "routers": 64,
    "expert_families": 192, "unknown_tensors": 0
  },
  "unknown_slots": []
}
```

The two `coverage_schema_version` values distinguish the coverage
manifest schema (`coverage_schema_version = 1`) from the outer
`ModelInventory` schema (`schema_version = 2`). This allows the two
versions to evolve independently.