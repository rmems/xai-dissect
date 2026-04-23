# Grok-2 Future Support

This document exists to keep future Grok-2 work bounded. It is a planning
checklist, not a commitment.

## Preconditions

Work should not start until all of the following are true:

- public Grok-2 weights are actually available
- the weight license is compatible with this repo's lawful-analysis scope
- at least one checkpoint layout can be inspected locally

## First Questions To Answer

- Is the shard/container format still compatible with the current parser?
- Are the dtype assumptions still limited to `f32` and `int8`, or do new
  dtypes appear?
- Does the expert/routing layout still match the current structural heuristics?
- Do output artifacts still fit the same inventory/expert/routing/stats model?

## Minimal Acceptance For Initial Grok-2 Support

- `inventory` can parse one real Grok-2 checkpoint directory
- `experts` can describe expert structure without a forward pass
- `routing-report` can identify likely routing tensors structurally
- `stats` and `saaq-readiness` can run offline without turning into a runtime
- outputs remain under the same `reports/`, `exports/`, and `manifests/`
  conventions

## Explicit Non-goals For The First Grok-2 Pass

- no inference runtime
- no checkpoint conversion pipeline
- no quantization runtime
- no "support every Grok-2 variant" promise before one real checkpoint works

## Suggested Work Breakdown

1. Parser compatibility audit
2. Inventory/schema adjustments only where the checkpoint requires them
3. Expert and routing heuristic validation
4. Stats/readiness validation
5. README and changelog updates after one real checkpoint is confirmed

## Information To Capture In Future Issues

- checkpoint source and license
- shard naming and directory layout
- observed dtypes
- one or two representative tensor shapes
- exact command and failure mode if the current CLI breaks
