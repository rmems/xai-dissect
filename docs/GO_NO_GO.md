# Grok-1 Quantization GO / NO-GO Gate

This checklist is the explicit decision gate for broader or full Grok-1 quantization work.

`xai-dissect` remains a structural-analysis and planning tool. It does **not** execute
full quantization runs, mutate checkpoints, or serve as an inference runtime. This gate
exists so downstream pilot or quantization repos only proceed once the required evidence
exists.

## Required inputs before GO

- Clean structural baseline passes with profile `grok1-map-v1-clean`
- Readiness report ranks expert and attention families as candidates
- `quant-plan.json` exists and validates
- Pilot selection plan exists for the representative block set
- Routers and norms remain protected in first-pass pilot planning
- Route-preservation report exists
- Artifact directories / filenames are stable and documented
- Cloud runbook exists before burning time-limited credits

Each threshold metric (router top-1/top-2 agreement, block output cosine,
etc.) is defined and its measurement methodology is explained in
[`docs/metric-definitions.md`](metric-definitions.md).

## GO thresholds

All of the following must be true:

- `unknown_tensors == 0`
- `routers == 64`
- `expert_families == 192`
- router top-1 agreement `>= 99.0%`
- router top-2 set agreement `>= 99.5%`
- block output cosine `>= 0.995`
- no router tensors modified
- no `block_norm` or `final_norm` tensors modified

## NO-GO triggers

Any one of the following stops the sprint from proceeding to broader/full runs:

- unknown tensors reappear
- router count changes
- expert family count changes
- route-preservation metrics are missing
- router agreement falls below threshold
- pilot artifacts are not reproducible
- cloud runbook is missing

## Reusable PR checklist block

```markdown
## Grok-1 Quantization Gate Decision

- [ ] clean baseline: `grok1-map-v1-clean`
- [ ] readiness ranking present
- [ ] `quant-plan.json` present
- [ ] pilot selection plan present
- [ ] route-preservation report present
- [ ] runbook linked
- [ ] router top-1 agreement >= 99.0%
- [ ] router top-2 set agreement >= 99.5%
- [ ] block output cosine >= 0.995
- [ ] routers untouched
- [ ] norms untouched

Decision: GO / NO-GO
Evidence links:
- baseline:
- readiness:
- quant-plan:
- pilot plan:
- route-preservation:
- runbook:
```
