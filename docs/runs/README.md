# Campaign run notes

Large Grok-1 campaign artifacts are **not** stored in this repo.

| What | Where |
|------|--------|
| Current baseline (run3) | `~/rmems/grok-result/xai-dissect/LATEST_CORRECT_GROK1_RUN` |
| May baseline (run2) | `~/rmems/grok-result/xai-dissect/grok1_run2_after_fixes_20260525T002904Z` |
| Comparison | [grok1_run3_vs_run2_comparison.md](./grok1_run3_vs_run2_comparison.md) |
| Storage root README | `~/rmems/grok-result/README.md` |

Optional local convenience (gitignored):

```bash
# If `out` already exists as a normal directory, `ln -sfn` can nest a link
# inside it instead of replacing the path. Refuse that footgun:
if [ -e out ] && [ ! -L out ]; then
  printf '%s\n' 'out exists and is not a symlink; remove or rename it first.' >&2
  exit 1
fi
ln -sfn "$HOME/rmems/grok-result/xai-dissect" out
```
