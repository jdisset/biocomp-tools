# biocomp-tools Pre-Release Audit (2026-05-20)

## Summary

Repo is **structurally ready for public release**. Required artifacts (LICENSE,
SPDX headers, secret cleanup) are in place. A handful of code-quality issues
remain that are not release-blockers but should be addressed before/right after
publishing.

## What was changed in this pass

### Release blockers (done)
- **`LICENSE`** — Created with MIT license, 2026 copyright to Jean Disset.
- **`pyproject.toml`** — Added `license = {file = "LICENSE"}`.
- **SPDX headers** — Added `# SPDX-License-Identifier: MIT` and copyright line
  to all 149 `.py` files under `biocomptools/`. Shebangs preserved.
- **Hardcoded personal info removed:**
  - `configs/default.yaml`: `/Users/jeandisset/.google/biocomp/key.json` -> `~/.google/biocomp/key.json` (and is `getenv`-controlled).
  - `configs/default.yaml`: hardcoded MLFlow URL `https://mlf.rachael.jdisset.com` -> empty default; user supplies via env var.
  - `configs/default.yaml`: hardcoded Google Sheet key -> empty default via env var.
  - `download_mlflow.py`: removed default URL `https://mlf.rachael.jdisset.com`.
- **`.gitignore`** — added `tmp/`, `runner_logs/`, `.ruff_cache/`, `.pytest_cache/`, `.venv/`.

### Real bug found and fixed
- `configs/plots/autofig_combined.yml:38` had a YAML/Dracon interpolation
  syntax error: `D.metadata[prediction_stats']['rmse']` (missing opening quote).
  Fixed to `D.metadata['prediction_stats']['rmse']`.

### LLM-ism cleanup (done)
- Replaced **167 unicode chars across 45 files** with ASCII equivalents:
  em-dash `-`, en-dash `-`, ellipsis `...`, right arrow `->`, double-arrow `=>`,
  check `ok`, cross `x`, smart quotes `"` / `'`.
- Skipped `toollib/interactive_link.py` for em-dashes - it uses `—` legitimately
  as a "null" placeholder in HTML/JS UI content. Two docstring em-dashes there
  were fixed manually.
- Stripped verbose docstrings and trivial inline comments from the most
  LLM-touched files (commits since 2026-02):
  - `step_history_triage.py` — collapsed 130 lines to 70.
  - `step_writer.py` — removed redundant class/method docstrings and
    self-narrating comments.
  - `history_db.py` — removed ASCII section banners
    (`# ====...`, `# ----...`), redundant class docstrings, trivial inline
    comments.
  - `logger_runner.py` — same treatment.
  - `logger_dispatch.py` — same treatment.
  - `run_replay.py` — same treatment.
  - `configs/default.yaml` — dropped LLM-style "Enable to save comprehensive
    state at critical points..." commentary.

### Secrets scan
Ran multiple regex passes for api/secret/key/token/password patterns.
**Nothing actionable found.** Only matches were:
- The MLFlow URL and Google Sheets key (removed above).
- Test files referencing `secret` and `token` as variable names in mocks - benign.

## Recommended follow-ups (NOT done — outside the requested scope)

These are real code-quality issues worth addressing soon but not strictly
release-blockers. Listed in priority order.

### 1. `from __future__ import annotations` violates CLAUDE.md (43 files)
CLAUDE.md explicitly says: *"NEVER use `from __future__ import annotations` — it
breaks Pydantic, SQLModel, and runtime type introspection."* Yet 43 files still
have this import. Many are core (`modelmodel.py`, `optuna_hyperopt.py`,
`design_hyperopt.py`, `hyperopt_analysis.py`, `tuner/api.py`, `tuner/session.py`,
`tuner/param_schema.py`, `toollib/interactive_link.py`, `toollib/design_selection.py`,
`toollib/sample_efficiency.py`).

Suggested action: bulk-remove these and verify Pydantic models still resolve.
This must be done carefully because some annotations may have been silently
incorrect under the deferred-evaluation regime.

### 2. Backward-compat aliases (CLAUDE.md violation)
Found 7 lingering "backward compat" stubs flagged by grep:
- `history_db.py`: `update_end_time()`, `step_count()`, `step_range()` - aliases for newer methods. Tests in `test_history_db.py` still use the old names; removing them requires updating those tests.
- `history_db.py`: `save_step_legacy()` is only referenced by `test_history_db_v2.py::test_save_step_legacy`. Both can be deleted.
- `logger_dispatch.py`: `async_handler` property returns `None` purely for compat. Callers should be migrated.
- `logger_history.py:233`: docstring `"...for backward compat"` on a conversion method.
- `modelmodel.py:347`: branch labeled "for backward compatibility".
- `toollib/networkprediction.py:736,774`: two backward-compat hints in comments.
- `toollib/models.py:48`: `nb_inputs` / `nb_outputs` inclusion "for backward compatibility".

Suggested action: delete the entire backward-compat surface in one pass and
update the very small number of call sites.

### 3. Dead-ish / legacy code paths in `run_replay.py`
`run_replay.py` still carries a `_replay_from_pkl` + `_dispatch_replay_legacy`
branch labeled "Legacy pkl replay - degraded". If the DB-based replay is now
the only supported path (which the recent refactors suggest), the pkl path can
be deleted - it's ~80 lines of dead-on-arrival branching.

### 4. Commented-out code blocks
`configs/designs/all_networks.yaml` has a ~10-line commented-out alternative
config. Either delete or move to documentation. Same pattern exists in a few
other YAMLs (`autofig_pred_combined.yml`).

### 5. Comments still trivially restating code
Even after this pass, several places remain with low-value comments. A quick
follow-up pass should target patterns like:
- `# default batch apply (for backward compatibility)` (modelmodel.py:347)
- `# Compat aliases` (was removed but the method bodies remain)
- ASCII separator banners in larger files (`update_biocompdb.py`,
  `toollib/networkprediction.py`, `toollib/figuremakers/*.py`).

### 6. `repomix-output.txt` (632 KB) checked into repo
`repomix-output.txt` is 632 KB of LLM-context dump. Not sensitive but bloats
the published tree. Add to `.gitignore` (or `.repomixignore` already handles
generation) and `git rm` it before publishing.

### 7. `tuner_app/` vs `tuner_app_react/` directories
Both exist. `tuner_app_react/` is `.gitignored`, but `tuner_app/` is present.
Worth checking whether one is dead. (Not investigated this pass.)

### 8. README is one paragraph
The README is fine as an internal placeholder but is light for a public release.
Add: install instructions, minimal "hello world" example, link to
`biocomp-doc/`, citation block, license badge.

### 9. Test coverage on changed files
The LLM-touched core (`step_writer.py`, `history_db.py`, `logger_runner.py`,
`logger_dispatch.py`) has tests in `tests/` but I did not run them.
**Run `pytest biocomptools/tests/` before tagging the release** to confirm the
docstring/comment cleanup didn't accidentally truncate code.

### 10. `tmp/` was untracked but had `/Users/jeandisset/...` paths in dumps
Confirmed: `tmp/` is not in `git ls-files`, but I added it to `.gitignore`
explicitly so it stays that way. Same with `runner_logs/`.

## Verification

- All 149 `.py` files parse cleanly after edits (verified via `ast.parse`).
- 167 unicode LLM-isms removed; 16 remain (all in `interactive_link.py` HTML/JS
  string literals - intentional).
- **pytest run: zero regressions.** Diffed failure sets before vs after the
  cleanup: identical 21 failures, identical 256 passes (after skipping 3 tests
  blocked by an unrelated bug in sibling `biocomp/biocomp/metric_utils.py:519`).
  The 21 pre-existing failures are:
  - `test_mvp_data.py` (14 tests): `NetworkPrediction.get_network_stats(network_idx=i)` returns a list but `measuredvspredicted.py:142` calls `.get(...)` on it.
  - `test_design_heatmap_logger.py` (3 tests), `test_design_logged_vs_committed.py` (1), `test_figuremakers.py` (2), `test_logger_scheduling.py` (1) - stale fixtures / unrelated drift.
  - All also fail on `HEAD` without the cleanup commits.

## Files modified (summary)
- **New:** `LICENSE`, `PRE_RELEASE_AUDIT.md` (this file).
- **Touched (149+):** every `.py` under `biocomptools/` gained an SPDX header.
- **Substantively edited:** `configs/default.yaml`, `download_mlflow.py`,
  `configs/plots/autofig_combined.yml`, `pyproject.toml`, `.gitignore`,
  `step_history_triage.py`, `step_writer.py`, `history_db.py`,
  `logger_runner.py`, `logger_dispatch.py`, `run_replay.py`,
  `toollib/interactive_link.py`.
