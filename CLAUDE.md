# WideFlow — Project Notes

## Session bootstrap
Before doing any analysis work (reading sessions, computing metrics, etc.), also
read `WideFlow_data_analysis/README.md` and `WideFlow_data_analysis/CLAUDE.md` in
full — they hold the current experimental design, which is needed alongside this
file's data/pipeline conventions to correctly locate and interpret session data.
`WideFlow_data_analysis/docs/experiment_history.md` has the full change history but
isn't needed for day-to-day work — only consult it when specifically pointed there.

Real-time closed-loop neurofeedback system for wide-field calcium imaging in mice,
plus the offline analysis pipeline built on top of acquired sessions.
Full architecture/hardware/config details: see `README.md` at repo root.

## Repo Layout (as observed)
- `wideflow/core/` — session / pipeline / process abstractions (the live acquisition-feedback loop)
- `wideflow/Imaging/` — camera + visualization code, imaging configuration templates
- `wideflow/devices/` — hardware controllers (arduino/, mock_devices/ for testing without hardware)
- `wideflow/DeepLabCut/` — optional behavioral pose-tracking integration
- `wideflow/analysis/` — analysis utilities
- `wideflow/Lena_scripts/` — Lena's offline analysis scripts across mice/sessions (learning curves,
  threshold-crossing counts, correlation maps, PSTR plots, cross-mouse stats). Includes cached
  intermediate data (`x_data_*`, `y_data_*` per mouse/session type) — treat these as derived/regenerable,
  not source data.
- `figures/` — plotting scripts for papers/figures; `LK_*` prefix = Lena's versions of shared scripts
- `data/cortex_map/` — Allen atlas ROI maps and per-mouse functional parcellation data
- `untracked_files/` — scripts deliberately kept outside git tracking
- `test_rsync1/` — looks like leftover rsync test output (duplicate images) — confirm before touching/removing

## Environment
- **`WideFlow1`** (conda env, `~/.conda/envs/WideFlow1`) is the confirmed working environment — every
  top-level `wideflow` subpackage (`core`, `Imaging`, `devices`, `DeepLabCut`, `GUI`, `analysis`, `utils`,
  `Lena_scripts`) imports cleanly under it (verified 2026-08-02). Treat as the default env for this project.
- **`environment.yml`** (repo root) is a direct export of this actually-verified `WideFlow1` env
  (`conda env export`'s own pip-freeze step silently drops most pip packages on this env — this file's
  pip section was built from `conda run -n WideFlow1 pip freeze` instead, added 2026-09-06). Prefer this
  over `requirements.txt` (see below) when recreating the environment elsewhere. It deliberately excludes
  `wideflow` itself — install that separately as an editable install pointing at your local checkout.
- `venv/` (2023-01-02, Python 3.8) was deleted 2026-08-02 — investigation found it was a broken venv
  copied in from another account (`wfield`, the machine that runs the physical rig) around that date,
  never successfully used since. Confirmed dead, removed.
- `.venv/` (Python 3.10) still exists — see TODO below, revisit ~end of Oct 2026.
- `requirements.txt` at repo root may be stale/unreliable — `WideFlow1` was verified by actually importing
  the code, not by trusting this file. A second `requirements.txt` exists under `Lena_scripts/`.

## TODO — revisit `.venv/` (~end of Oct 2026)
Investigated 2026-08-02: `.venv/` (created 2025-04-28) looks like it was set up reactively while configuring
PyCharm for a student (`linoyshvartz`) on this project — evidence: her account's own `.pyc` cache files in
`wideflow/utils/__pycache__/` dated 2025-12-03 15:41, and a matching one under Lena's own account 17 min
later, both compiled via `.venv`'s Python 3.10. A large batch of packages (numpy, pandas, scipy stack,
matplotlib, seaborn, h5py, statsmodels, pingouin, patsy, joblib, mat4py, `cupy-cuda114`, PyQt5) was
pip-installed into it piecemeal that same day — consistent with hitting `ModuleNotFoundError`s while
running scripts together and installing whatever was missing, not a deliberate separate environment.
`WideFlow1` (conda) remains the confirmed working env throughout.

**Action once the student has finished her project**: re-run the investigation (check `.venv/` and
`wideflow/__pycache__/` file timestamps + ownership for any activity since 2025-12-03) to see whether it's
been used again, and by whom, before deciding to delete it (same treatment as `venv/` above).

**Once `.venv/` is deleted: delete this whole "TODO — revisit `.venv/`" section, including this line.**

## Conventions Observed
- Experiment-specific scripts in `Lena_scripts/` are prefixed by experiment code (`EXP2.1`, `EXP3`,
  `EXP3.4`, `EXP4`, etc.) — [meaning of each experiment code: fill in]
- `LK_` prefix marks Lena's variant of a shared analysis/figure script
- Cortical area abbreviations seen in `Lena_scripts` comments/data (`BC`, `RC`, `M1`, etc.) are
  study-specific — defined in `WideFlow_data_analysis/README.md`'s "Terminology" section, not here.

## Frame counting convention — raw frames vs. hemo-corrected frames
The system images two wavelengths (405/450nm) alternately, 30ms exposure each,
combined into one hemodynamic-corrected sample per pair, at a combined ~13Hz rate.
This creates two different frame counts for the "same" data, and code/scripts are
inconsistent about which one they use:
- **Raw frames (no hemo correction)**: one per single-wavelength exposure.
  `metadata.txt` (written live during acquisition) logs one line per raw frame — so
  its timestamp/metric/cue columns are at raw-frame length, with each real
  hemo-corrected sample's value duplicated across its 2 consecutive raw-frame lines.
- **Hemo-corrected frames**: one per combined/corrected sample — half the raw-frame
  count. Post-hoc analysis outputs (e.g. h5 z-score results) are typically stored at
  this length.

**Whenever a frame count appears in code, data, or discussion, explicitly check and
state whether it means raw frames (no hemo correction) or hemo-corrected frames —
don't assume.**

## Mice metadata pipeline (`Lena_scripts/mice_meta/`)
Terminology note: the Excel/H5 field called `group_number` (e.g. `3.4`, `4.0`, `4.1`)
is what we call a **cohort** (mice that ran the experiment together) in conversation
and documentation — not to be confused with `group`/`exp group`, which is the NF vs.
control condition assignment. Both fields say "group," which is easy to conflate.
(Renaming `group_number` itself, in the Excel/H5/code, was considered and deferred —
it would cascade through `excel_reader.py`, `h5_store.py`, `mouse_metadata_loader.py`,
and the `results_exp{group_num}_...` results-file naming pattern, plus require
migrating the field name in the already-populated `mice_metadata.h5`.)

Per-mouse metadata (group, ROI names/cortical areas/thresholds, session dates/parts)
lives in an Excel sheet (in OneDrive, fetched locally via `rclone`), synced into
`mice_metadata.h5` by `sync_excel_to_h5.py` (new mice auto-written; changed/removed
mice require interactive confirmation, so a typo in Excel can't silently clobber
good data). Downstream scripts (e.g. `SR_from_mice_meta/`) read from the H5, not the
Excel directly.

Session columns are fixed and ordered: `spont`, `CRC`, `NF1`...`NF7`, `NF_control`,
`NF8`...`NF14`. Per-mouse, per-session overrides live in a `part_exceptions_and_skips`
cell:
- `skip_gap` — reserves a position in the sequence but no session is emitted (a real
  gap in that mouse's numbering).
- `skip_shift` — drops that session's date entirely and shifts every *later
  NF-numbered* session back one slot to fill the gap (e.g. if `NF7` is `skip_shift`,
  `NF8`'s real data becomes displayed as `NF7`). The H5 stores both the display name
  (`sessions/names`) and the real source column (`sessions/raw_names`) — always use
  `raw_names` for file/results lookups, `names` for display/plot labels.
- `skip_shift` **never** applies to `NF_control` (or `spont`/`CRC`) — it has its own
  fixed column and is never a shift source or target at this layer. (A *separate*,
  display-only remapping of `NF_control` into another slot, e.g. for cross-mouse
  plotting, exists independently in individual analysis scripts — not part of this
  sync pipeline.)

## Locating a session's raw metadata.txt (mouse + session name → file)

Chain to get from `(mouse_id, session_name)` to that session's raw per-frame log:

0. **Resolve a mouse number to its full ID** (skip if already given the full ID —
   it's used as-is in step 1): mouse IDs are the number, followed by sex (`M`/`F`),
   followed by an ear-labeling code — some combination of `R`/`L` (e.g. `RL`, `RR`,
   `LL`), or `N` for no labeling. Numbers are unique across all mice (no two mice
   share a number with different suffixes), so if only given a number, match it
   against `mice_metadata.h5`'s top-level keys (e.g.
   `[k for k in h5py.File(path)]` filtered by numeric prefix) to get the full ID.

1. **Look up the date/parts**: `load_mouse_metadata(mice_metadata_h5_path, mouse_id)`
   (`Lena_scripts/SR_from_mice_meta/mouse_metadata_loader.py`) returns each
   session's `date`, `raw_session_name`, and `parts` (e.g. `['p1','p2','p3']` if
   the session was split across multiple recordings, `[]` otherwise). Match on
   `raw_session_name`, not the display `name` (see mice metadata pipeline section
   above).

2. **Build the path** — verified directly on disk, no loader code for this step.
   Path is `` base_path/date/mouse_id/date_mouse_id_raw_session_name[_part]/metadata.txt ``,
   e.g. `.../20260809/231ML/20260809_231ML_NF1/metadata.txt` (underscores join the
   three name parts; `_part` is only appended when `parts` is non-empty).

   `base_path = /claustrum-storage/pblab_shared_data/Lena/WideFlow_prj` — use this
   for all analysis reads. A second mount, `/data/Lena/WideFlow_prj`, exists but is
   used for data transfer/staging from the rig — reading from it for analysis is
   very slow. **Do not read from `/data` for analysis.**

   If `parts` is non-empty, there is **one folder+file per part** — not one
   combined file.

   **Gotcha**: `trial number` resets to 1 at the start of each part (while
   `timestamp` stays continuous across parts, since the session clock keeps
   running). This is why the established convention computes any per-trial or
   per-session metric **separately per part, then averages across parts** to get
   one value per session (see `EXP3.4_succes_rates_over_sessions_metadata_metric.py`,
   which reshapes to `(mice, sessions, 3 parts)` and does `.mean(axis=2)`) —
   rather than concatenating parts and renumbering trials. Unweighted averaging
   assumes parts are equal-sized (verify trial/frame counts if not) and it
   collapses within-session structure, so it's the wrong tool for any analysis of
   trends *within* a session (learning/fatigue/drift over time).

3. **Parse the file**: `extract_from_metadata_file(path)`
   (`analysis/utils/extract_from_metadata_file.py`) skips the header (a config
   dump: `base_path`, `mouse_id`, `session_name`, camera/serial/behavioral/DLC/
   acquisition/feedback config) up to the `frames metadata:` marker line, then
   parses each subsequent line into 6 parallel lists, at raw-frame length (see
   "Frame counting convention" above):

   **Added 2026-08-12**: `path` may also be an already-open text stream instead
   of only a file path. This lets a caller read the file's raw bytes once (e.g.
   to hash them for a reproducibility checksum), then reuse that same
   in-memory content for parsing — instead of opening and reading the file
   twice. Existing callers passing a path string are completely unaffected.
   See `WideFlow_data_analysis/IO/metadata_loader.py`'s `get_session_frames`
   for the pattern (hash raw bytes → wrap as `io.StringIO` → pass to
   `extract_from_metadata_file`).
   - `timestamp`
   - `cue` — reward indicator: `1` on the frame a reward was actually delivered
     (metric crossed threshold and feedback fired), `0` otherwise. Not a
     stimulus-onset cue — it marks reward delivery.
   - `metric_result`
   - `threshold`
   - `serial_readout` — lick sensor, **inverted**: raw `1` = no lick, `0` = lick.
     Every existing analysis script flips it (`1 - serial_readout`) before use —
     do the same in any new code, don't use the raw value directly.
   - `trial_number` — **the last trial in any session/part can be cut short**
     (recording ends mid-trial), so it should not be counted. Use
     `max(trial_number) - 1` as the trial count, not the raw max.
     When computing rewards/trials, count **all** `cue==1` events unfiltered —
     don't exclude a reward that happens to fall on the excluded last trial.
     This matches the established convention in
     `Lena_scripts/SR_from_mice_meta/success_rate_core.py` (`rewards =
     np.sum(cue)`, unfiltered). In that script's simulated trial logic a reward
     always starts a new trial number, so the excluded last trial essentially
     never carries one; for real recorded `trial_number`, the write-time logic
     isn't in this repo to fully verify the same guarantee holds, but it's the
     precedented choice, and the risk of a reward landing there is small.

## Results H5 structure (`results_exp{cohort}_parc_ROI{1|2}.h5`)
Verified by tracing the actual computations in `untracked_files/post_session_procedure.py`
and `Lena_scripts/calc_and_save_zscores_and_diff5_all_rois_with_MH_removal_per_roi.py`
— **not** from variable names or code comments, which are sometimes misleading (see
notes below). **If either of these scripts is changed to add/remove/rename what
gets saved into this h5 file, update this list to match.**

### `/{mouse_id}/{session_id}/rois_traces/channel_N` (written by `post_session_procedure.py`)
Per-ROI mean values, one array per session, at each pipeline stage. ΔF/F throughout
uses a **rolling-minimum** baseline (not mean), over the current buffer window.

- `channel_0`: final, fully hemo-corrected blue-channel ΔF/F — (raw blue ΔF/F) −
  (violet ΔF/F rescaled by a per-pixel linear regression) − (rolling mean of that
  difference over the buffer window). **This is "the" calcium signal used
  everywhere downstream.**
- `channel_1`: the regression-rescaled violet signal (`a·violet_dff + b`) — the
  *predicted hemodynamic contamination*, in blue-channel-equivalent units. Not raw
  violet.
- `channel_2`: blue ΔF/F *before* hemodynamic correction.
- `channel_3`: raw blue fluorescence (masked/warped pixel intensity, before any
  ΔF/F normalization).
- `channel_5`: raw violet fluorescence (same as channel_3, violet channel).
- `channel_7`: raw violet ΔF/F *before* regression rescaling.
- `channel_4`, `channel_6`: blue/violet rolling-minimum F0 baselines — **computed
  every frame but never saved** (the save code is commented out).

Naming trap: Python variables are off-by-one from h5 group names
(`rois_traces_ch1`→`channel_0`, `rois_traces_ch2`→`channel_1`).

### `/{mouse_id}/{session_id}/metric_results` (written by `post_session_procedure.py`)
The replayed real-time NF metric — for each ROI, 10-frame-back diff on `channel_0`;
exclude top 15% of ROIs by that diff value (exact-count, `argpartition`); z-score
the target ROI against the remaining 85%'s mean/std. In this reprocessing script,
**neighbor/MH exclusion is explicitly disabled** (`closest_rois=[]`). Stored at raw
frame length (no hemo correction — see "Frame counting convention" above), each
real value duplicated across 2 consecutive raw-frame slots (same pattern as
`metadata.txt`).

**⚠️ Never use `metric_results` in any further analysis code.**

### `/{mouse_id}/{session_id}/post_session_analysis_LK2/*` (written by `calc_and_save_zscores_and_diff5_all_rois_with_MH_removal_per_roi.py`)
Starting from `channel_0`, with the candidate target ROI's spatial neighbors
deleted from the pool ("MH removal"):

- `diff5`/`diff10`/`diff20`/`diff30`: raw N-frame temporal difference. Zero-padded
  at the start, so **the first N timepoints aren't real diffs** (value minus an
  implicit zero).
- `zsores_MH`: **not** a per-ROI-across-time z-score — it's per-timepoint,
  cross-ROI (at each instant, normalize each ROI against all ROIs' values *at that
  same instant*). Answers "is this ROI elevated relative to the rest of the brain
  right now," not "relative to its own baseline." Same axis as the live metric;
  correct for the NF task's purpose, but a trap if reused for something expecting
  per-ROI temporal normalization.
- `zsores_MH_diff5`/`diff10`/`diff20`/`diff30`: same cross-ROI z-score, applied to
  the diffed traces.
- `zsores_MH_diff{5,10}_exc_top15`: same, but also excluding the top 15% of ROIs
  (by value, per-timepoint) from the mean/std baseline — via `percentile(col,85)` +
  `<`, a **different algorithm** than the live metric's exact-count `argpartition`
  exclusion (usually close, not guaranteed identical).
- `zsores_MH_diff10_exc_top15_NEW`, `_NEW2`: groups exist in the file but are
  **empty** — the code that would populate them is commented out.
- `zsores_MH_diff10_exc_top15_NEW4`: populated, but is an **exact duplicate** of
  `zsores_MH_diff10_exc_top15`.

**`zsores_MH_diff10_exc_top15` vs. live `metric_results`**: same core formula, but
differs in (1) MH/neighbor exclusion (off in `metric_results`, on here), (2)
top-15% exclusion algorithm (rank-based vs. percentile-based), (3) windowing (live:
circular buffer, warmed up before real data; offline: zero-padded from session
start).

## Documentation organization convention
Where new project information goes:
- **README.md** (in whichever repo the fact belongs to): current-state facts —
  project purpose, terminology, current experimental design once documented.
  Not history — write what's true now, not what changed from what.
- **CLAUDE.md** (in whichever repo the fact belongs to): behavioral conventions
  and data/code gotchas — things to always check or do, not facts that change with
  the experiment. Keep it lean, since it auto-loads every session.
- **`WideFlow_data_analysis/docs/experiment_history.md`**: the only place for
  experiment design history. The "Original proposal" section is a frozen record of
  the Jan 2025 proposal; whenever something changes, flag the outdated line inline
  with `⚠️ CHANGED — see Changelog <date>` and add a dated entry to the Changelog
  section describing old → new.
- **Canonical vs. pointer**: if a fact belongs to one repo's domain (e.g. WideFlow's
  acquisition system, data pipeline, results-file structure), write it once there,
  and add a short pointer from the other repo's CLAUDE.md — don't duplicate the
  content itself, to avoid drift.

## Open Questions / Fill In As We Go
- What EXP2.1 / EXP3 / EXP3.4 / EXP4 refer to (dates, manipulations, cohorts)
- Whether `Lena_scripts/` scripts are meant to become part of `analysis/` eventually, or stay exploratory
- Current in-progress work / active branch context
