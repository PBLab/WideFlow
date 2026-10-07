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

## Two-machine sync
See `WideFlow_data_analysis/CLAUDE.md`'s "Two-machine sync convention" section
(canonical) — applies to this repo too: real paths/hosts go in gitignored
`wideflow/config_local.py`, never a literal in tracked code, and IDE/build
artifacts stay untracked.

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

### Raw ↔ corrected mapping (verified against the raw tifs, 2026-10-06)
- Corrected frame `i` = raw frames `2i` (blue) + `2i+1` (violet), from the first frame
  of each part (no offset). **Corrected index = raw index // 2.**
- Which raw frame each stored `rois_traces` channel comes from:

  | Channel | Raw frame |
  |---|---|
  | `channel_2` (blue ΔF/F), `channel_3` (blue raw), `channel_0` | `2i` |
  | `channel_7` (violet ΔF/F), `channel_1` (the violet the hemo correction subtracts) | `2i+1` |
  | `channel_5` (violet raw) | **`2i−1`** (see the warning in "Results H5 structure") |
- Raw tifs: `wf_raw_data_N.tif`, 2000 pages each (the last file holds the remainder).
  **Only the last file's name is space-padded** (`wf_raw_data_  9.tif`,
  `wf_raw_data_ 32.tif`). Raw frame `r` = file `r // 2000`, page `r % 2000` (0-based).
  Fiji counts slices from 1, so there blue is on **odd** slices.
- **Trial onsets after a hit fall on an odd raw frame:** the reward is on even frame
  `r`, the delay covers `r+1…r+200`, and the next trial starts at `r+201`. One
  corrected frame then holds the last delay frame and the first trial frame. Assign a
  corrected frame to the trial its blue frame (`2i`) is in. Timeout trials last
  1000 raw frames, so they start on even frames.
- **NF rewards (`cue==1`) fall on the even (blue) raw frame**, the frame where the
  metric crossed threshold. **Control rewards fall on either parity**, since they're
  probabilistic, not metric-gated. Both map to corrected frame `r // 2`.
- **The lick sensor varies at raw-frame resolution** (~2.5% of raw pairs differ),
  unlike the imaging data. A combined raster must either bin both at corrected
  resolution or label the lick trace as finer.
- **Each part is processed from scratch:** the ΔF/F rolling baseline and hemo running
  mean restart, and `diffN` is zero-padded, so a part's first N corrected frames
  aren't real differences.

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

   `base_path` comes from `wideflow/config.py` (imports the gitignored, per-machine
   `wideflow/config_local.py`'s `BASE_PATH` — see that file's `.example.py`
   template) — never hardcode a literal path here. A second path,
   `DATA_STAGING_PATH`, exists but is used for data transfer/staging from the rig —
   reading from it for analysis is very slow. **Do not read from the staging path
   for analysis.**

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
   - `timestamp` — software time (`perf_counter()`), not camera exposure time.
     **Convention changed between cohorts during the post-outcome delay** (every
     frame after a reward or timeout, until the next trial): cohorts ≤3.5 (sessions up
     to 2025-09-10) stamp each delay frame at the END of its acquisition-loop iteration,
     cohorts ≥4.0 (from 2025-12-03) at its START, like trial frames. Old-convention delay
     times are ~26-39 ms later. Signature: delay -> next-trial interval 0-1 ms (old) vs.
     ~39 ms (new). Frame order and counts are unaffected. Details and source:
     `WideFlow_data_analysis/docs/experiment_history.md` (2026-10-01).
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
and `Lena_scripts/calc_and_save_zscores_and_diff5_all_rois_with_MH_removal_per _roi.py`
— **not** from variable names or code comments, which are sometimes misleading (see
notes below). **If either of these scripts is changed to add/remove/rename what
gets saved into this h5 file, update this list to match.**

### Location and which files to use
Path: `BASE_PATH + '/Results/'` (`BASE_PATH` from `wideflow/config.py`, i.e. the
per-machine `config_local.py`; never hardcode it). The Results folder holds many
h5 files from earlier processing rounds. **Only these are the clean, current results:**

| Cohort | Files |
|---|---|
| 3.5 | `results_exp3.5_parc_ROI1.h5`, `results_exp3.5_parc_ROI2.h5` |
| 4.0 | `results_exp4_parc_ROI1.h5`, `results_exp4_parc_ROI2.h5` (no `.0` in the name) |
| 4.1 | `results_exp4.1_parc_ROI1.h5`, `results_exp4.1_parc_ROI2.h5` |
| 3.4 | `results_exp3.4_parc_ROI1.h5` only (no ROI2 file; 3.4 is excluded from most analyses anyway) |

The name must match `results_exp{cohort}_parc_ROI{1|2}.h5` exactly. A file with any
extra suffix (e.g. `results_exp3.5_parc_ROI1_before_repack_june2026.h5`) or any other
name is not clean. **Ask Lena before using any other file in Results/.**

Each file has one group per mouse, then one group per recording part, keyed by the
raw session name: `{date}_{mouse_id}_{raw_session_name}_{part}`. Each file contains
every session of the mouse (spont, CRC, NF1 onward, and NF_control where it exists),
whichever block it was trained in.

### ROI1 vs. ROI2 files: why two versions exist
Every training target is a merged pair of parcellation ROIs, stored under one merged
key (`roi_{roi1_name}` or `roi_{roi2_name}` from `mice_metadata.h5`, e.g. `roi_30_37`).
Each file is the full post-session processing run with one target merged:
- `_ROI1` file: ROI1's pair merged, ROI2's two ROIs kept separate.
- `_ROI2` file: ROI2's pair merged, ROI1's two ROIs kept separate.

This lets each target be analyzed in every session, including sessions where the
other ROI was the one being trained (e.g. ROI2's activity during ROI1's NF2–NF6).
Neither file can stand in for the other for the z-scored outputs:
- `rois_traces/*` (e.g. `channel_0`): a merged ROI's trace is exactly the
  pixel-count-weighted mean of its two halves (verified: weights sum to 1, zero
  residual). The two files hold identical traces for every ROI they share. You can't
  un-merge a pair, but you never need to, because the other file has the halves.
  **For these traces, either file gives the same answer.**
- `post_session_analysis_LK2/zsores_MH*`: **not** reconstructable. Each z-score is
  computed against the other ROIs at the same instant. Merging changes that pool:
  the merged ROI counts as one member instead of two, a different set of neighbours
  is MH-removed, and the top-15% count changes. Rebuilding the merged z-score from
  the other file's halves leaves errors up to ~28% of the signal's SD. **For a
  target's z-scored metric, use the file where that target is the merged ROI.** The
  separate halves' z-scores in the other file are not that target's metric.

Coupling caveat: during one target's training, that target is part of the pool the
other target's z-score is computed against. A rise in the trained ROI can lower the
other ROI's z-score without any change in the other ROI's own activity. Check
`channel_0`, which has no pool, alongside the z-score.

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
  **⚠️ One frame behind:** `channel_5[i]` is raw frame `2i−1` (the previous pair's
  violet), not `2i+1` like `channel_1`/`channel_7`. It's read from a different buffer.
  Don't compare `channel_3[i]` with `channel_5[i]` as if they were the same pair.
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

### `/{mouse_id}/{session_id}/post_session_analysis_LK2/*` (written by `calc_and_save_zscores_and_diff5_all_rois_with_MH_removal_per _roi.py`)
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

**`zsores_MH_diff10_exc_top15` vs. the replayed `metric_results`**: same core formula, but
differs in (1) MH/neighbor exclusion (off in `metric_results`, on here), (2)
top-15% exclusion algorithm (rank-based vs. percentile-based), (3) windowing (live:
circular buffer, warmed up before real data; offline: zero-padded from session
start).

**`zsores_MH_diff10_exc_top15` vs. the true live metric (`metadata.txt`
`metric_result`)** (checked 2026-10-06). Two differences, both fully accounting for
the gap (recomputing with the live rule and the session's live `closest_rois`
matches live to ~2×10⁻⁵ in all 12 NF mice of 3.5/4.0/4.1):
1. **Top-15% exclusion count:** live (`core/metrics/ROI_Diff.py`) excludes exactly
   `int(0.15·n)` ROIs; offline excludes everything ≥ the 85th percentile; n = ROIs
   left after neighbour removal. They disagree by one ROI for most pool sizes.
2. **Neighbour list:** live, the target's excluded neighbours (`closest_rois` in
   `session_config.json`) were **chosen manually by Lena from the parcellation
   image**. Offline, every ROI's neighbours come from `closest_dict_NEW_ROI{1,2}.h5`,
   generated automatically by `Lena_scripts/create_closest_dict.py`: two ROIs count
   as neighbours if more than one outline pixel of one sits exactly 1 apart (in
   flattened pixel index) from the other's outline, i.e. side by side in the same
   image row. ROIs touching mostly above/below or at a corner are missed. Known,
   small mismatch, accepted (Lena's decision). It shows up in block 1 (NF1–NF6),
   where 8 of 12 NF mice had 1–2 extra manual neighbours (258FL, 260FN, 261MR,
   263MRL, 276FL, 281MRL, 322MR, 329FRR); block 2 and NF_control match (except 258FL
   NF_control).

The pool itself matches: in each block the live session used the same ROI set as the
offline file for that block (ROI1 pair merged in block 1 = `_ROI1` file, ROI2 pair
merged in block 2 = `_ROI2` file). For 3.5/4.0 block 1, the live dictionary file had
a different name (`_NEW.h5` / `_NEW2.h5`, not reachable from the analysis machines);
that it had the same ROI1-merged set is inferred from the ~2×10⁻⁵ match.

- Typical size: ~1–3×10⁻² (z units); exact match (~2×10⁻⁵) only when both agree.
  Negligible for the metric as an activity trace, but noticeable near threshold:
  offline has ~14% more above-threshold frames, and at 0.7% of NF reward frames
  offline is just below threshold. **Never mix live and offline values in one
  comparison** (see `WideFlow_data_analysis/CLAUDE.md`).
- Effect on threshold calibration (30% replay on each NF1 part, thresholds averaged
  over the 3 parts): the chosen thresholds are ~0.07 z (~2%) too high (median; 281MRL
  ROI2 ~0.2 z, 277FRL ROI2 ~0.6 z on its larger scale). That's about half the
  uncertainty of the 3-part average (~0.13 z), always in the same direction, so the
  live success rate at the chosen threshold is ~2 points lower than intended on average.

## ⚠️ Blue/violet channel swaps (frame errors at acquisition)
Occasionally the raw frame sequence has an extra or missing frame (two consecutive
frames of the same wavelength), shifting the blue/violet alternation. After that
point, "blue" is violet and vice versa, so `channel_0` and every metric built on it
are wrong. **The live NF metric was swapped too** (the live and saved frame sequences
are identical), so rewards in those stretches came from a swapped computation.

**Sustained swaps (to the end of the part); excluded whole from every analysis** (decided 2026-10-06):
| Part | Swapped from corrected frame (±1) to end |
|---|---|
| `20260609_327FL_NF2_p1` | 5910 |
| `20260618_331FN_NF9_p3` | ~116 |
| `20260620_329FRR_NF11_p3` | ~55 |
| `20251202_277FRL_spont_p2` | ~189 |

**🚨 `277FRL spont p2` must never be used as part of a spontaneous baseline**,
e.g. when comparing brain activity before vs. after NF training against the spont
recording. Almost the entire part is swapped.

**Brief events** (back in sync within ≤17 corrected frames) in 10 parts of cohorts
3.5/4.0/4.1: exclude the event + 75 corrected frames after it (the disturbance
echoes through the 32-frame ΔF/F baseline and hemo running mean). None affected a
reward. **Cohort 3.4 has more events of both kinds that were NOT fully tracked**:
check it before any 3.4 imaging analysis.

Full list, method and numbers: `WideFlow_data_analysis/docs/results_channel_swap_check.md`.

**New cohort → screen it for swaps** before analysing its imaging data (procedure in
`WideFlow_data_analysis/docs/results_channel_swap_check.md`, "Screening a new cohort").

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
