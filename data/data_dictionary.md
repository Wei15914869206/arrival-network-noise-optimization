# Data dictionary

## Conventions used throughout

- **Length unit**: km. The model grid has a spacing of 1 km on both axes
  (plate carrée, 120 km per degree; see `PROVENANCE.md`).
- **Coordinates**: `latitude_deg` / `longitude_deg` are decimal degrees, WGS-84.
  `grid_x_km` / `grid_y_km` are the model Cartesian coordinates, measured from the
  origin at 29°31′15″ N, 120°19′15″ E with x increasing eastward and y increasing
  northward.
- **Altitudes** are in metres above mean sea level.
- **`N`** is the population exposed above the 65 dB single-event $L_{AE}$ threshold,
  counted once per grid cell. **`L`** is the total route length of the network.
  **`R_N = (N0 − N)/N0`** and **`R_L = (L0 − L)/L0`** are improvement rates against the
  published baseline (N0 = 6,169,767; L0 = 829.2 km); positive means improvement.
- **Seeds** are the integer seeds passed to `numpy.random.RandomState` for each
  independent simulated-annealing chain.

## `entry_points_if_faf.csv`

| Field | Meaning |
|---|---|
| `role` | `entry_point (E1..E5)`, `runway_threshold`, `final_approach_fix`, `intermediate_fix` |
| `name` | Waypoint name used in the manuscript (the runway threshold is P) |
| `latitude_deg`, `longitude_deg` | Geographic coordinates; empty for threshold/FAF/IF, which are defined directly in model coordinates |
| `altitude_m` | Entry altitude for entry points; empty for the other rows |
| `grid_x_km`, `grid_y_km` | Model coordinates |

**E-index mapping**: E1 = MATNU, E2 = DUMET, E3 = LISHE, E4 = ANDONG, E5 = SASAN.
The indices come from the numbering rule of Section 2.1.1 and are **not** the order in
which the procedures appear in Table 2 of the manuscript. See `PROVENANCE.md` §4.

## `waypoints_published_procedures.csv`

| Field | Meaning |
|---|---|
| `waypoint` | Waypoint name as it appears in the published procedure |
| `latitude_deg`, `longitude_deg` | Geographic coordinates |
| `altitude_m` | Published crossing altitude; empty where the source gives none |
| `used_by_procedures` | Semicolon-separated list of the procedures that pass through this point (shared points have more than one) |

## `procedure_chains.csv`

| Field | Meaning |
|---|---|
| `procedure` | Procedure label (`SASAN Arrival`, …) |
| `stop_order` | Position along the chain, starting at 1 at the entry point |
| `waypoint` | Waypoint name |

Chains run from the entry point to the runway threshold (P) and include the shared
convergence segment and the final approach segment. LISHE merges into the ANDONG chain
at SAMKI; DUMET merges into the MATNU chain at BEKOK.

## `../result/` — per-run records

| File | Contents |
|---|---|
| `multi_seed_beta090_per_seed.csv` | One row per independent run of the repeated-run study (30 chains) |
| `multi_seed_beta090_stats.csv` | Summary statistics of the same |
| `multi_seed_beta090_trace.csv` | Best-so-far objective per generation, per chain |
| `multi_seed_convergence.csv` | Aggregated convergence traces used for the figure |
| `exhaustive_topology_beta090.xlsx` | One row per enumerated topology, with feasibility and outcome |
| `t0_calib_final.csv`, `t0_acceptance_T150.csv` | Annealing-schedule calibration and acceptance profile |
| `r2_1_vertical_profile.xlsx` | Fixed-factor A/B/C comparison and the additive decomposition |
| `w_n_sweep_decoupled_per_run.csv` | One row per chain of the preference-weight sweep (11 weights × 20 chains) |
| `w_n_sweep_decoupled_summary.csv` | Medians and interquartile ranges per weight |
| `tradeoff_report.txt` | Dominance relations, the nondominated set and the knee-point computation |
| `r2_8_E1.csv` … `r2_8_E5.csv` | Per-run records of the sensitivity experiments (threshold; structural; algorithmic; population and acoustic; joint) |
| `r2_8_robustness.xlsx` | Consolidated sensitivity results |
| `fig7_data.csv` | Series plotted in the threshold-sensitivity figure |
