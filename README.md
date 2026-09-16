# Arrival route network design for reducing noise-exposed population

Code and processed data for the manuscript

> *Arrival Route Network Design for Reducing Noise-Exposed Population in Terminal Maneuvering Area*

submitted to **Aerospace** (MDPI). The study optimises the convergence structure of
arrival routes at Shanghai Pudong International Airport (Runway 35, north-landing
configuration) with a bi-level framework: an outer simulated-annealing search over
convergence topologies and merge-point positions, coupled to an inner improved A\*
decoder whose edge cost carries the same population criterion as the outer objective.

## Contents

```
.
├── README.md                          this file
├── PROVENANCE.md                      data sources, licences, effective dates
├── requirements.txt                   Python dependencies and versions
├── export_reference_tables.py         regenerates the derived tables in data/
├── population_data_square.xlsx        296 x 296 population raster (model input)
├── test.py                            core model: L_AE acoustics, A* decoder, SA loop
├── baseline_compare.py                published-procedure baseline (N0, L0)
├── sa_temperature_scale.py            annealing-schedule calibration
├── multi_seed_beta090.py              independent repeated runs
├── exhaustive_topology_beta090.py     exhaustive enumeration of the topology space
├── plot_multi_seed_convergence.py     convergence figure
├── run_wn_trajectories.py             the three representative weight configurations
├── make_fig456.py                     converts the trajectory figures to PDF
├── r2_1_vertical_profile.py           fixed-factor A/B/C comparison (vertical vs horizontal)
├── w_n_sweep_decoupled.py             preference-weight sweep on a fixed decoder
├── analyze_tradeoff.py                dominance and knee-point analysis
├── r2_8_robustness.py                 single-factor and joint sensitivity experiments
├── diag_r2_8_infeasibility.py         diagnostic for the infeasible joint scenarios
├── make_fig7_r2_8.py                  threshold-sensitivity figure
├── viz_procedures.py                  published-procedure waypoints and baseline noise map
├── plot_population.py, map_base.py    population-density map
├── data/                              derived reference tables and data dictionary
└── result/                            per-run records underlying the reported tables
```

## Script to manuscript mapping

| Manuscript item | Script | Output |
|---|---|---|
| §4.1.3 Baseline $N_0$, $L_0$ | `baseline_compare.py` | console / `result/` |
| §4.1.4 reliability table | `multi_seed_beta090.py` | `result/multi_seed_beta090_per_seed.csv`, `_stats.csv` |
| §4.1.4 convergence figure | `plot_multi_seed_convergence.py` | `result/multi_seed_convergence.csv` |
| §4.1.4 schedule calibration | `sa_temperature_scale.py` | `result/t0_calib_final.csv`, `t0_acceptance_T150.csv` |
| §4.1.4 topology enumeration | `exhaustive_topology_beta090.py` | `result/exhaustive_topology_beta090.xlsx` |
| §4.2 three configurations | `run_wn_trajectories.py`, `make_fig456.py` | `result/traj_wN*.png` |
| §4.2 fixed-factor decomposition | `r2_1_vertical_profile.py` | `result/r2_1_vertical_profile.xlsx` |
| §4.3 preference sweep, knee | `w_n_sweep_decoupled.py`, `analyze_tradeoff.py` | `result/w_n_sweep_decoupled_per_run.csv`, `result/tradeoff_report.txt` |
| §4.4 sensitivity experiments | `r2_8_robustness.py`, `diag_r2_8_infeasibility.py` | `result/r2_8_E1..E5.csv`, `result/r2_8_robustness.xlsx` |
| §4.4 threshold figure | `make_fig7_r2_8.py` | `result/fig7_data.csv` |
| population map | `plot_population.py`, `map_base.py` | — |

## Environment

All results in the manuscript were produced with **Python 3.12** on a single
workstation (AMD Ryzen 7 6800H, 8 physical cores / 16 logical processors, 16 GB RAM).
Package versions are pinned in `requirements.txt`.

## Running

All scripts use paths relative to the repository root, so run them **from the root
directory**, with `population_data_square.xlsx` present:

```bash
pip install -r requirements.txt

# core model and baseline
python baseline_compare.py

# §4.3: preference-weight sweep (220 chains; the longest single experiment)
python w_n_sweep_decoupled.py
python analyze_tradeoff.py

# §4.4: sensitivity experiments (280 chains)
python r2_8_robustness.py
```

The sweep and sensitivity scripts use `multiprocessing` and by default start 12 to 14
worker processes; reduce `N_WORKERS` at the top of each script if your machine has
fewer logical cores. A single annealing chain of 150 iterations takes roughly
500 s on the reference machine.

To regenerate the derived reference tables in `data/`:

```bash
python export_reference_tables.py
```

## Derived reference tables

`data/` holds tables derived from the model inputs so that the geometry does not have
to be re-derived by hand. They are produced by `export_reference_tables.py` and contain
no AIP text, only the numeric waypoint coordinates used as model input.

| File | Contents |
|---|---|
| `data/entry_points_if_faf.csv` | The five entry points, the runway threshold, the FAF and the IF, in both geographic and model coordinates |
| `data/waypoints_published_procedures.csv` | Every waypoint of the five published arrival procedures, with coordinates and altitudes |
| `data/procedure_chains.csv` | The ordered waypoint chain of each published procedure |
| `data/data_dictionary.md` | Field definitions, units and conventions |

## Data and licence

See `PROVENANCE.md`. The population raster is derived from LandScan Global 2024
(CC BY 4.0, Oak Ridge National Laboratory); the arrival procedures are derived from the
Chinese AIP, AIRAC cycle 2606. Please keep the attribution statement when reusing the
population data.

Licences: the **source code** is released under the MIT licence (`LICENSE`); the
**data** under CC BY 4.0 (`LICENSE-DATA`).
