# Data provenance and licences

## 1. Population data

**Source.** LandScan Global 2024, Oak Ridge National Laboratory.

**Citation.**

> Lebakula, V.; Gonzales, J.; Stipek, C.; Tsybina, E.; Zimmer, A.; Nukavarapu, N.;
> Byeonghwa, J.; Reynolds, B.; Kaufman, J.; Fan, J.; Martin, A.; Buck, W.; Basford, S.;
> Faxon, A.; Meade, S.; Urban, M. *LandScan Global 2024*; Oak Ridge National Laboratory:
> Oak Ridge, TN, USA, 2024. https://doi.org/10.48690/1532445

**Licence.** Creative Commons Attribution 4.0 International (CC BY 4.0). Use, copying,
distribution, transmission and adaptation for commercial and non-commercial purposes are
permitted without restriction, provided that clear attribution of the source is given.

**Version used / access date.** The dataset identified by the DOI above, accessed on
30 May 2026.

**Processing applied.** The study region was cropped from the global raster at its native
30 arc-second (1/120°) resolution, over the rectangular window
29°31′ N–31°59′ N, 120°19′ E–122°47′ E. The window bounds coincide with LandScan cell
boundaries and span exactly 296 × 296 cells. **No aggregation, interpolation or
resampling was applied**; the population total of the window is preserved exactly. The
only transformation is that no-data cells are stored as zero.

`population_data_square.xlsx` in this repository is that 296 × 296 crop.

## 2. Arrival procedures

**Source.** Chinese Aeronautical Information Publication (AIP), published by the Air
Traffic Management Bureau of the Civil Aviation Administration of China.
**AIRAC cycle 2606, effective 11 June 2026** (10 June 2026, 16:00 UTC).

**What is distributed here.** Only a **derived table of waypoint coordinates and
altitudes** (`data/waypoints_published_procedures.csv`,
`data/procedure_chains.csv`), produced by `export_reference_tables.py`. **No AIP text,
chart or document is redistributed.** The five published procedures are SASAN, ANDONG,
LISHE, MATNU and DUMET arrivals.

## 3. Coordinate frame

The model uses a Cartesian frame (unit: km) in a **plate carrée** projection at a
constant scale of **120 km per degree on both axes**, with the origin at
29°31′15″ N, 120°19′15″ E. One grid cell is therefore 1 km × 1 km in the model.

**Known distortion.** At the mean latitude of the study region (30.75° N) a true
30 arc-second cell spans 0.80 km east–west and 0.92 km north–south, so the frame
overstates east–west distances by about 25% and north–south distances by about 8%.
Both the published procedures and the optimised networks are evaluated in this single
frame, so the reported improvement rates are internally consistent; the residual effect
depends on the difference in orientation mix between the two networks.

## 4. Entry-point numbering

The topology results in the manuscript are expressed with the indices E1–E5, which are
assigned by the numbering rule of Section 2.1.1 (ascending clockwise angle of the vector
from the entry point to the runway threshold, relative to the vector from the threshold
to the final merge point). The mapping to the named procedures is:

| Index | Entry point | Entry altitude (m) |
|---|---|---|
| E1 | MATNU | 5100 |
| E2 | DUMET | 4800 |
| E3 | LISHE | 6000 |
| E4 | ANDONG | 6000 |
| E5 | SASAN | 6000 |

Note that this order is **not** the order in which the procedures are listed in Table 2
of the manuscript, which is alphabetical by name. A bracket expression such as
`((1,2),(3,(4,5)))` therefore reads as `((MATNU,DUMET),(LISHE,(ANDONG,SASAN)))`.

## 5. Software

Python 3.12 with the packages pinned in `requirements.txt`. No external aviation or
noise software was used; the acoustic and optimisation models are implemented in this
repository.

## 6. Licences

Two licences are used, one per material type.

| Material | Licence | File |
|---|---|---|
| Source code (all `.py` files) | MIT | `LICENSE` |
| Data (`data/`, `population_data_square.xlsx`) | CC BY 4.0 | `LICENSE-DATA` |

The split is deliberate. The data carry an upstream attribution requirement (LandScan
Global 2024 is distributed under CC BY 4.0), which a permissive data licence preserves, whereas
Creative Commons licences are not recommended for software because they grant no patent
rights and their attribution requirement is awkward to apply per source file.

Reusers of the population raster must retain the LandScan attribution given in §1.
