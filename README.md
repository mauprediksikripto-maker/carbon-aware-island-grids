# Carbon-aware scheduling on coal-dominated island grids - code, data and results (v1.0.0)

This record contains the simulation code, input data and complete results for the article:

> D. Irawan, Sudarmaji, I. Samsudin, *Model-based marginal emission signals for carbon-aware scheduling on
> coal-dominated island grids: a computational regime map along Indonesia's transition pathway*, submitted to
> the Journal of Computational Science (2026).

**Authors**

| Name | ORCID | Affiliation | Contact |
|---|---|---|---|
| Dedi Irawan (corresponding author) | 0009-0007-4973-926X | Faculty of Computer Science, Universitas Muhammadiyah Metro, Indonesia | dedimti@ummetro.ac.id |
| Sudarmaji | 0009-0005-5549-4886 | Faculty of Computer Science, Universitas Muhammadiyah Metro, Indonesia | |
| Imam Samsudin | 0009-0006-2482-4209 | Faculty of Computer Science, Universitas Muhammadiyah Metro, Indonesia | |

## What the model does

An hourly merit-order dispatch model of Indonesian island grids (Jawa-Bali, Sumatera, South Sulawesi):

- must-run renewables first;
- coal with a minimum stable level, then CCGT, then peakers;
- a take-or-pay gas block on Jawa-Bali;
- curtailment.

The model is calibrated to the official 2024 energy mix. From it we derive two hourly signals, average carbon intensity and the marginal emission factor (MEF). Schedulers for delay-tolerant data-centre jobs follow each signal, and each schedule is evaluated by consequential accounting: the job load is added to demand and the grid is re-dispatched. The analyses are:

- Sobol global sensitivity analysis;
- a 252-state regime map (2,520 runs over 10 seeds);
- calibration age;
- ablation of the marginal signal;
- PPO transfer;
- an emission-factor envelope;
- load-shape robustness;
- the dispatch mechanism;
- slack sensitivity.

## Contents

```
README.md                         this file
LICENSE                           MIT licence (code)
LICENSES/MIT.txt, CC-BY-4.0.txt   licence texts (code: MIT; data and results: CC BY 4.0)
CITATION.cff                      citation metadata
.zenodo.json                      Zenodo deposit metadata
requirements.txt                  Python packages
SHA256SUMS.txt                    SHA-256 checksum of every file in this record
code/
  carbon_aware_v10_p2b.ipynb      the notebook that produced every result (run v10_p2b)
  carbon_aware_v10_p2b.py         the same notebook as a percent-format script (# %% cells)
  build_notebook.py               converts the .py into the .ipynb
data/
  p2b_jawa_bali_2024_load_shape.json   digitized Jawa-Bali load shape (as used by the code)
  p2b_daily_load_curves_2024.csv       hourly curves: weekday, Saturday, Sunday
  p2b_weekly_peak_2024.csv             weekly peak load, weeks 1-53 of 2024
  p2b_holidays_2024.csv                holidays and their treatment
results/
  run_metadata.json               configuration, stage timestamps, SHA-256 of every result file
  tables/                         32 CSV tables
  figures/                        9 figures, each as vector PDF and PNG
  raw/                            Sobol designs and per-run outputs
```

## How to reproduce

1. Open `code/carbon_aware_v10_p2b.ipynb` in Google Colab (CPU runtime) or Jupyter.
2. Run all cells. In Colab the notebook mounts Google Drive and writes to `MyDrive/jocs_pilot/v10_p2b/`. Elsewhere it writes to `./jocs_pilot/v10_p2b/`.
3. A complete run takes about two hours of CPU time. Each session is limited to `MAX_RUNTIME_MIN = 120` minutes. If it stops or disconnects, run all cells again: finished stages are skipped and partial stages resume.
4. When every stage is finished, the notebook writes a ZIP of all results and `run_metadata.json` with SHA-256 checksums.

`SALib`, `gymnasium` and `stable-baselines3` are installed by the notebook itself. The `.py` file contains an IPython `!pip` line, so run it as a notebook (Jupyter or VS Code), not with `python`.

The real European traces are downloaded at run time from a pinned commit of `HewlettPackard/sustain-cluster` (commit `3f6ea95cb835b89ba50b0ef76d66d14b8037643e`), so the run needs internet access.

### Determinism

Seeds are fixed (`SEED = 42`). The full run and an independent reduced local run produced byte-identical files for these tables:

- calibration check (table 0);
- consequential savings with reserve (table 5b);
- dispatch mechanism (table 19);
- slack sensitivity (table 20).

PPO results differ slightly between platforms, because PyTorch training is not bit-reproducible across hardware.

### Software versions

The Colab run did not record package versions. The versions of the reduced local verification run were:

- Python 3.13;
- NumPy 2.5.3;
- pandas 3.0.5;
- Matplotlib 3.11.2;
- SALib 1.6.0;
- Stable-Baselines3 2.9.2;
- Gymnasium 1.4.0;
- PyTorch 2.15.0.

## Notebook release edits

The code cells are exactly those of run `v10_p2b` (25 Nov 2026). For this release only the explanatory text cells were edited:

- the header was rewritten to describe the article and the data;
- internal development labels were removed from section titles.

Some development history remains in the code itself:

- Section numbers follow the order in which stages were added; there is no section 9, and the provenance section is numbered 15 and placed last.
- Output folder names (`jocs_pilot`) are kept unchanged.

## Results: file guide

All savings are percentage changes in system emissions relative to carbon-agnostic scheduling; positive = emissions avoided. `avgCI` = scheduler following the average carbon intensity; `MEF` = scheduler following the model's marginal emission factor.

| File | Content |
|---|---|
| `tables/grid_summary.csv` | Mean, SD and daily range of hourly CI, mean MEF, unserved hours per grid |
| `tables/table0_calibration_check.csv` | Simulated vs official 2024 energy mix (and Jawa-Bali load factor) |
| `tables/table1_heuristic_savings.csv` | Heuristic policies under average and MEF accounting |
| `tables/table2_ppo_transfer_*.csv` | PPO trained on the European-style grid, tested on Indonesian grids (per seed, summary) |
| `tables/table3_solar_sweep.csv` | Solar build-out sweep on Sumatera |
| `tables/table4_sobol_indices.csv` | Sobol S1/ST, MEF-sum accounting |
| `tables/table5_consequential.csv`, `table5b_consequential_reserve.csv` | Consequential savings at 0.5, 2 and 5 % data-centre load (5b: peaker capacity ×3, no unserved energy) |
| `tables/table6_misspecification*.csv` | MEF planner with mis-specified grid parameters |
| `tables/table7_official_ef_check.csv` | Model emission rates vs official 2019 grid emission factors (operating margin) |
| `tables/table8_sobol_consequential.csv` | Sobol S1/ST, consequential accounting |
| `tables/table9_ablation*.csv` | Ablation of the marginal signal (no coal minimum, wrong merit order, 24-h profile, net load) |
| `tables/table10_regime_map.csv` | Regime map: 2 grids × 2 merit orders × coal minimum × solar build-out (252 states) |
| `tables/table11_calibration_age.csv` | Recalibrated, one-step-lag and stale MEF along the solar path |
| `tables/table12_sobol_convergence.csv` | Sobol indices at N = 256, 512, 1024 |
| `tables/table16_ef_envelope*.csv` | Headline results over the emission-factor envelope allowed by official data |
| `tables/table17a_*`, `table17b_*`, `table17c_*` | 10-seed replicates of the consequential table, regime map (2,520 runs) and calibration age |
| `tables/table18_load_shape*.csv` | Load-shape variants (empirical, flat season, coal-capacity bounds, analytic shape; Sumatera daily-swing sweep) |
| `tables/table19_mechanism.csv` | Marginal unit class and mean ACI/MEF by hour of day |
| `tables/table20_slack.csv` | Sensitivity to the jobs' slack distribution |
| `raw/sobol_X.npy` | Sobol input design (4,096 × 6) |
| `raw/sobol_runs_*.csv`, `raw/sobolC_runs_*.csv`, `raw/sobolC_ext_runs_*.csv` | Per-run Sobol outputs (MEF-sum, consequential, convergence extension), 4,096 rows each |
| `figures/fig1`–`fig9` (PDF + PNG) | CI week, solar sweep, Sobol (two accountings), regime map, calibration age, mechanism, regime-map replicates, daily swing |

## Data sources and licences

- **Code:** MIT licence.
- **Results** (`results/`) **and the files in `data/`:** CC BY 4.0. Cite the article and this record.
- **Statistik PLN 2024 and ESDM grid emission factors** (public): calibration targets and fuel-use data are hard-coded in section 1 of the notebook with table references.
- **PT PLN (Persero) UIP2B Jawa Madura Bali, *Statistik Sistem Jawa Madura Bali 2024* (2025):** an internal statistical report of the Jawa-Bali dispatch centre without a public URL. The report itself is not included. `data/` holds only values the authors digitized from its figures (pp. 29–33 and 35):
  - half-hourly load curves of the 2024 peak weekday (9 Oct), Saturday (12 Oct) and Sunday (22 Sep);
  - the weekly peak curve;
  - holiday peaks.

  Digitization was done at 300 dpi; the digitized daily peaks lie within 0.3–0.9 % of the printed values. Curve values are load at hh:30 divided by the day's maximum. The MW columns in `p2b_daily_load_curves_2024.csv` are these fractions multiplied by the printed daily peak (32,759, 30,450 and 28,699 MW); they are not printed values. Cite the UIP2B report as the source of these data.
- **Electricity Maps 2024 hourly carbon intensity, Germany and Great Britain** (ODbL 1.0): not redistributed here. They are downloaded at run time, and Electricity Maps must be credited as the source.
