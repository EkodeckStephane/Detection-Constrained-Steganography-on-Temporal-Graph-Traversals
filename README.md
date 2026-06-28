# Detection-Constrained Steganography on Temporal Graph Traversals

This repository contains the reproducibility artifact for the work
**Detection-Constrained Steganography over Temporal Graph Traversals**.

The project studies a behavioral steganographic channel over temporal graphs.
Instead of changing graph topology, the encoder hides bits by selecting among
valid time-conditioned continuations of an observed traversal. The system uses
a causal cover model, explicit abstention, deterministic synchronization,
locally decodable coding, distributional audits, and layered steganalysis.

## Public Repository Scope

The public repository is intended to contain the executable artifact:

- `code/`: Python package, scripts, models, codecs, detectors, and tests.
- `experiments/`: frozen YAML configurations for the reported experiments.
- `results/tables/`: derived CSV/JSON result tables used by the paper.
- `results/figures/`: generated figures used to document the measurements.
- `datasets/`: placeholder structure and metadata for data reacquisition.
- `docs/`: reproducibility notes, methodological notes, and revision logs.
- `check_labels_refs.py`, `pytest.ini`, `.gitignore`, `.gitattributes`.

## Scientific Problem

Many security-relevant systems emit ordered relational traces: edits in a wiki,
learning-platform events, listening histories, location transitions, or network
interaction streams. A sender can use the freedom inside these traces to signal
covertly, but each selected action affects the future context and can expose a
detectable deviation.

The project addresses this question:

> How can a sender encode short messages through valid temporal graph
> continuations while keeping the emitted traversal inside an explicit
> steganalysis budget?

The channel is designed for low-rate signaling such as control commands,
session markers, compact metadata, or short capability tokens. It is not a bulk
transport channel.

## Method Overview

The artifact implements the following components.

### Temporal Traversal Channel

A transmitted object is a sequence of valid interactions
`(a_1, ..., a_L, STOP)`. At each step, the public temporal context defines the
admissible set of next interactions. The encoder maps secret bits to one valid
continuation when the local state is suitable. The receiver uses the same public
history, candidate ordering, model state, key, and finite-precision coding rule
to recover the consumed bit prefix.

### Causal Cover Model

The primary cover model is a temporal source-plus-history backoff model. It is
used because its calibration, candidate probabilities, cold-start behavior, and
state coverage are transparent under causal train/validation/test splits.

The repository also contains neural alternatives:

- GRU sequence cover model.
- Transformer-style sequence cover model.
- Temporal-graph neural cover model.

These models are useful for diagnostics and future capacity improvements. The
reported detector-constrained operating points use the calibrated backoff model.

### Abstention Policy

The controller supports four actions:

- `EMBED`: consume secret bits and emit the encoded valid continuation.
- `COVER`: emit the natural cover action without consuming bits.
- `PAUSE`: maintain synchronization while deferring embedding.
- `STOP`: terminate the traversal.

The policy uses local entropy, calibration uncertainty, rank constraints,
local divergence, payload pressure, dead-end risk, and adversarial budget
checks.

### Coding

Two coding families are implemented:

- Local exact range coding over each action.
- Sequential range-coding prototype over blocks of safe transitions.

The local codec consumes a bit prefix only when the emitted action corresponds
to a singleton integer interval at shared finite precision. This preserves the
Alice-Bob decoding contract and makes payload accounting exact.

### Security Evaluation

The artifact evaluates the channel through:

- Public-feature external steganalysis.
- Independent neural sequence steganalysis.
- Bounded white-box steganalysis over public diagnostics.
- Oracle-leakage audits over internal instrumentation variables.
- Distributional audits using MMD, Wasserstein, and Jensen-Shannon distances.
- Capacity-detectability sweeps across safety thresholds.
- Active-channel reliability under bounded deletion, insertion, reordering,
  and retention.

## Main Reported Results

At the conservative operating point:

- TGBL-Wiki public-feature detector AUC reaches 0.540.
- T-Drive public-feature detector AUC reaches 0.523.
- Bounded white-box AUC reaches 0.530 on TGBL-Wiki and 0.521 on T-Drive.
- Oracle-instrumented AUC reaches 0.537 on TGBL-Wiki and 0.533 on T-Drive.
- The active-channel layer recovers the tested payloads with zero
  post-decoding BER under the configured bounded attacks.

The capacity-detectability sweep exposes higher selectable operating points
under public-feature AUC 0.60:

| Dataset | Operating point | Bits/transition | Max public AUC |
|---|---:|---:|---:|
| TGBL-Wiki | open-rank | 0.1338 | 0.59295 |
| T-Drive cells | balanced | 0.1106 | 0.56561 |
| LastFM | two-bit probe | 0.0536 | 0.52476 |
| MOOC | open-rank | 0.0254 | 0.52380 |
| GeoLife cells | balanced | 0.0007 | 0.50007 |

Short-payload accounting under the same public-feature AUC budget:

| Dataset | 8 bits | 32 bits | 64 bits | 128 bits |
|---|---:|---:|---:|---:|
| TGBL-Wiki | 60 | 240 | 479 | 957 |
| T-Drive cells | 73 | 290 | 579 | 1,158 |
| LastFM | 150 | 598 | 1,195 | 2,389 |
| MOOC | 315 | 1,260 | 2,520 | 5,040 |
| GeoLife cells | 11,429 | 45,715 | 91,429 | 182,858 |

Oracle-leakage audit:

| Dataset | Best bounded AUC | Best oracle AUC | Oracle epsilon |
|---|---:|---:|---:|
| TGBL-Wiki | 0.52952 | 0.53716 | 0.00764 |
| T-Drive cells | 0.52139 | 0.53294 | 0.01155 |
| MOOC | 0.50087 | 0.50087 | 0.00000 |
| LastFM | 0.50000 | 0.50000 | 0.00000 |
| GeoLife cells | 0.50000 | 0.50000 | 0.00000 |

## Repository Layout

```text
.
|-- code/
|   |-- src/
|   |   |-- baselines/       # graph steganography and walk baselines
|   |   |-- controllers/     # abstention controller and baselines
|   |   |-- data/            # dataset adapters and causal splits
|   |   |-- models/          # cover models
|   |   |-- steganalysis/    # detectors and adversarial audits
|   |   `-- stego/           # coding, encryption, active-channel recovery
|   |-- scripts/             # deterministic experiment entry points
|   |-- tests/               # pytest suite
|   |-- requirements.txt
|   `-- pyproject.toml
|-- datasets/
|   |-- raw/                 # user-acquired datasets, ignored by git
|   |-- interim/             # intermediate data, ignored by git
|   `-- processed/           # processed data, ignored by git
|-- experiments/
|   `-- real_world/          # frozen YAML experiment configurations
|-- results/
|   |-- figures/             # derived figures
|   `-- tables/              # derived CSV/JSON tables
|-- docs/
|   `-- roadmap/             # methodology and revision notes
|-- README.md
|-- check_labels_refs.py
`-- pytest.ini
```

## Environment

The artifact was exercised on Windows with Python 3.13. Linux/macOS should work
with equivalent package versions.

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r code\requirements.txt
```

Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r code/requirements.txt
```

Core dependencies include NumPy, pandas, SciPy, scikit-learn, PyArrow,
Matplotlib, PyTorch, PyTorch Geometric, TGB, scikit-fuzzy, reedsolo, pytest,
and Ruff.

## Data

Raw third-party datasets are not redistributed. The repository records the
scripts, metadata, checksums where available, and preprocessing logic needed to
recreate processed event tables after acquiring the raw data from the original
sources.

The reported experiments use:

- TGBL-Wiki interaction stream.
- MOOC interaction stream.
- LastFM interaction stream.
- GeoLife mobility trajectories discretized into cells.
- T-Drive mobility trajectories discretized into cells.

Expected local structure:

```text
datasets/
|-- raw/
|-- interim/
`-- processed/
    |-- tgbl-wiki/
    |-- mooc/
    |-- lastfm/
    |-- geolife_cells/
    `-- t_drive_cells/
```

The processed event files are Parquet tables with at least:

- `source`
- `destination`
- `timestamp`
- `split`

The `split` column is causal and takes values `train`, `validation`, or `test`.

## Quick Validation

From the repository root:

```powershell
pytest code\tests
python code\scripts\validate_project.py
```

Expected validation in the current artifact:

- `44 passed` for the test suite.
- `Project structure is valid` for structure validation.

## Reproducing the Results

The script filenames preserve stable internal identifiers. The scientific role
of each command is described below.

### Dataset Catalog and Causal Splits

```powershell
python code\scripts\run_phase3_data_pipeline.py
python code\scripts\validate_phase3_outputs.py
```

Outputs:

- `results/tables/phase3_dataset_statistics.csv`
- `results/tables/phase3_dataset_statistics.json`
- `results/tables/phase3_validation.json`

### Cover-Model Diagnostics

```powershell
python code\scripts\run_phase4_cover_model.py
python code\scripts\run_phase4_temporal_cover_model.py
python code\scripts\run_phase4_neural_cover_model.py
python code\scripts\run_phase9_temporal_gnn_cover_model.py
```

Outputs include:

- `results/tables/phase4_cover_model_baseline.csv`
- `results/tables/phase4_temporal_cover_model.csv`
- `results/tables/phase4_neural_cover_model.csv`
- `results/tables/phase9_temporal_gnn_cover_model.csv`

### Mobility Discretization Diagnostics

```powershell
python code\scripts\run_phase4_spatial_discretization.py
python code\scripts\run_phase4_spatial_sensitivity.py
```

Outputs include:

- `results/tables/phase4_spatial_discretization.json`
- `results/tables/phase4_spatial_sensitivity.csv`

### Codec and Controller Contracts

```powershell
python code\scripts\run_phase5_codec.py
python code\scripts\run_phase6_fuzzy_controller.py
python code\scripts\optimize_fuzzy_weights.py
```

Outputs include:

- `results/tables/phase5_codec_baseline.json`
- `results/tables/phase6_fuzzy_controller_baseline.csv`
- `results/tables/phase6_fuzzy_controller_real_data.csv`
- `results/tables/phase6_fuzzy_weight_optimization.json`

### External Steganalysis

```powershell
python code\scripts\run_phase7_steganalysis.py
python code\scripts\run_phase7_multiseed.py
```

Outputs include:

- `results/tables/phase7_steganalysis.csv`
- `results/tables/phase7_steganalysis.json`
- `results/tables/phase7_steganalysis_multiseed.csv`

### Robustness and Active Channel

```powershell
python code\scripts\run_phase8_robustness.py
python code\scripts\run_phase9_active_channel_reliability.py
```

Outputs include:

- `results/tables/phase8_robustness.csv`
- `results/tables/phase9_active_channel_reliability.csv`
- `results/figures/active_channel_reliability.tikz`

### Independent Neural and Adaptive Steganalysis

```powershell
python code\scripts\run_phase9_independent_neural_steganalysis.py
python code\scripts\run_phase9_adaptive_steganalysis.py
python code\scripts\run_phase11_adversarial_audit.py
```

Outputs include:

- `results/tables/phase9_independent_neural_steganalysis.csv`
- `results/tables/phase9_adaptive_steganalysis.csv`
- `results/tables/phase11_adversarial_audit.csv`

### Capacity and Distributional Audits

```powershell
python code\scripts\run_phase11_capacity_audit.py
python code\scripts\run_phase11_distribution_audit.py
python code\scripts\run_phase11_range_codec.py
```

Outputs include:

- `results/tables/phase11_capacity_audit.csv`
- `results/tables/phase11_distribution_audit.csv`
- `results/tables/phase11_range_codec.csv`

### Capacity-Detectability Frontier and Practical Payloads

```powershell
python code\scripts\run_phase12_capacity_detectability_sweep.py
python code\scripts\run_phase12_oracle_leakage_audit.py
python code\scripts\run_phase12_cover_sensitivity.py
python code\scripts\run_phase12_practical_payloads.py
```

Outputs include:

- `results/tables/phase12_capacity_detectability_summary.csv`
- `results/tables/phase12_capacity_detectability_sweep.csv`
- `results/figures/phase12_capacity_detectability_sweep.pdf`
- `results/tables/phase12_oracle_leakage_audit.csv`
- `results/tables/phase12_oracle_leakage_correlations.csv`
- `results/tables/phase12_cover_sensitivity.csv`
- `results/tables/phase12_practical_payloads.csv`

## Reproducing the Main Tables Used in the Paper

The paper-level tables can be checked against:

- Capacity audit:
  `results/tables/phase11_capacity_audit.csv`
- Conservative external steganalysis:
  `results/tables/phase7_steganalysis.csv`
- Distributional audit:
  `results/tables/phase11_distribution_audit.csv`
- Layered adversarial audit:
  `results/tables/phase11_adversarial_audit.csv`
- Capacity-detectability frontier:
  `results/tables/phase12_capacity_detectability_summary.csv`
- Oracle leakage:
  `results/tables/phase12_oracle_leakage_audit.csv`
- Practical payloads:
  `results/tables/phase12_practical_payloads.csv`

## GitHub Export

For public release, prepare an export that excludes manuscript working
directories and private local material:

```powershell
robocopy . github_publish\Detection-Constrained-Steganography-on-Temporal-Graph-Traversals /E /XD .git article IEEE_TSP DCS_Projects github_publish datasets\raw datasets\interim datasets\processed .pytest_cache .ruff_cache .benchmarks .miktex /XF *.aux *.bbl *.blg *.fdb_latexmk *.fls *.log *.out *.synctex.gz
```

Then initialize and push from the export directory:

```powershell
cd github_publish\Detection-Constrained-Steganography-on-Temporal-Graph-Traversals
git init
git remote add origin https://github.com/EkodeckStephane/Detection-Constrained-Steganography-on-Temporal-Graph-Traversals
git add .
git commit -m "Initial public reproducibility artifact"
git push -u origin master
```

Review `git status --short` before pushing. The export must not contain
`article/`, `IEEE_TSP/`, or `DCS_Projects/`.

## Citation

If you use this artifact, cite the associated manuscript:

```text
S. G. R. Ekodeck et al.,
"Detection-Constrained Steganography over Temporal Graph Traversals,"
manuscript in preparation, 2026.
```

## License and Data Terms

The source code is intended for research use under the project license. Raw
third-party datasets remain governed by their original licenses and are not
redistributed in this repository.

## Contact

Corresponding author:

```text
Stephane Gael R. Ekodeck
stephane-gael.ekodeck@facsciences-uy1.cm
```
