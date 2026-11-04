# Learning transferable spatial interaction representations for street-level mobility forecasting across cities

Code, trained model checkpoints, and a processed evaluation dataset accompanying the manuscript:

**Learning transferable spatial interaction representations for street-level mobility forecasting across cities**  
Hongrong Yang and Markus Schläpfer  
Columbia University

## Repository contents

```text
street-mobility-transfer/
├── model/
│   ├── pre_la.pth
│   ├── cold_start_sf_9d.pth
│   ├── rl_sf_9d.pth
│   └── sl_sf_full.pth
│
├── POI_data/
│   └── poi_type_mapping_la_to_sf.pkl  # Los Angeles-to-San Francisco POI-type mapping
│
├── model.py
├── model_performance_test.py
├── evaluate_released_models.py
│
├── pre_training_ztp.py
├── cold_start_ztp.py
├── fine_tuning_PPO.py
├── PCA_RL.py
│
├── graph_data_loader_slide_LA.py
├── graph_data_loader_slide_SF.py
├── graph_data_loader_slide_SF_RLFT.py
│
├── DCRNN.py
├── DCRNN_test.py
├── STGCN.py
├── STGCN_test.py
├── Graphwave.py
├── Graphwave_test.py
├── mcgcn.py
├── mcgcn_test.py
├── edge_mlp.py
├── edge_mlp_test.py
│
├── README.md
└── .gitignore
```

The repository provides Python source code rather than a compiled standalone executable. The San Francisco evaluation graphs are distributed separately as `SF_test.zip`, attached to [GitHub Release data-v1.0](https://github.com/hongrongyang/street-mobility-transfer/releases/tag/data-v1.0); they are not stored in the main repository tree.

## System requirements

### Operating systems

The code has been tested with Python 3.12.6 on Windows 11, version 25H2.

Other operating systems may also be compatible but have not been formally tested.

### Software dependencies

The tested environment used:

```text
Python                 3.12.6
NumPy                  1.26.4
pandas                 2.2.3
SciPy                  1.13.1
scikit-learn           1.8.0
PyTorch                2.5.1
PyTorch Geometric      2.6.1
Stable-Baselines3      2.5.0
GeoPandas              1.0.1
OSMnx                  2.0.0
NetworkX               2.8.8
pytz                   2024.2
```

### Hardware

A CUDA-capable GPU is recommended for full training and reinforcement-learning refinement. The released model checkpoints can be evaluated on a CPU, although evaluation is faster on a GPU. No other non-standard hardware is required.

## Installation

Clone the repository and create a clean Python environment:

```bash
git clone https://github.com/hongrongyang/street-mobility-transfer.git
cd street-mobility-transfer

python -m venv .venv
```

Activate the environment on macOS or Linux:

```bash
source .venv/bin/activate
```

Activate it on Windows:

```text
.venv\Scripts\activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install numpy==1.26.4 pandas==2.2.3 scipy==1.13.1 scikit-learn==1.8.0
python -m pip install torch==2.5.1 torch-geometric==2.6.1 stable-baselines3==2.5.0 networkx==2.8.8 pytz==2024.2
python -m pip install geopandas==1.0.1 osmnx==2.0.0
```

For a CUDA-enabled installation of PyTorch, use the command appropriate for the local CUDA version from the official PyTorch installation guide.

Typical installation time on a normal desktop computer with an existing Python installation and a standard internet connection is approximately 10–20 minutes, excluding download of the evaluation data.

## Data availability

The raw anonymized mobility data used in this study were provided by CITYDATA.ai and are subject to privacy and licensing restrictions. The authors cannot redistribute the raw mobility records. Access to the full dataset may be requested directly from CITYDATA.ai and is subject to its licensing terms.

A processed San Francisco evaluation subset sufficient to run the code demonstration and evaluate the released San Francisco models is provided as [`SF_test.zip`](https://github.com/hongrongyang/street-mobility-transfer/releases/download/data-v1.0/SF_test.zip), attached to [GitHub Release data-v1.0](https://github.com/hongrongyang/street-mobility-transfer/releases/tag/data-v1.0). The archive is not tracked in the main branch of the repository.

**Download [`SF_test.zip`](https://github.com/hongrongyang/street-mobility-transfer/releases/download/data-v1.0/SF_test.zip), not the automatically generated “Source code (zip)” or “Source code (tar.gz)” archives.** Those archives contain the repository source code, not the evaluation graphs.

Extract the dataset locally so that the project directory has the following structure:

```text
street-mobility-transfer/
└── graph_data/
    └── SF_test/
        ├── graph_YYYYMMDD_HHMMSS.gpickle
        └── ...
```

The released graph files contain aggregated, time-indexed POI-level graph data. They do not contain raw GPS records or device-level trajectories.

The repository also contains [`POI_data/poi_type_mapping_la_to_sf.pkl`](POI_data/poi_type_mapping_la_to_sf.pkl), which maps Los Angeles POI categories to the San Francisco POI-type representation used by the transfer and evaluation code.

## Model checkpoints

Four trained model checkpoints are provided:

```text
model/pre_la.pth             Los Angeles pretrained model
model/cold_start_sf_9d.pth   San Francisco 9-day cold-start adapted model
model/rl_sf_9d.pth           San Francisco 9-day reinforcement-learning-refined model
model/sl_sf_full.pth         San Francisco fully supervised model
```

The checkpoint files contain model state dictionaries only. They do not include raw mobility data, optimizer states, device-level trajectories, or training logs.

## Demo: evaluating the released models

`evaluate_released_models.py` evaluates the four released checkpoints on the processed San Francisco test graphs and prints each metric next to the value reported in the manuscript (Supplementary Table 3).

1. Download [`SF_test.zip`](https://github.com/hongrongyang/street-mobility-transfer/releases/download/data-v1.0/SF_test.zip) and extract it to `graph_data/SF_test/` as described above.
2. Run from the repository root:

```bash
python evaluate_released_models.py --test_dir ./graph_data/SF_test --model_dir ./model
```

The script loads `pre_la.pth` (Los Angeles pretrained model, zero-shot), `cold_start_sf_9d.pth` (9-day cold-start adaptation), `rl_sf_9d.pth` (9-day cold start with reinforcement-learning refinement) and `sl_sf_full.pth` (supervised training on the full San Francisco month) and evaluates each on the 108 hourly test graphs (27–31 March 2023).

### Expected output

For each checkpoint, the script prints the overall and Top-1% ZTP NLL, MSE and MAE, the corresponding values from the manuscript, and the relative difference. The reported values are:

| Model | Top-1% ZTP NLL | Top-1% MSE | Top-1% MAE | Overall ZTP NLL | Overall MSE | Overall MAE |
| --- | --- | --- | --- | --- | --- | --- |
| `pre_la.pth` (zero-shot) | 3.849 | 12.696 | 2.008 | 0.248 | 0.274 | 0.227 |
| `cold_start_sf_9d.pth` | 2.388 | 6.893 | 1.450 | 0.225 | 0.339 | 0.218 |
| `rl_sf_9d.pth` | 2.336 | 6.772 | 1.441 | 0.240 | 0.375 | 0.237 |
| `sl_sf_full.pth` | 2.349 | 7.141 | 1.415 | 0.290 | 0.385 | 0.291 |

ZTP NLL is the unweighted zero-truncated Poisson negative log-likelihood; MSE and MAE are computed between the observed count and the ZTP mean. The Top-1% subset consists of the ⌈0.01 M⌉ test edge–hour observations with the largest observed counts among the M test observations, with ties at the threshold count resolved in a fixed order, as in the manuscript. Small numerical differences may occur across hardware and software environments.

**Typical runtime:** approximately 3–5 minutes per checkpoint on a normal desktop computer, less on a GPU.

`model_performance_test.py` remains available for evaluating a single checkpoint with the data and model paths set inside the script.

## Optional training demo

The released San Francisco graphs may also be divided chronologically into small training, validation, and test subsets to demonstrate that the training code executes. Temporal order should be preserved when creating these subsets to prevent future observations from leaking into earlier splits.

Update the relevant data paths in the San Francisco data loader and training script, then run the selected training stage. For example, the cold-start adaptation code can be run with:

```bash
python cold_start_ztp.py
```

This lightweight split is intended only as a software demonstration. It is not the experimental split used for the manuscript and will not reproduce the reported numerical results.

The expected output is epoch-by-epoch training and validation loss, followed by a saved model checkpoint according to the output path configured in the script. Runtime depends on the chosen subset size, number of epochs, and hardware.

## Instructions for use with other data

To use the software with another city or dataset:

1. Convert the data into time-ordered NetworkX graph snapshots saved as `.gpickle` files.
2. Use consistent node identifiers and node ordering across snapshots.
3. Provide the node attributes required by the model, including POI type, coordinates, dynamic population, temperature, precipitation, and wind speed.
4. Store directed mobility counts in the edge attribute `population_flow`.
5. Adapt the Los Angeles or San Francisco data loader to the new city. Set its data paths, coordinate bounds, feature normalization ranges, and POI-type mapping.
6. Preserve chronological order when constructing training, validation, and test splits.
7. Run the desired training, adaptation, or evaluation script.

The principal workflows are:

```bash
python pre_training_ztp.py        # Source-city pretraining
python cold_start_ztp.py          # Target-city cold-start adaptation
python fine_tuning_PPO.py         # Reinforcement-learning refinement
python model_performance_test.py  # Model evaluation
```

## Reproducing manuscript results

The released `SF_test` subset and model checkpoints reproduce the San Francisco entries of Supplementary Table 3 for zero-shot transfer, 9-day cold-start adaptation, 9-day cold-start adaptation with RL refinement and full supervision; run `evaluate_released_models.py` as described above.

Reproducing the training runs requires access to the licensed CITYDATA.ai records. The public repository provides graph data loaders for Los Angeles and San Francisco.

## Baseline models

| Baseline | Implementation | Evaluation script |
| --- | --- | --- |
| Extended DCRNN | `DCRNN.py` | `DCRNN_test.py` |
| Extended STGCN | `STGCN.py` | `STGCN_test.py` |
| Extended Graph WaveNet | `Graphwave.py` | `Graphwave_test.py` |
| Extended MCGCN | `mcgcn.py` | `mcgcn_test.py` |
| Shared edge MLP | `edge_mlp.py` | `edge_mlp_test.py` |

The manuscript evaluates MCGCN in Los Angeles because its learned graph depends on the identities of the source-city POIs. The cross-city comparisons include the extended DCRNN, STGCN, Graph WaveNet, and shared edge MLP (Supplementary Note 10). The four released checkpoints listed above belong to the proposed model.

Both require the Los Angeles graphs, which are not part of the public release.

## License

No explicit software license has been assigned at this stage. The CITYDATA.ai-derived data and trained model checkpoints remain subject to the applicable data-use and licensing restrictions.

## Citation

This repository accompanies the manuscript:

**Learning transferable spatial interaction representations for street-level mobility forecasting across cities**  
Hongrong Yang and Markus Schläpfer

A formal citation will be added after publication or after a preprint becomes available.

## Contact

For questions about the code, please contact:

Hongrong Yang  
Columbia University
