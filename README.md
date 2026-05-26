# Street-level mobility interactions transfer across cities with minimal local data

Code, processed graph data for the San Francisco test dataset, and trained model checkpoints for the paper:

**Street-level mobility interactions transfer across cities with minimal local data**  
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
├── model.py
├── model_performance_test.py
├── pair_save.py
├── figure_2.py
│
├── pre_training_ztp.py
├── cold_start_ztp.py
├── fine_tuning_PPO.py
├── PCA_RL.py
│
├── graph_data_loader_slide_LA.py
├── graph_data_loader_slide_SF.py
├── graph_data_loader_slide_SF_RLFT.py
├── graph_data_loader_slide_FRE.py
├── graph_data_loader_slide_FRE_RLFT.py
│
├── DCRNN.py
├── DCRNN_test.py
├── STGCN.py
├── STGCN_test.py
├── Graphwave.py
├── Graphwave_test.py
│
├── SF.png
├── SF.pdf
├── README.md
└── .gitignore
```

## Data availability

Due to licensing restrictions from CITYDATA.ai, the full training data used in this study are not publicly available.

The processed San Francisco test graph data are provided in the GitHub Release `v1.0.0` for reproducing the reported metrics and case-study figures. The released data do not include raw mobility trajectories, device-level records, or personally identifiable information.

For access to the full mobility dataset, please contact CITYDATA.ai.

## Released graph data

The processed graph data are provided as a compressed archive in the GitHub Release `v1.0.0`.

After downloading and extracting the archive, place the extracted graph data folder in the project directory expected by the data loader scripts.

The released San Francisco graph data are used by:

```text
model_performance_test.py
pair_save.py
figure_2.py
graph_data_loader_slide_SF.py
graph_data_loader_slide_SF_RLFT.py
```

## Model checkpoints

The repository includes four trained model checkpoints:

```text
model/pre_la.pth              Los Angeles pretrained model
model/cold_start_sf_9d.pth    San Francisco 9-day cold-start adapted model
model/rl_sf_9d.pth            San Francisco 9-day reinforcement-learning refined model
model/sl_sf_full.pth          San Francisco fully supervised model
```

The checkpoint files contain model state dictionaries only. They do not include raw mobility data, training data, optimizer states, device-level trajectories, or training logs.

## Reproducing model performance

To evaluate the released models on the San Francisco test graph data, run:

```bash
python model_performance_test.py
```

This script computes the main prediction metrics used in the paper, including:

```text
Overall performance
Top-1% high-intensity-flow performance
Top-0.1% high-intensity-flow performance
```

The output can be used to verify the reported model performance.

## Reproducing the Figure 2 case study

Figure 2 evaluates whether 9-day cross-city adaptation from Los Angeles can reproduce directed POI-to-POI mobility dynamics in San Francisco.

To generate the one-day bidirectional flow predictions for the three representative San Francisco POI pairs shown in Fig. 2b, run:

```bash
python pair_save.py
```

This script saves the predicted and observed hourly flows for the selected POI pairs, including results from the 9-day cross-city adapted model and the fully supervised San Francisco model.

After running `pair_save.py`, generate the Figure 2 visualization with:

```bash
python figure_2.py
```

This script generates the Figure 2 image and associated result files, including the POI-to-POI flow comparison and the three-pair bidirectional hourly flow panels with ZTP probability heatmaps.

The reproduction workflow is:

```bash
python model_performance_test.py
python pair_save.py
python figure_2.py
```

The first script verifies model metrics, including overall, Top-1%, and Top-0.1% performance. The second script generates the San Francisco three-pair one-day prediction results. The third script generates the Figure 2 image and corresponding outputs.

## Training and adaptation scripts

The repository also includes scripts for the main training and adaptation stages:

```text
pre_training_ztp.py      Source-city pretraining with the ZTP objective
cold_start_ztp.py        Cold-start target-city adaptation
fine_tuning_PPO.py       Reinforcement-learning refinement
```

Additional analysis and data-loading scripts:

```text
PCA_RL.py
graph_data_loader_slide_LA.py
graph_data_loader_slide_SF.py
graph_data_loader_slide_SF_RLFT.py
graph_data_loader_slide_FRE.py
graph_data_loader_slide_FRE_RLFT.py
```

Baseline model implementations and testing scripts are included for comparison:

```text
DCRNN.py
DCRNN_test.py
STGCN.py
STGCN_test.py
Graphwave.py
Graphwave_test.py
```

## Notes on reproducibility

The released San Francisco test graph data and trained checkpoints are intended to reproduce the reported San Francisco metrics and the Figure 2 case-study results.

Full end-to-end retraining from raw mobility trajectories requires access to the restricted CITYDATA.ai dataset.

## Citation

This repository accompanies the manuscript:

**Street-level mobility interactions transfer across cities with minimal local data**  
Hongrong Yang and Markus Schläpfer

A formal citation will be added after publication or after a preprint becomes available.

## Contact

For questions about the code, please contact:

Hongrong Yang  
Columbia University
