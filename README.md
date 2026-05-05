# CF Distance Attack — Execution Guide

This README is a GitHub-renderable companion to the notebook "CF_distance attack execution guide.ipynb" and summarizes the steps, commands, and requirements to reproduce the counterfactual-distance attack pipeline described in that notebook.

## Overview

This guide covers the workflow used to:
- Train models for each dataset and counterfactual (CF) method.
- Generate counterfactual explanations (CFs) using multiple CF methods (e.g., `dice_gradient`, `scfe`, etc.).
- Convert saved CFs into attack datasets for the membership inference / distance-based attack (`cf_distance_attack.py`).
- Run the CF distance attack and aggregate/visualize results.

Notes from the notebook:
- Two different Python environments are required because `dice_gradient` uses TensorFlow while `scfe` uses PyTorch.
  - Use `tfdice_requirements.txt` for TensorFlow-related CF generation (`dice_gradient`).
  - Use `cfdist_requirements.txt` for other CF methods.

## Requirements

- Python 3.8+ (or environment matching the project requirements).
- Two requirement files are referenced in the notebook (install in separate venvs as needed):
  - `tfdice_requirements.txt` (for TensorFlow / `dice_gradient` workflows)
  - `cfdist_requirements.txt` (for other CF methods and CF-distance attack tools)

Create a virtual environment and install requirements (example):

```bash
python -m venv .venv-cfdist
source .venv-cfdist/bin/activate
pip install -r cfdist_requirements.txt
```

And for TensorFlow-based CFs:

```bash
python -m venv .venv-tfdice
source .venv-tfdice/bin/activate
pip install -r tfdice_requirements.txt
```

Adjust Python interpreter and package versions to match your platform and GPU availability.

## Execution Steps (high level)

1) Train models and save explainers for each dataset and CF method

- For each dataset (e.g., `adult`, `acs_income`, ...), run the TensorFlow trainer `MIA_model_trainer_TF.py` with the target `--cf_method` to create and save explainers. Example (from notebook):

```bash
python MIA_model_trainer_TF.py --dataset adult --rseed 2 --model RF --cf_method dice_gradient
```

- Exception: for `scfe` the model training and CF generation are combined in `train_model_scfe_parallel.py`.

2) Convert generated CF files into attack-ready datasets

- After CFs are generated and saved, convert them to synthetic data / attack datasets using `new_MIA_dataloader.py` for all seeds, datasets and CF methods:

```bash
python new_MIA_dataloader.py --dataset adult --rseed 2 --cf_method dice_gradient
```

3) Analyze CF statistics (optional but recommended)

- Run the following scripts (for all datasets & methods) to compute aggregated CF statistics and visualizations:
  - `CF_resutl_analysis_updateformia.py`
  - `cf_visuasualize_results_updateformia.py`
  - `aggregation_updateformia.py`

4) Run the CF-distance attack

- Execute `cf_distance_attack.py` across datasets, CF methods, and random seeds. Example:

```bash
python cf_distance_attack.py --dataset adult --rseed 2 --cf_method dice_gradient
```

5) Analyze and visualize attack results

- To analyze aggregated attack outputs run `attack_results_analysis.py` for each dataset.
- To draw final unified figures and evaluation tables run:

```bash
python draw_plots_cf_distance_attack.py
```

6) (Optional) Ensemble / further attack analysis

- The notebook notes moving generated attack datasets and synthetic data to the `ensemble-mia` folder and following the ensemble attack execution guide there.

## Example Commands 

- Train (example):

```bash
python MIA_model_trainer_TF.py --dataset adult --rseed 2 --model RF --cf_method dice_gradient
```

- Convert CF to attack dataset (example):

```bash
python new_MIA_dataloader.py --dataset adult --rseed 2 --cf_method dice_gradient
```

- Run CF distance attack (example):

```bash
python cf_distance_attack.py --dataset adult --rseed 2 --cf_method dice_gradient
```

- Draw plots / aggregate results:

```bash
python attack_results_analysis.py
python draw_plots_cf_distance_attack.py
```

## Paper citation

This citation will be updated upon official publication.

```bibtex
@article{---,
	title={Quantifying the Privacy of Counterfactuals by Leveraging Membership Inference Attacks against Synthetic Data},
	author={Babaei M., Wang Y., Lautraite H., Arcolezi H.H., Aivodji U., Gambs S.},
	journal={ACM Conference on Fairness, Accountability, and Transparency (FAccT)},
	year={2026}
}
```

## Resource citation
This work builds upon and extends the following repository:

```bibtex
@inproceedings{pawelczyk2023privacy,
  title={On the privacy risks of algorithmic recourse},
  author={Pawelczyk, Martin and Lakkaraju, Himabindu and Neel, Seth},
  booktitle={International Conference on Artificial Intelligence and Statistics},
  pages={9680--9696},
  year={2023},
  organization={PMLR}
}
```
GitHub repository: [https://github.com/safr-ai-lab/CounterfactualDistanceAttack](https://github.com/safr-ai-lab/CounterfactualDistanceAttack)

*This README was generated automatically from the notebook "CF_distance attack execution guide.ipynb".*
