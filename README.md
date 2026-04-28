# Mental Manipulation Detection

A reproducible NLP project for detecting manipulative language in dialogue using both discriminative and generative model workflows.

## Project Overview

This repository supports dataset preparation, model training, evaluation, and prediction for the MentalManip dataset. The project includes:

- raw and processed data splits for noisy and consensus annotations
- DistilBERT-based training and fine-tuning pipelines
- LoRA adapter training for efficient parameter updates
- a Qwen-based generative model training workflow
- a Gradio prediction UI for side-by-side model comparison

## Repository Structure

- `data/` — raw files and preprocessed experiment datasets
- `src/` — preprocessing, dataset, training, and prediction scripts
- `models/` — saved model checkpoints, adapters, and Qwen assets
- `results/` — evaluation reports, confusion matrices, and logs
- `requirements.txt` — Python dependencies

## File Structure

```text
mental-manipulation/
├── README.md
├── requirements.txt
├── RESULTS-README.md
├── data/
│   ├── README.md
│   ├── BalancedDataset/
│   ├── exp1_maj/
│   ├── exp2_con/
│   ├── exp3_clean/
│   ├── processed/
│   └── raw/
├── model/
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer_config.json
│   └── tokenizer.json
├── model_lora/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── README.md
├── models/
│   ├── bert/
│   ├── clean-finetune/
│   ├── clean-finetune-lora/
│   ├── offload/
│   ├── Qwen_1.5_BaseModel/
│   └── qwen-manipulation-detector-model/
├── results/
├── results-qwen/
├── src/
│   ├── README.md
│   ├── dataset.py
│   ├── predict-ui.py
│   ├── preprocess.py
│   ├── Qwen_Train_Eval.py
│   ├── train-v2.py
│   ├── train-v3-finetune.py
│   ├── train-v4-lora-finetune.py
│   └── utils/
└── testing/
```

## Key Steps

### 1. Data Preparation

The dataset workflow is described in `data/README.md`.

This step produces the following experimental splits:

- `data/exp1_maj/` — majority-only (noisy labels)
- `data/exp2_con/` — consensus-only (clean labels)
- `data/exp3_clean/` — clean evaluation split with no dialogue leakage
- `data/BalancedDataset/` — balanced training split derived from the clean protocol

The preprocessing step standardizes each CSV to `text`, `label`, and `technique`, cleans missing values, and creates train/validation/test partitions. For full details, see `data/README.md`.

### 2. Training and Evaluation

The core training scripts are documented in `src/README.md`.

The main flows are:

- `src/train-v2.py` — baseline DistilBERT training on dataset splits
- `src/train-v3-finetune.py` — fine-tuning with custom weighting and threshold tuning
- `src/train-v4-lora-finetune.py` — LoRA adapter training on an existing fine-tuned model
- `src/Qwen_Train_Eval.py` — Qwen-based model training with a balanced dataset and adapter tuning

These scripts save trained checkpoints under `models/` and evaluation artifacts under `results-v2/`, `results-v3/`, `results-v4/`, and `results-qwen`.

### 3. Prediction Interface

Once models are trained, `src/predict-ui.py` launches a Gradio web interface for real-time prediction.

It compares:

- the DistilBERT model and its LoRA adapter
- the Qwen model adapter

The UI is designed to show predictions in parallel and requires the relevant saved model directories to be present.

## UI Model Download (Optional)

If you want to run `src/predict-ui.py` directly without retraining, download the two model packages from:

- `<DOWNLOAD_URL_FOR_MODELS>`

The packages should include the DistilBERT and Qwen model assets. Place the extracted model directories exactly as follows:

- `models/clean-finetune/clean-finetune`
- `models/clean-finetune-lora/clean-finetune-lora`
- `models/Qwen_1.5_BaseModel`
- `models/qwen-manipulation-detector-model`

Also ensure the following folder exists before launching the UI:

- `models/offload/`

This ensures `src/predict-ui.py` can load both the discriminative and generative prediction pipelines.

For required model paths and more details, see `src/README.md`.

## Quick Start

1. Install dependencies from the repository root:

```powershell
pip install -r requirements.txt
```

2. Run preprocessing:

```powershell
python src/preprocess.py
```

3. Train baseline models:

```powershell
python src/train-v2.py --data_dir data/exp1_maj --exp_name maj
python src/train-v2.py --data_dir data/exp2_con --exp_name con
python src/train-v2.py --data_dir data/exp3_clean --exp_name clean
```

4. Fine-tune on the clean dataset:

```powershell
python src/train-v3-finetune.py --data_dir data/exp3_clean --exp_name clean_finetune
```

5. Train the LoRA-enabled model:

```powershell
python src/train-v4-lora-finetune.py --data_dir data/exp3_clean --exp_name clean_finetune_lora
```

6. Train the Qwen model adapter:

```powershell
python src/Qwen_Train_Eval.py
```

7. Launch the prediction interface:

```powershell
python src/predict-ui.py
```

## Notes on Experiments

The repository supports multiple evaluation strategies:

- `exp1_maj` measures performance on noisy majority labels as a baseline.
- `exp2_con` measures performance on high-quality consensus labels.
- `exp3_clean` uses a clean protocol with no dialogue overlap between train and test.
- `BalancedDataset` ensures class parity for training while preserving a clean held-out test set.

For the exact split logic and dataset generation details, consult `data/README.md`.

## Results and Artifacts

Generated outputs include:

- model checkpoints in `models/`
- evaluation logs and reports in `results-v2/`, `results-v3/`, `results-v4/` and `results-qwen`
- confusion matrices and classification reports for all experiments

### Results Comparison

| Metric                     | Original CLEAN | Fine-tuned CLEAN | CLEAN + LoRA | QWEN Model |
|----------------------------|----------------|------------------|--------------|------------|
| **Accuracy**               | 0.72           | 0.67             | 0.71         | **0.76**   |
| **Macro F1**               | 0.64           | 0.65             | 0.66         | **0.71**   |
| **Control Precision**      | 0.58           | 0.48             | 0.53         | **0.62**   |
| **Control Recall**         | 0.38           | **0.66**         | 0.52         | 0.56       |
| **Control F1**             | 0.46           | 0.55             | 0.53         | **0.59**   |
| **Manipulation Precision** | 0.76           | 0.74             | 0.79         | **0.81**   |
| **Manipulation Recall**    | **0.88**       | 0.68             | 0.79         | 0.85       |
| **Manipulation F1**        | 0.81           | 0.74             | 0.79         | **0.83**   |

For a full comparison of the training pipeline and metrics, see `RESULTS-README.md`.

## Additional Documentation

- `src/README.md` — script-level usage, model workflow, and training examples
- `data/README.md` — dataset variants, experiment definitions, and preprocessing details
- `RESULTS-README.md` — evaluation summary, performance comparison, and final results

## Recommended Workflow

1. preprocess data
2. train baseline models
3. fine-tune and train LoRA adapters
4. train Qwen adapter
5. evaluate outputs and use the Gradio prediction UI

This README serves as the top-level guide. For implementation details, follow the linked internal READMEs in `data/` and `src/`.