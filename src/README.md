# src/ README

This folder contains the core preprocessing, dataset, and model training scripts for the Mental Manipulation NLP project.

## Files

- `preprocess.py`
  - Loads raw CSV files from `data/raw/`.
  - Cleans the data and standardizes column names to `text`, `label`, and `technique`.
  - Creates three experiment datasets:
    - `data/exp1_maj/` — majority-only split
    - `data/exp2_con/` — control-only split
    - `data/exp3_clean/` — clean combined split
  - Run with:
    ```powershell
    python src/preprocess.py
    ```

- `baseline.py`
  - Trains a classical TF-IDF + Logistic Regression classifier.
  - Loads data from `data/processed/train.csv`, `data/processed/val.csv`, and `data/processed/processed_con.csv`.
  - Evaluates on train, validation, and test sets.
  - Writes outputs to `results/` including metrics, confusion matrices, and reports.
  - Run with:
    ```powershell
    python src/baseline.py
    ```

- `train.py`
  - Trains a DistilBERT sequence classification model.
  - Uses `data/processed/*` as input.
  - Applies class weighting and saves the best model to `model/`.
  - Logs metrics and evaluation outputs to `results/`.
  - Run with:
    ```powershell
    python src/train.py
    ```

- `train_lora.py`
  - Trains a LoRA-adapted DistilBERT model using PEFT.
  - Loads the same processed data as `train.py`.
  - Saves the best LoRA model to `model_lora/`.
  - Writes logs to `results/`.
  - Run with:
    ```powershell
    python src/train_lora.py
    ```

- `train-v2.py`
  - Improved DistilBERT training script with dynamic dataset path support.
  - Accepts experiment dataset folders and writes results to `results-v2/`.
  - Example:
    ```powershell
    python src/train-v2.py --data_dir data/exp1_maj --exp_name maj
    python src/train-v2.py --data_dir data/exp2_con --exp_name con
    python src/train-v2.py --data_dir data/exp3_clean --exp_name clean
    ```

- `train-v3-finetune.py`
  - Fine-tunes DistilBERT with custom class weights and threshold tuning.
  - Saves best models under `models/clean-finetune/<exp_name>`.
  - Outputs results to `results-v3/`.
  - Example:
    ```powershell
    python src/train-v3-finetune.py --data_dir data/exp3_clean --exp_name clean_finetune
    ```

- `train-v4-lora-finetune.py`
  - Fine-tunes a LoRA-enabled DistilBERT model starting from a pre-trained `models/clean-finetune/clean-finetune` checkpoint.
  - Uses an experiment dataset folder and saves results to `results-v4/`.
  - Example:
    ```powershell
    python src/train-v4-lora-finetune.py --data_dir data/exp3_clean --exp_name clean_finetune_lora
    ```

- `dataset.py`
  - Defines the `ManipulationDataset` PyTorch dataset class.
  - Tokenizes input text with the Hugging Face tokenizer.
  - Returns `input_ids`, `attention_mask`, and `labels` for model training.

- `utils/logger.py`
  - Provides helper functions for result logging:
    - `init_results_dir()`
    - `log_to_file(message, filename)`
    - `save_metrics(result, filename)`
    - `save_confusion_matrix(cm, filename)`
    - `save_report(report, filename)`

## How to run

1. Install dependencies from the project root:

   ```powershell
   pip install -r requirements.txt
   ```

2. Preprocess raw files:

   ```powershell
   python src/preprocess.py
   ```

3. Run a baseline model:

   ```powershell
   python src/baseline.py
   ```

4. Train DistilBERT:

   ```powershell
   python src/train.py
   ```

5. Train LoRA DistilBERT:

   ```powershell
   python src/train_lora.py
   ```

6. Train versioned experiments:

   ```powershell
   python src/train-v2.py --data_dir data/exp3_clean --exp_name exp3_clean
   python src/train-v3-finetune.py --data_dir data/exp3_clean --exp_name clean_finetune
   python src/train-v4-lora-finetune.py --data_dir data/exp3_clean --exp_name clean_finetune_lora
   ```

## Notes

- Most training scripts assume `data/processed/` or experiment directories with `train.csv`, `val.csv`, and `test.csv`.
- The scripts use `results/`, `results-v2/`, `results-v3/`, and `results-v4/` for output storage.
- `train-v3-finetune.py` and `train-v4-lora-finetune.py` tune evaluation thresholds before final test evaluation.
