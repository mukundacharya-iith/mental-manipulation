# src/ README

This folder contains the core preprocessing, dataset, and model training scripts for the Mental Manipulation NLP project.

## Files

- `preprocess.py`
  - Loads raw CSV files from `data/raw/`.
  - Cleans the data and standardizes column names to `text`, `label`, and `technique`.
  - Creates four experiment datasets:
    - `data/exp1_maj/` — majority-only split
    - `data/exp2_con/` — control-only split
    - `data/exp3_clean/` — clean combined split
    - `data/BalancedDataset/` — clean combined split + balances the class imbalance present
  - Run with:
    ```powershell
    python src/preprocess.py
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

- `Qwen_Train_Eval.py`
  - Loads the `models/Qwen_1.5_BaseModel` in 4-bit precision to save memory.
  - Uses the balanced split present in `data/BalancedDataset`
  - Initializes a LoRA adapter targeting specific attention modules (`q_proj`, `k_proj`, `v_proj`, `o_proj`).
  - Runs 3 epochs of Supervised Fine-Tuning (SFT).
  - Automatically saves logs and the final model to `models/qwen-manipulation-detector-model`.
  - **Prerequisite**
      - Download the base Qwen model (`Qwen/Qwen2.5-1.5B-Instruct`) locally into the folder named `models/Qwen_1.5_BaseModel`,
  - Example:
    ```powershell
  python src/Qwen_Train_Eval.py
    ```

- `predict-ui.py`
  - Launches a Gradio web interface for real-time comparison between the Discriminative (DistilBERT) and Generative (Qwen) models.
  - Loads the DistilBERT model and its LoRA adapter from `models/clean-finetune` and Qwen from the `models/qwen-manipulation-detector-model directory`.
  - Results are yielded in parallel, meaning whichever model finishes first populates its box while the other continues processing.
  - **Prerequisites** 
      - A `models/offload` folder must be pre-created in the project root to handle model weights.
      - Ensure all model paths are in place:
        - `models/clean-finetune/clean-finetune` (BERT Base)
        - `models/clean-finetune-lora/clean-finetune-lora` (BERT Adapter)
        - `models/Qwen_1.5_BaseModel` (Qwen Base)
        - `models/qwen-manipulation-detector-model` (Qwen Adapter)
  - Example:
    ```powershell
  python src/predict-ui.py
    ```

## How to run

1. Install dependencies from the project root:

   ```powershell
   pip install -r requirements.txt
   ```

2. Preprocess raw files:

   ```powershell
   python src/preprocess.py
   ```

3. Train the base DitilBERT on all the 3 dataset splits:

   ```powershell
    python train-v2.py --data_dir data/exp1_maj --exp_name maj
    python train-v2.py --data_dir data/exp2_con --exp_name con
    python train-v2.py --data_dir data/exp3_clean --exp_name clean

   ```

4. Train the CLEAN dataset on Finetune and then Finetune + LoRA experiments:

   ```powershell
   python src/train-v3-finetune.py --data_dir data/exp3_clean --exp_name clean_finetune
   python src/train-v4-lora-finetune.py --data_dir data/exp3_clean --exp_name clean_finetune_lora

   ```

5. Train the Qwen model on the balanced dataset

   ```powershell
   python src/Qwen_Train_Eval.py

   ```

6. Make predictions in a Gradio based web interface after training both the Qwen and DistilBERT + LoRA model:

   ```powershell
   python src/predict-ui.py
   ```
