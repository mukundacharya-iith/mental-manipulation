# Training Pipeline & Results (DistilBERT → CLEAN → Fine-tune → LoRA → QWEN1.5B)

## Overview

This document summarizes the full training progression for manipulation detection:

```text
ALL datasets → CLEAN selection → Fine-tuned BERT → LoRA-enhanced model → QWEN1.5B
```

The focus was on improving **class balance and real-world performance**, not just accuracy.

---

## Step 1 — Initial Models (ALL: MAJ / CON / CLEAN)

Three dataset strategies were tested:

* **MAJ** (majority skewed)
* **CON** (combined)
* **CLEAN** (filtered dataset)

### Observation

* CLEAN consistently outperformed others
* MAJ showed strong bias toward manipulation
* CON was more balanced but lower performance

**Decision:** Proceed with CLEAN dataset

---

## Step 2 — CLEAN Baseline (Raw BERT)

* Model: `DistilBERT`
* Default threshold: `0.5`
* No class weighting

### Results

* Accuracy: **~0.72**
* Macro F1: **~0.63**
* Control recall: **~0.38**
* Manipulation recall: **~0.88**

### Issue

* Model heavily biased toward **manipulation**
* High accuracy but poor minority detection

---

## Step 3 — Fine-Tuned CLEAN Model

Improvements applied:

* Class weights: `[1.69, 0.71]`
* Threshold tuning (dynamic, not fixed)
* Model selection using **macro F1**

### Results

* Accuracy: **~0.67**
* Macro F1: **~0.65**
* Control recall: **~0.66**
* Manipulation recall: **~0.68**

### Impact

* Bias significantly reduced
* Balanced classification achieved
* Slight drop in accuracy (expected)

---

## Step 4 — CLEAN + LoRA

LoRA applied on top of fine-tuned CLEAN model:

* Target layers: `q_lin`, `v_lin`
* Only small % parameters trained
* Threshold re-tuned after training

### Results

* Accuracy: **~0.71**
* Macro F1: **~0.66 (best)**
* Control recall: **~0.52**
* Manipulation recall: **~0.79**

### Impact

* Improved overall performance
* Slight shift back toward manipulation
* Still far more balanced than original model

### Tradeoff

* Quick for real world as it is a small model
* May not be as accurate as the higer paramter models
* Easy to train as only ~1-2% of total parameters are trained out of the total 66M paramters

---

## Step 5 — Generative Qwen + LoRA (Balanced)

LoRA applied to Qwen2.5-1.5B-Instruct using the balanced training pool:

* Target layers: `q_proj`, `k_proj`, `v_proj`, `o_proj`
* Quantization: 4-bit NF4 for efficiency
* Runs 3 epochs of Supervised Fine-Tuning (SFT).
* Training: 3 epochs on a 1:1 class-ratio dataset

### Results

* Accuracy: **~0.76 (best)**
* Macro F1: **~0.71 (best)**
* Control recall: **~0.56**
* Manipulation recall: **~0.85 (best)**

### Impact

* Best performing model across all the experiments
* Context Awareness: Much better at understanding nuance, sarcasm, and intent than the shallow classifier.
* Balanced Bias: Effectively solves the "majority class bias" by training on equal samples of both labels.

### Tradeoff

* Speed: Notably slower than DistilBERT; ~1.5s per inference (vs. ~20ms for DistilBERT)
* Resource Demand: Requires significantly more VRAM and compute compared to the DistilBERT approach.
* Still far more balanced than original model

---

## Final Comparison


| Metric                | Original CLEAN | Fine-tuned CLEAN | CLEAN + LoRA |QWEN Model|
|-----------------------|----------------|------------------|--------------|----------|
| **Accuracy**          | 0.72           | 0.67             | 0.71         | **0.76** |
| **Macro F1**          | 0.64           | 0.65             | 0.66         | **0.71** |
| **Control Precision** | 0.58           | 0.48             | 0.53         | **0.62** |
| **Control Recall**    | 0.38           | **0.66**         | 0.52         | 0.56     |
| **Control F1**        | 0.46           | 0.55             | 0.53         | **0.59** |
| **Manip Precision**   | 0.76           | 0.74             | 0.79         | **0.81** |
| **Manip Recall**      | **0.88**       | 0.68             | 0.79         | 0.85     |
| **Manip F1**          | 0.81           | 0.74             | 0.79         | **0.83** |


---
