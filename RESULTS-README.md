# Training Pipeline & Results (DistilBERT → CLEAN → Fine-tune → LoRA)

## Overview

This document summarizes the full training progression for manipulation detection:

```text
ALL datasets → CLEAN selection → Fine-tuned BERT → LoRA-enhanced model
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
* Control recall: **~0.38 ❌**
* Manipulation recall: **~0.88 ⚠️**

### Issue

* Model heavily biased toward **manipulation**
* High accuracy but poor minority detection

---

## Step 3 — Fine-Tuned CLEAN Model

Improvements applied:

* ✅ Class weights: `[1.69, 0.71]`
* ✅ Threshold tuning (dynamic, not fixed)
* ✅ Model selection using **macro F1**

### Results

* Accuracy: **~0.67**
* Macro F1: **~0.65**
* Control recall: **~0.66 ✅**
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

### Tradeoff

* Improved overall performance
* Slight shift back toward manipulation
* Still far more balanced than original model

---

## Final Comparison

| Metric            | Original CLEAN (No weights) | Full Fine-tune (Weights) | LoRA + Weights |
| ----------------- | --------------------------- | ------------------------ | -------------- |
| Macro F1          | 0.635                       | 0.649                    | **0.658**      |
| Accuracy          | **0.723**                   | 0.675                    | **0.709**      | 
| Control Recall    | 0.38                        | **0.66**                 | 0.52           |
| Manipulation F1   | 0.81                        | 0.74                     | **0.79**       |
| Control Precision | 0.58                        | 0.48                     | **0.53**       |

---

### Takeaway

* Class weighting + threshold tuning **fixed severe class imbalance** (control recall: 0.38 → 0.66).
* LoRA **improved overall performance (Macro F1, accuracy)** while maintaining reasonable balance.
* Final model (**CLEAN + LoRA**) offers the best trade-off between performance and class fairness.

---

## Final Model

```text
DistilBERT (CLEAN) + Class Weights + Threshold Tuning + LoRA
```

Best balance of:

* performance
* generalization
* efficiency

---
