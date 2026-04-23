# Dataset Preparation & Preprocessing

## 1. Overview

This project uses the **MentalManip dataset** for detecting manipulative language in conversations.

We work with two dataset variants:

* **mentalmanip_maj.csv** → Majority annotations (noisy labels)
* **mentalmanip_con.csv** → Consensus annotations (high-quality labels)

Both datasets contain the **same dialogues**, but differ in label quality.

---

## 2. Dataset Variants

### Majority Dataset (mentalmanip_maj.csv)

* Larger dataset (~4000 samples)
* Labels based on **majority agreement**
* Contains **noisy / ambiguous labels**

---

### Consensus Dataset (mentalmanip_con.csv)

* Smaller dataset (~2915 samples)
* Labels with **full annotator agreement**
* Considered **high-quality ground truth**

---

## 3. Task Definition

### Binary Manipulation Detection

| Label | Meaning                   |
| ----- | ------------------------- |
| 0     | Control (no manipulation) |
| 1     | Manipulation present      |

---

### Technique Classification (Auxiliary)

Examples include:

* Denial
* Evasion
* Rationalization
* Playing the Victim Role
* Shaming or Belittlement
* Intimidation
* Persuasion or Seduction

---

## 4. Experimental Setups

Since MAJ and CON share the same dialogues, we design **three evaluation strategies**.

---

### Experiment 1: MAJ Only (Noisy Baseline)

```text
Train → MAJ (80%)
Val   → MAJ (10%)
Test  → MAJ (10%)
```

**Purpose:**

* Evaluate performance on noisy labels
* Establish baseline

---

### Experiment 2: CON Only (Clean Upper Bound)

```text
Train → CON (80%)
Val   → CON (10%)
Test  → CON (10%)
```

**Purpose:**

* Measure best possible performance
* Understand impact of clean annotations

---

### Experiment 3: Clean Protocol (No Leakage)

This setup ensures **no dialogue overlap between train and test**.

#### Step 1: Split CON

```text
CON → 90% Training Pool + 10% Test
```

* Test set is **clean and unseen**

---

#### Step 2: Identify Ambiguous Data

```text
Ambiguous = MAJ - CON
```

* Dialogues where annotators disagreed

---

#### Step 3: Build Training Pool

```text
Training Pool =
    90% CON (clean)
  + MAJ - CON (ambiguous)
```

---

#### Step 4: Final Split

```text
Train → 90% of Training Pool
Val   → 10% of Training Pool
Test  → 10% CON (held-out clean set)
```

---

## 5. Generated Files

After preprocessing:

```text
data/
├── exp1_maj/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
│
├── exp2_con/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
│
├── exp3_clean/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
```

---

## 6. Dataset Sizes (Approx)

| Experiment | Train | Val | Test |
| ---------- | ----- | --- | ---- |
| MAJ        | 3200  | 400 | 400  |
| CON        | 2332  | 292 | 292  |
| Clean      | 3337  | 371 | 292  |

---

## 7. Preprocessing Steps

Minimal preprocessing is applied:

### Step 1: Load CSV

```python
df = pd.read_csv(path)
```

### Step 2: Select Columns

```text
Dialogue, Manipulative, Technique
```

### Step 3: Rename

```text
Dialogue     → text  
Manipulative → label  
Technique    → technique
```

### Step 4: Handle Missing Values

* Drop rows with missing text/label
* Fill missing technique with `"control"`

### Step 5: Clean Text

* Convert to string
* Strip whitespace

### Step 6: Normalize Labels

* Binary labels → int (0/1)
* Technique → lowercase

---

## 8. Final Schema

| Column    | Description        |
| --------- | ------------------ |
| text      | Input dialogue     |
| label     | Binary label (0/1) |
| technique | Manipulation type  |

---

## 9. Design Decisions

* Do not merge MAJ and CON blindly (to avoid leakage)
* Use **multiple setups** to analyze label quality
* Keep preprocessing minimal
* Focus on binary classification

---

## 10. Summary

* MAJ → noisy but large
* CON → clean but smaller
* Clean protocol → best practical setup

---
