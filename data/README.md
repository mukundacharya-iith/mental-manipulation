# Dataset Preparation & Preprocessing

## 1. Overview

This project uses the **MentalManip dataset** for detecting manipulative language in conversations.

We work with the processed CSV files provided in the experiments folder of the dataset:

* `mentalmanip_maj.csv` (majority annotations)
* `mentalmanip_con.csv` (consensus annotations)

---

## 2. Dataset Variants

### Majority Dataset (`mentalmanip_maj.csv`)

* Larger dataset
* Labels based on majority agreement
* Will be used for **training**

### Consensus Dataset (`mentalmanip_con.csv`)

* Smaller dataset
* Labels with full annotator agreement
* Will be used for **testing** as we have a consesnus among all annotators, meaning this is high quality data.

---

## 3. Task Definition

**Binary Manipulation Detection**

| Label | Meaning                   |
| ----- | ------------------------- |
| 0     | Control (no manipulation) |
| 1     | Manipulation present      |

**Technique Classification**

The full labels include:

* Denial
* Evasion
* Feigning Innocence
* Rationalization
* Playing the Victim Role
* Playing the Servant Role
* Shaming or Belittlement
* Intimidation
* Brandishing Anger
* Accusation
* Persuasion or Seduction

---

## 4. Data Split Strategy

We will follow the below evaluation setup:

```text
Train       → mentalmanip_maj (80%)
Validation  → mentalmanip_maj (20%)
Test        → mentalmanip_con (100%)
```

### Generated Files

After running preprocessing:

```text
data/processed/
├── processed_maj.csv
├── processed_con.csv
├── train.csv
├── val.csv
```

### Reproducibility

The split is performed using:

```python
train_test_split(
    maj_df,
    test_size=0.2,
    stratify=maj_df["label"],
    random_state=42
)
```

This ensures:

* Same split across runs
* Balanced class distribution

---

### Dataset Sizes

```text
Train size: 3200
Validation size: 800
Test size: 2915
```

---

### Label Distribution Check

We verify that the class distribution is preserved:

```text
Train:
0 → 2254
1 → 946

Validation:
0 → 564
1 → 236

Test (CON):
0 → 2016
1 → 899
```

---

## 5. Preprocessing Steps

We perform minimal preprocessing to preserve alignment with the original dataset.

### Step 1: Load CSV

```python
df = pd.read_csv(path)
```

---

### Step 2: Select Relevant Columns

We retain only the necessary columns:

```text
Dialogue, Manipulative, Technique
```

---

### Step 3: Rename Columns

To standardize the pipeline:

```text
Dialogue     → text  
Manipulative → label
Technique    → technique
```

---

### Step 4: Handle Missing Values

* Remove rows with missing text or labels
* Fill missing techniques with `"control"`

---

### Step 5: Clean Text

* Convert to string
* Strip leading/trailing spaces

---

### Step 6: Normalize Labels

#### Binary Label:

```text
0 → control  
1 → manipulation
```

#### Technique:

* Convert to lowercase
* Normalize formatting

---

## 6. Output Format

After preprocessing, both datasets are saved as:

```text
data/processed/processed_maj.csv  
data/processed/processed_con.csv
```

---

### Final Schema

| Column    | Description                     |
| --------- | ------------------------------- |
| text      | Input dialogue                  |
| label     | Binary manipulation label (0/1) |
| technique | Type of manipulation            |

---

### Example

```text
text                                                                                         label   technique
----------------------------------------------------------------------------------------------------------------------
"You will or you'll be back at BET so quick you'll never know what hit you"                  1       shaming or belittlement
"I couldn't help but notice you pain"                                                        0       control
```

---

## 7. Design Decisions

* We **do not modify dataset structure heavily** to remain consistent with prior work.
* We **do not merge MAJ and CON datasets** to avoid label inconsistency.
* We ignore **vulnerability labels** as they are not required for the primary task.

---

## 8. Summary

* Use MAJ for training and validation
* Use CON for final evaluation
* Perform minimal preprocessing
* Focus on binary classification as the core task

---

## 9. How to Run

```bash
python src/preprocess.py
```

This will generate the processed datasets required for training and evaluation.
