import pandas as pd
import os
from sklearn.model_selection import train_test_split
import csv


def load_csv(path):
    data = []
    columns = None

    with open(path, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        for idx, row in enumerate(reader):
            if idx == 0:
                columns = row
            else:
                if len(row) != len(columns):
                    continue
                data.append(row)

    df = pd.DataFrame(data, columns=columns)

    if 'ID' in df.columns:
        df = df.drop(['ID'], axis=1)

    return df


def clean_and_standardize(df):
    df = df[["Dialogue", "Manipulative", "Technique"]]

    df = df.rename(columns={
        "Dialogue": "text",
        "Manipulative": "label",
        "Technique": "technique"
    })

    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str).str.strip()

    return df


def preprocess(df):
    df = clean_and_standardize(df)
    df["label"] = df["label"].astype(int)
    df["technique"] = df["technique"].fillna("control").astype(str).str.lower().str.strip()
    return df


def save_csv(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved → {path}")


def main():
    maj_path = "data/raw/mentalmanip_maj.csv"
    con_path = "data/raw/mentalmanip_con.csv"

    maj_df = preprocess(load_csv(maj_path))
    con_df = preprocess(load_csv(con_path))

    # =========================
    # 1. MAJ ONLY (80/10/10)
    # =========================
    maj_train, maj_temp = train_test_split(
        maj_df, test_size=0.2, stratify=maj_df["label"], random_state=42
    )

    maj_val, maj_test = train_test_split(
        maj_temp, test_size=0.5, stratify=maj_temp["label"], random_state=42
    )

    save_csv(maj_train, "data/exp1_maj/train.csv")
    save_csv(maj_val, "data/exp1_maj/val.csv")
    save_csv(maj_test, "data/exp1_maj/test.csv")


    # =========================
    # 2. CON ONLY (80/10/10)
    # =========================
    con_train, con_temp = train_test_split(
        con_df, test_size=0.2, stratify=con_df["label"], random_state=42
    )

    con_val, con_test = train_test_split(
        con_temp, test_size=0.5, stratify=con_temp["label"], random_state=42
    )

    save_csv(con_train, "data/exp2_con/train.csv")
    save_csv(con_val, "data/exp2_con/val.csv")
    save_csv(con_test, "data/exp2_con/test.csv")


    # =========================
    # 3. CLEAN PROTOCOL
    # =========================

    # Step 1: split CON
    con_train_pool, con_test = train_test_split(
        con_df, test_size=0.1, stratify=con_df["label"], random_state=42
    )

    # Step 2: identify ambiguous MAJ (MAJ - CON)
    # NOTE: using text match since ID was dropped
    con_texts = set(con_df["text"])
    maj_ambiguous = maj_df[~maj_df["text"].isin(con_texts)]

    print("Ambiguous MAJ size:", len(maj_ambiguous))

    # Step 3: combine training pool
    train_pool = pd.concat([con_train_pool, maj_ambiguous], ignore_index=True)

    train_df, val_df = train_test_split(
        train_pool, test_size=0.1, stratify=train_pool["label"], random_state=42
    )

    # Save
    save_csv(train_df, "data/exp3_clean/train.csv")
    save_csv(val_df, "data/exp3_clean/val.csv")
    save_csv(con_test, "data/exp3_clean/test.csv")
    
    # ==========================================
    #   4: BALANCE THE TRAIN POOL 
    # ==========================================
    # Separate the classes within the combined training pool
    class_0 = train_pool[train_pool["label"] == 0] # Non-Manipulative
    class_1 = train_pool[train_pool["label"] == 1] # Manipulative
    
    # Find the size of the minority class
    min_size = min(len(class_0), len(class_1))
    
    # Undersample the majority class to match the minority class
    class_0_balanced = class_0.sample(n=min_size, random_state=42)
    class_1_balanced = class_1.sample(n=min_size, random_state=42)
    
    # Recombine into a perfectly balanced training pool
    balanced_train_pool = pd.concat([class_0_balanced, class_1_balanced]).sample(frac=1, random_state=42).reset_index(drop=True)
    # ==========================================
    
    # Step 4: split the BALANCED train pool → train/val
    train_df_balanced, val_df_balanced = train_test_split(
        balanced_train_pool, test_size=0.1, stratify=balanced_train_pool["label"], random_state=42
    )

    # Save
    save_csv(train_df_balanced, "data/balanced/train.csv")
    save_csv(val_df_balanced, "data/balanced/val.csv")
    save_csv(con_test, "data/balanced/test.csv")

    print("\nFinal Sizes:")
    print(f"Exp1 MAJ Train: {len(maj_train)}")
    print(f"Exp2 CON Train: {len(con_train)}")
    print(f"Exp3 Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(con_test)}")


if __name__ == "__main__":
    main()