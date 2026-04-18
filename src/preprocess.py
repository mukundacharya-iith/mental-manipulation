import pandas as pd
import os
from sklearn.model_selection import train_test_split

def load_csv(path):
    df = pd.read_csv(path)
    return df

def inspect(df, name="dataset"):
    print(f"\n{name} shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nSample rows:")
    print(df.head())

    print("\nManipulation distribution:")
    print(df["Manipulative"].value_counts())

    print("\nTechnique distribution:")
    print(df["Technique"].value_counts())

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

def process_label(df):
    df["label"] = df["label"].astype(int)
    return df

def process_technique(df):
    df["technique"] = df["technique"].fillna("control")
    df["technique"] = df["technique"].astype(str).str.lower().str.strip()
    return df

def preprocess(df):
    df = clean_and_standardize(df)
    df = process_label(df)
    df = process_technique(df)

    return df

def save_csv(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved → {path}")

def main():
    maj_path = "data/raw/mentalmanip_maj.csv"
    con_path = "data/raw/mentalmanip_con.csv"

    maj_out = "data/processed/processed_maj.csv"
    con_out = "data/processed/processed_con.csv"

    maj_df = load_csv(maj_path)
    con_df = load_csv(con_path)

    inspect(maj_df, "MAJ")
    inspect(con_df, "CON")

    maj_df = preprocess(maj_df)
    con_df = preprocess(con_df)

    save_csv(maj_df, maj_out)
    save_csv(con_df, con_out)

    train_df, val_df = train_test_split(
        maj_df,
        test_size=0.2,
        stratify=maj_df["label"],
        random_state=42
    )
    save_csv(train_df, "data/processed/train.csv")
    save_csv(val_df, "data/processed/val.csv")
    print("\nSplit sizes:")
    print(f"Train size: {len(train_df)}")
    print(f"Validation size: {len(val_df)}")
    print(f"Test size (CON): {len(con_df)}")

    print("\nTrain label distribution:")
    print(train_df["label"].value_counts())

    print("\nValidation label distribution:")
    print(val_df["label"].value_counts())

    print("\nTest (CON) label distribution:")
    print(con_df["label"].value_counts())


if __name__ == "__main__":
    main()