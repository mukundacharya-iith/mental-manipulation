import pandas as pd
import os

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


if __name__ == "__main__":
    main()