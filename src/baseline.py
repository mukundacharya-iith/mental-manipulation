import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)

from utils.logger import (
    init_results_dir,
    log_to_file,
    save_metrics,
    save_confusion_matrix,
    save_report
)


def load_data():
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")
    test_df = pd.read_csv("data/processed/processed_con.csv")

    return train_df, val_df, test_df


def create_vectorizer():
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2)
    )
    return vectorizer


def train_model(vectorizer, train_df):
    X_train = vectorizer.fit_transform(train_df["text"])
    y_train = train_df["label"]

    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(X_train, y_train)

    return model


def evaluate(model, vectorizer, df, name="Dataset"):
    X = vectorizer.transform(df["text"])
    y_true = df["label"]

    y_pred = model.predict(X)

    # metrics
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    precision = precision_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")

    print(f"\n{name} Results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 (Macro): {f1_macro:.4f}")
    print(f"F1 (Weighted): {f1_weighted:.4f}")
    print(f"Precision (Macro): {precision:.4f}")
    print(f"Recall (Macro): {recall:.4f}")

    cm = confusion_matrix(y_true, y_pred)

    print("\nConfusion Matrix:")
    print(cm)

    report = classification_report(
        y_true,
        y_pred,
        target_names=["control", "manipulation"]
    )

    print("\nClassification Report:")
    print(report)

    return {
        "name": name,
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "precision_macro": precision,
        "recall_macro": recall,
        "confusion_matrix": cm,
        "report": report
    }


def main():
    init_results_dir()

    LOG_FILE = "results/baseline_log.txt"
    METRICS_FILE = "results/baseline_metrics.csv"

    train_df, val_df, test_df = load_data()

    print("\nDataset Sizes:")
    print(f"Train: {len(train_df)}")
    print(f"Validation: {len(val_df)}")
    print(f"Test (CON): {len(test_df)}")

    vectorizer = create_vectorizer()

    model = train_model(vectorizer, train_df)

    for name, df in [
        ("Train", train_df),
        ("Validation", val_df),
        ("Test_CON", test_df)
    ]:
        result = evaluate(model, vectorizer, df, name)

        log_to_file(f"\n{name} Results:\n{result}", LOG_FILE)

        save_metrics(result, METRICS_FILE)

        save_confusion_matrix(
            result["confusion_matrix"],
            f"results/baseline_confusion_{name}.csv"
        )

        save_report(
            result["report"],
            f"results/baseline_report_{name}.txt"
        )


if __name__ == "__main__":
    main()