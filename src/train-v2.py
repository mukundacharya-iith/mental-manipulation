import pandas as pd
import os
import torch
import argparse
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)

from dataset import ManipulationDataset
from utils.logger import (
    init_results_dir,
    log_to_file,
    save_metrics,
    save_confusion_matrix,
    save_report
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ✅ UPDATED: dynamic dataset loading
def load_data(base_path):
    train_df = pd.read_csv(f"{base_path}/train.csv")
    val_df = pd.read_csv(f"{base_path}/val.csv")
    test_df = pd.read_csv(f"{base_path}/test.csv")
    return train_df, val_df, test_df


def get_tokenizer():
    return DistilBertTokenizer.from_pretrained("distilbert-base-uncased")


def create_dataloaders(train_df, val_df, test_df, tokenizer):
    train_dataset = ManipulationDataset(
        train_df["text"].tolist(),
        train_df["label"].tolist(),
        tokenizer
    )

    val_dataset = ManipulationDataset(
        val_df["text"].tolist(),
        val_df["label"].tolist(),
        tokenizer
    )

    test_dataset = ManipulationDataset(
        test_df["text"].tolist(),
        test_df["label"].tolist(),
        tokenizer
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    test_loader = DataLoader(test_dataset, batch_size=16)

    return train_loader, val_loader, test_loader


# ✅ FIXED: proper dropout usage
def get_model():
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2,
        dropout=0.2,
        attention_dropout=0.2
    )
    return model.to(device)


def get_class_weights(train_df):
    counts = train_df["label"].value_counts().sort_index()
    total = counts.sum()

    weights = total / (2 * counts)
    return torch.tensor(weights.values, dtype=torch.float).to(device)


def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0

    for batch in loader:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        logits = outputs.logits
        loss = loss_fn(logits, labels)

        loss.backward()

        # ✅ ADDED: gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, threshold=0.5, name="Dataset"):
    model.eval()

    preds = []
    true = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            probs = torch.softmax(outputs.logits, dim=1)
            pred = (probs[:, 1] > threshold).long()

            preds.extend(pred.cpu().tolist())
            true.extend(labels.cpu().tolist())

    acc = accuracy_score(true, preds)
    f1_macro = f1_score(true, preds, average="macro")
    f1_weighted = f1_score(true, preds, average="weighted")
    precision = precision_score(true, preds, average="macro")
    recall = recall_score(true, preds, average="macro")

    print(f"\n{name} Results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 (Macro): {f1_macro:.4f}")
    print(f"F1 (Weighted): {f1_weighted:.4f}")
    print(f"Precision (Macro): {precision:.4f}")
    print(f"Recall (Macro): {recall:.4f}")

    cm = confusion_matrix(true, preds)
    print("\nConfusion Matrix:")
    print(cm)

    report = classification_report(true, preds, target_names=["control", "manipulation"])
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--exp_name", type=str, required=True)
    args = parser.parse_args()

    init_results_dir()

    LOG_FILE = f"results-v2/{args.exp_name}_log.txt"
    METRICS_FILE = f"results-v2/{args.exp_name}_metrics.csv"

    train_df, val_df, test_df = load_data(args.data_dir)

    tokenizer = get_tokenizer()
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df, tokenizer
    )

    model = get_model()

    # ✅ UPDATED LR (more stable)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    class_weights = get_class_weights(train_df)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    best_f1 = 0
    epochs = 5

    for epoch in range(epochs):
        loss = train_one_epoch(model, train_loader, optimizer, loss_fn)

        print(f"\nEpoch {epoch+1}")
        print(f"Train Loss: {loss:.4f}")

        log_to_file(f"\nEpoch {epoch+1} Loss: {loss:.4f}", LOG_FILE)

        val_result = evaluate(model, val_loader, threshold=0.5, name="Validation")

        log_to_file(f"\nValidation Results:\n{val_result}", LOG_FILE)
        save_metrics(val_result, METRICS_FILE)

        save_confusion_matrix(
            val_result["confusion_matrix"],
            f"results-v2/{args.exp_name}_confusion_val.csv"
        )

        save_report(
            val_result["report"],
            f"results-v2/{args.exp_name}_report_val.txt"
        )

        val_f1 = val_result["f1_macro"]

        if val_f1 > best_f1:
            best_f1 = val_f1
            model_dir = f"models/bert/{args.exp_name}"
            os.makedirs(model_dir, exist_ok=True)
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
            print("Saved best model to {model_dir}")
            log_to_file("Saved best model to {mode_dir}", LOG_FILE)

    print("\nFinal Test:")

    test_result = evaluate(model, test_loader, threshold=0.5, name="Test")

    log_to_file(f"\nTest Results:\n{test_result}", LOG_FILE)
    save_metrics(test_result, METRICS_FILE)

    save_confusion_matrix(
        test_result["confusion_matrix"],
        f"results-v2/{args.exp_name}_confusion_test.csv"
    )

    save_report(
        test_result["report"],
        f"results-v2/{args.exp_name}_report_test.txt"
    )


if __name__ == "__main__":
    main()