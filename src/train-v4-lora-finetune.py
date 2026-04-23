import pandas as pd
import os
import torch
import argparse
import numpy as np

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

from peft import LoraConfig, get_peft_model, TaskType

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

    return (
        DataLoader(train_dataset, batch_size=16, shuffle=True),
        DataLoader(val_dataset, batch_size=16),
        DataLoader(test_dataset, batch_size=16)
    )


def get_model(exp_name):
    base_model = DistilBertForSequenceClassification.from_pretrained(
        f"models/clean-finetune/clean-finetune"
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_lin", "v_lin"]
    )

    model = get_peft_model(base_model, lora_config)

    model.print_trainable_parameters()

    return model.to(device)


def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0

    for batch in loader:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        loss = loss_fn(logits, labels)
        loss.backward()

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

    cm = confusion_matrix(true, preds)
    report = classification_report(true, preds, target_names=["control", "manipulation"])

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


def find_best_threshold(model, loader):
    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)[:, 1]

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    best_f1 = 0
    best_t = 0.5

    for t in np.arange(0.4, 0.7, 0.05):
        preds = (all_probs > t).astype(int)
        f1 = f1_score(all_labels, preds, average="macro")

        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    return best_t, best_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--exp_name", type=str, required=True)
    args = parser.parse_args()

    init_results_dir()

    LOG_FILE = f"results-v4/{args.exp_name}_log.txt"
    METRICS_FILE = f"results-v4/{args.exp_name}_metrics.csv"

    train_df, val_df, test_df = load_data(args.data_dir)

    tokenizer = get_tokenizer()
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df, tokenizer
    )

    model = get_model(args.exp_name)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    class_weights = torch.tensor([1.69, 0.71]).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    best_f1 = 0
    best_threshold = 0.5

    for epoch in range(5):
        loss = train_one_epoch(model, train_loader, optimizer, loss_fn)

        print(f"\nEpoch {epoch+1}")
        print(f"Train Loss: {loss:.4f}")

        log_to_file(f"\nEpoch {epoch+1} Loss: {loss:.4f}", LOG_FILE)

        best_t, tuned_f1 = find_best_threshold(model, val_loader)

        print(f"Best Threshold: {best_t:.2f} | Tuned F1: {tuned_f1:.4f}")

        val_result = evaluate(model, val_loader, threshold=best_t, name="Validation")
        val_result["best_threshold"] = best_t

        log_to_file(f"\nValidation Results:\n{val_result}", LOG_FILE)
        save_metrics(val_result, METRICS_FILE)

        save_confusion_matrix(
            val_result["confusion_matrix"],
            f"results-v4/{args.exp_name}_confusion_val.csv"
        )

        save_report(
            val_result["report"],
            f"results-v4/{args.exp_name}_report_val.txt"
        )

        if tuned_f1 > best_f1:
            best_f1 = tuned_f1
            best_threshold = best_t

            model_dir = f"models/clean-finetune-lora/{args.exp_name}"
            os.makedirs(model_dir, exist_ok=True)

            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)

            print(f"Saved best model to {model_dir}")
            log_to_file(f"Saved best model to {model_dir}", LOG_FILE)

    print(f"\nUsing best threshold: {best_threshold}")

    test_result = evaluate(model, test_loader, threshold=best_threshold, name="Test")

    log_to_file(f"\nTest Results:\n{test_result}", LOG_FILE)
    save_metrics(test_result, METRICS_FILE)

    save_confusion_matrix(
        test_result["confusion_matrix"],
        f"results-v4/{args.exp_name}_confusion_test.csv"
    )

    save_report(
        test_result["report"],
        f"results-v4/{args.exp_name}_report_test.txt"
    )


if __name__ == "__main__":
    main()