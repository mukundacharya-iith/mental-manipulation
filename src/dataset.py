import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder


class ManipulationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128, mode="binary"):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mode = mode
        if mode == "binary":
            self.labels = labels

        else:
            self.label_encoder = LabelEncoder()
            self.labels = self.label_encoder.fit_transform(labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }