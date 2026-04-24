import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# 🔥 CHANGE THIS PATH if needed
MODEL_PATH = "models/clean-finetune-lora/clean-finetune-lora"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model + tokenizer
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True).to(device)
model.eval()


def predict(text, threshold=0.6):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)

    prob_manip = probs[0][1].item()
    prob_control = probs[0][0].item()

    label = "MANIPULATION" if prob_manip > threshold else "CONTROL"

    return {
        "label": label,
        "manipulation_score": round(prob_manip, 4),
        "control_score": round(prob_control, 4)
    }


if __name__ == "__main__":
    while True:
        text = input("\nEnter text (or 'q' to quit): ")

        if text.lower() == "q":
            break

        result = predict(text)

        print("\nPrediction:")
        print(f"Label: {result['label']}")
        print(f"Manipulation Score: {result['manipulation_score']}")
        print(f"Control Score: {result['control_score']}")