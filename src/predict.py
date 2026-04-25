import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from peft import PeftModel

# 🔥 PATHS (update if needed)
BASE_MODEL_PATH = "models/clean-finetune/clean-finetune"
LORA_MODEL_PATH = "models/clean-finetune-lora/clean-finetune-lora"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading model... please wait ⏳")

# ---------------- LOAD TOKENIZER ----------------
tokenizer = DistilBertTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    local_files_only=True
)

# ---------------- LOAD BASE MODEL ----------------
base_model = DistilBertForSequenceClassification.from_pretrained(
    BASE_MODEL_PATH,
    local_files_only=True
)

# ---------------- ATTACH LORA ----------------
model = PeftModel.from_pretrained(
    base_model,
    LORA_MODEL_PATH
).to(device)

# Optional: merge for faster inference
# model = model.merge_and_unload()

model.eval()

print("Model loaded successfully ✅")


# ---------------- PREDICT FUNCTION ----------------
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


# ---------------- CLI LOOP ----------------
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