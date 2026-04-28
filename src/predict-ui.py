import gradio as gr
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    DistilBertTokenizer, 
    DistilBertForSequenceClassification
)
from peft import PeftModel

# ==========================================
# 1. LOAD BOTH MODELS
# ==========================================
print("Loading models... please wait.")

BERT_BASE_PATH = "models/clean-finetune/clean-finetune"
BERT_LORA_PATH = "models/clean-finetune-lora/clean-finetune-lora"
QWEN_BASE_NAME = "models/Qwen_1.5_BaseModel"
QWEN_ADAPTER_PATH = "models/qwen-manipulation-detector-model"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load DistilBERT
bert_tokenizer = DistilBertTokenizer.from_pretrained(BERT_BASE_PATH, local_files_only=True)
bert_base_model = DistilBertForSequenceClassification.from_pretrained(BERT_BASE_PATH, local_files_only=True)
bert_model = PeftModel.from_pretrained(bert_base_model, BERT_LORA_PATH).to(device)
bert_model.eval()

# Load Qwen
qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_ADAPTER_PATH)
qwen_base_model = AutoModelForCausalLM.from_pretrained(
    QWEN_BASE_NAME, 
    dtype=torch.bfloat16, 
    device_map="auto",
    offload_folder="models/offload",
    trust_remote_code=True
)
qwen_model = PeftModel.from_pretrained(qwen_base_model, QWEN_ADAPTER_PATH, offload_folder="models/offload")
qwen_model.eval()

print("Models loaded successfully!")

# ==========================================
# 2. UNIFIED PREDICTION LOGIC
# ==========================================
def detect_dual(dialogue):
    if not dialogue.strip():
        yield "Please enter text.", "Please enter text."
        return

    yield "Processing... ⏳", "Processing... ⏳"

    # --- DistilBERT pass ---
    bert_inputs = bert_tokenizer(dialogue, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        bert_outputs = bert_model(**bert_inputs)
        probs = torch.softmax(bert_outputs.logits, dim=1)

    m_score = probs[0][1].item()
    c_score = probs[0][0].item()
    
    if m_score > 0.6:
        bert_res = f"⚠️ Manipulative\nManipulation: {m_score*100:.1f}%\nControl: {c_score*100:.1f}%"
    else:
        bert_res = f"✅ Non-Manipulative\nManipulation: {m_score*100:.1f}%\nControl: {c_score*100:.1f}%"

    yield bert_res, "Processing... ⏳"

    # --- Qwen pass ---
    messages = [
        {"role": "system", "content": "You are an expert psychological analyst. Analyze for manipulation. Answer only 'Yes' or 'No'."},
        {"role": "user", "content": dialogue}
    ]
    qwen_text = qwen_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    qwen_inputs = qwen_tokenizer([qwen_text], return_tensors="pt").to(qwen_model.device)
    
    with torch.no_grad():
        generated_ids = qwen_model.generate(**qwen_inputs, max_new_tokens=5, temperature=0.1, do_sample=False, pad_token_id=qwen_tokenizer.pad_token_id)
        
    new_ids = generated_ids[0][len(qwen_inputs.input_ids[0]):]
    qwen_resp = qwen_tokenizer.decode(new_ids, skip_special_tokens=True).strip().lower()
    
    if "yes" in qwen_resp:
        qwen_res = "⚠️ Manipulative (Model answered: Yes)"
    else:
        qwen_res = "✅ Non-Manipulative (Model answered: No)"

    yield bert_res, qwen_res

# ==========================================
# 3. UI LAYOUT
# ==========================================
with gr.Blocks(theme="soft", title="Psychological Manipulation Detector") as demo:
    gr.Markdown("# 🧠 Psychological Manipulation Detector")
    gr.Markdown("Powered by Qwen 1.5B and DistilBERT. Enter a conversation to detect signs of gaslighting, guilt-tripping, or coercion.")
    
    input_box = gr.Textbox(lines=5, placeholder="Enter a dialogue or conversation here...", label="Input Conversation")
    
    with gr.Row():
        submit_btn = gr.Button("Analyze", variant="primary")
        clear_btn = gr.Button("Clear")

    with gr.Row():
        bert_output = gr.Textbox(label="Analysis by DistilBERT + LoRA", lines=4)
        qwen_output = gr.Textbox(label="Analysis by Qwen1.5B", lines=4)

    # Functionality
    submit_btn.click(fn=detect_dual, inputs=input_box, outputs=[bert_output, qwen_output])
    
    # Clear button resets everything
    clear_btn.click(lambda: ["", "", ""], None, [input_box, bert_output, qwen_output])

if __name__ == "__main__":
    demo.launch()