import csv
import os
import random
from datasets import Dataset
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def load_and_format_data(csv_file_path):
    formatted_data = []
    
    with open(csv_file_path, 'r', newline='', encoding='utf-8') as infile:
        # Using DictReader to map columns by header name automatically
        content = csv.DictReader(infile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        
        for row in content:
            dialogue = row['text']
            # Map the binary label to a text response for the Instruct model
            label_text = "Yes" if row['label'] == '1' else "No"
            
            # Format according to Qwen's expected chat template
            messages = [
                {
                    "role": "system", 
                    "content": "You are an expert psychological analyst. Analyze the following conversation to determine if it contains mental manipulation. Be highly conservative. Do not flag normal arguments, disagreements, or emotional expressions as manipulation. Only answer 'Yes' if there is clear evidence of gaslighting, intimidation, or guilt-tripping. Otherwise, answer 'No'."
                },
                {
                    "role": "user", 
                    "content": dialogue
                },
                {
                    "role": "assistant", 
                    "content": label_text
                }
            ]
            formatted_data.append({"messages": messages})
            
    return formatted_data


def getLabelCount(data):    
    NonManipulativeCount = 0
    ManipulatedCount = 0
    for i in data:
        if( i["messages"][2]['content'] == "Yes" ):
            ManipulatedCount = ManipulatedCount+1
        else:
            NonManipulativeCount = NonManipulativeCount +1
    return f"Manipulated Count: {ManipulatedCount}  NonManipulativeCount: {NonManipulativeCount}"

def evaluate_model_on_dataset(model, tokenizer, eval_data):
    y_true = []
    y_pred = []
    
    # Ensure model is in evaluation mode
    model.eval()
    
    print(f"Starting evaluation on {len(eval_data)} samples...")
    
    # tqdm provides a nice progress bar
    for item in tqdm(eval_data, desc="Evaluating"):
        # 1. Extract dialogue and the actual label ("Yes" or "No")
        dialogue = item['messages'][1]['content']
        actual_label_text = item['messages'][2]['content']
        
        # Map actual text label to binary integer (1 for Yes, 0 for No)
        true_label = 1 if "Yes" in actual_label_text else 0
        y_true.append(true_label)
        
        # 2. Construct the prompt
        messages = [
            {"role": "system", "content": "You are an expert psychological analyst. Analyze the following conversation to determine if it contains mental manipulation. Be highly conservative. Do not flag normal arguments, disagreements, or emotional expressions as manipulation. Only answer 'Yes' if there is clear evidence of gaslighting, intimidation, or guilt-tripping. Otherwise, answer 'No'."},
            {"role": "user", "content": dialogue}
        ]
        
        # Apply Qwen's specific chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize input and move to the device (GPU)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        # 3. Generate output
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=5, 
                temperature=0.1, # Low temperature for deterministic outputs
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
            
        # Extract only the newly generated tokens
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        # 4. Map the model's text prediction to binary integer
        # We use .lower() to safely handle cases where the model might output "yes", "Yes.", or "YES"
        pred_label = 1 if "yes" in response.lower() else 0
        y_pred.append(pred_label)

    # 5. Calculate Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # 6. Print Results
    print("\n" + "="*40)
    print("         EVALUATION RESULTS")
    print("="*40)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f} (When it predicts manipulation, how often is it right?)")
    print(f"Recall:    {recall:.4f} (Out of all actual manipulations, how many did it find?)")
    print(f"F1 Score:  {f1:.4f} (Harmonic mean of Precision and Recall)")
    print("="*40)
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Non-Manipulative (0)", "Manipulative (1)"]))

def main():
    # Load the balanced dataset
    Train_list = load_and_format_data("data/BalancedDataset/train.csv")
    Val_list =  load_and_format_data("data/BalancedDataset/val.csv")
    Test_list =  load_and_format_data("data/BalancedDataset/test.csv")

        
    # Replace with the actual path to the folder containing your Base Model files
    base_model_path = "models/Qwen_1.5_BaseModel" 
    #Replace the final model Output path for saving teh model check points and final trained model
    model_checkpoint_path = "models/qwen-manipulation-detector-checkpoints"
    final_model_output_path = "models/qwen-manipulation-detector-model"
    
    # Convert to Hugging Face Dataset objects
    Train_dataset = Dataset.from_list(Train_list)
    Val_dataset = Dataset.from_list(Val_list)
    Test_dataset = Dataset.from_list(Test_list)
    
    
    print("=================================================")
    print("               TRAINING DATASET                  ")
    print(getLabelCount(Train_dataset))
    print("=================================================")
    print("              VALIDATION DATASET                 ")
    print(getLabelCount(Val_dataset))
    print("=================================================")
    print("               TESTING DATASET                   ")
    print(getLabelCount(Test_dataset))
    print("=================================================")
    

    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, local_files_only=True)
    # Qwen doesn't have a default pad token, so we assign one to prevent errors during batching
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # Load the base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        device_map="auto",
        local_files_only=True
    )
    print("Model Loaded Successfully")
    
    # Prepare model for PEFT
    model = prepare_model_for_kbit_training(model)
    
    # LoRA Configuration
    peft_config = LoraConfig(
        r=16, # Rank of the adapter (higher = more capacity, but slower)
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"] # Standard attention blocks to target
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Training Configuration
    training_args = SFTConfig(
        output_dir= model_checkpoint_path,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        max_steps=-1, # Train for full epochs
        num_train_epochs=3, # 3 epochs is usually a good starting point
        eval_strategy="epoch",
        save_strategy="epoch",
        bf16=True, # Use bfloat16 if supported by your GPU (RTX 3000+), otherwise use fp16=True
        max_length=512, # Adjust based on dialogue lengths to save memory
    )
    
    # Initialize Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=Train_dataset,
        eval_dataset=Val_dataset,
        args=training_args,
        processing_class=tokenizer
    )
    
    # Start Fine-tuning
    print("Starting training...")
    train_result = trainer.train()
    print("Training Completed...")
    
    #Log and save the final metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics) # This creates 'train_results.json'
    
    # 2. Save the trainer state (which contains the history of all steps)
    trainer.save_state() # This creates 'trainer_state.json
    
    # Save the final adapter weights
    trainer.model.save_pretrained(final_model_output_path)
    tokenizer.save_pretrained(final_model_output_path)
    print("Training complete and model saved!")

    # Run the evaluation using the 'Test_dataset' 
    # Note: Ensure 'model', 'tokenizer', and 'eval_list' are already loaded in your environment
    evaluate_model_on_dataset(model, tokenizer, Test_dataset)

if __name__ == "__main__":
    main()