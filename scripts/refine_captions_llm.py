import os
import json
import torch
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description="Refine dense captions into elements using LLM")
    parser.add_argument("--json_path", type=str, required=True, help="Path to captions_inclusive.json")
    parser.add_argument("--output_file", type=str, default="datasets/nuscenes_vlm/captions_elements.json", help="Output JSON file")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2-1.5B-Instruct", help="HuggingFace model path")
    
    args = parser.parse_args()

    # Load Source Captions
    print(f"Loading source captions from {args.json_path}...")
    with open(args.json_path, "r") as f:
        captions = json.load(f)

    # Initialize LLM
    print(f"Loading LLM: {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="cuda",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).eval()

    # System Prompt for Extraction (Improved to exclude negatives)
    system_prompt = (
        "You are an AI assistant for autonomous driving scene understanding. "
        "Your task is to extract a list of key elements (objects, actions, environment) that are ACTIVELY PRESENT in the scene. "
        "Strictly EXCLUDE objects that are mentioned as 'not present', 'no', 'missing', or 'free of'. "
        "For example, if the description says 'no pedestrians', DO NOT include 'pedestrians' or 'no pedestrians' in your list. "
        "Output ONLY a comma-separated list of short phrases. Do not write full sentences."
    )
    
    # Process Loop
    results = {}
    
    # Check for resume
    if os.path.exists(args.output_file):
        try:
            with open(args.output_file, "r") as f:
                results = json.load(f)
            print(f"Resuming with {len(results)} processed items.")
        except:
            pass

    keys = sorted(list(captions.keys()))
    
    # Batch processing isn't strictly necessary for LLM generation usually, doing one by one or small batches
    count = 0
    save_interval = 50

    for k in tqdm(keys):
        if k in results:
            continue
            
        description = captions[k]
        
        # Construct Prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Description: {description}\n\nList the elements:"}
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        try:
            with torch.no_grad():
                generated_ids = model.generate(
                    model_inputs.input_ids,
                    max_new_tokens=128,
                    do_sample=False, # Use deterministic decoding for stability
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Post-process: Split by comma and clean
            # Expected format: "Pedestrian crossing street, Red truck, Cloudy sky"
            # We store the raw string OR the list. Let's store the list of strings.
            elements = [x.strip() for x in response.split(",")]
            
            results[k] = elements
            count += 1
            
            if count % save_interval == 0:
                with open(args.output_file, "w") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"Error processing {k}: {e}")
            continue

    # Final Save
    with open(args.output_file, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Done. Saved elements to {args.output_file}")

if __name__ == "__main__":
    main()
