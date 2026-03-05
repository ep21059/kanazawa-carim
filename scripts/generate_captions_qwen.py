import os
import json
import torch
import argparse
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Generate dense captions using Qwen-VL-Chat")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to directory containing images/frames")
    parser.add_argument("--output_file", type=str, default="captions_inclusive.json", help="Output JSON file")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen-VL-Chat", help="HuggingFace model path")
    parser.add_argument("--prompt", type=str, default="Describe this autonomous driving scene in detail. Mention traffic conditions, road users (cars, pedestrians), weather, time of day, and any potential hazards.", help="Prompt for captioning")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (currently 1 is recommended for generation)")
    args = parser.parse_args()

    # Model Initialization
    print(f"Loading model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    # Remove quantization to avoid dtype mismatch (A6000 has enough VRAM)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="cuda",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).eval()
    
    # Collect Images (Recursively)
    image_paths = []
    for root, dirs, files in os.walk(args.data_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(root, file))
    
    image_paths.sort()
    print(f"Found {len(image_paths)} images.")

    results = {}
    # Check for existing results to resume
    if os.path.exists(args.output_file):
        print(f"Loading existing results from {args.output_file}...")
        try:
            with open(args.output_file, "r") as f:
                results = json.load(f)
            print(f"Resuming with {len(results)} already processed images.")
        except Exception as e:
            print(f"Failed to load existing file: {e}. Starting fresh.")
    
    # Filter out already processed images
    # Need to verify if keys match rel_path logic
    # Pre-calculate map output to speed up checking?
    # Key logic: rel_path = os.path.relpath(img_path, start=os.path.dirname(args.data_dir))
    
    # Let's iterate and skip
    
    # Generation Loop
    save_interval = 50
    count = 0
    
    for img_path in tqdm(image_paths):
        rel_path = os.path.relpath(img_path, start=os.path.dirname(args.data_dir))
        
        if rel_path in results:
            continue
            
        # Prepare Input
        query = tokenizer.from_list_format([
            {'image': img_path},
            {'text': args.prompt},
        ])
        
        # Generator
        try:
            response, history = model.chat(tokenizer, query=query, history=None)
            
            # Store Result
            results[rel_path] = response
            count += 1
            
            # Periodic Save
            if count % save_interval == 0:
                with open(args.output_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
            
    # Final Save
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Saved captions to {args.output_file}")

if __name__ == "__main__":
    main()
