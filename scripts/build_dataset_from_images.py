import os
import json
import argparse
import glob

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default="/home/ryoc1220/carim_ver1/kanazawa_ver/datasets/kanazawa_scene/samples/CAM_FRONT")
    parser.add_argument("--output_path", type=str, default="/home/ryoc1220/carim_ver1/kanazawa_ver/datasets/kanazawa_scene/processed/train.jsonl")
    args = parser.parse_args()
    
    image_dir = args.image_dir
    output_path = args.output_path
    
    # Ensure dir exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    samples = []
    
    # 1. Scan Images
    if not os.path.exists(image_dir):
        print(f"Error: {image_dir} not found.")
        return

    # Kanazawa dataset format (JPG files)
    files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])
    print(f"Found {len(files)} images in {image_dir}.")
    
    # 2. Build JSONL items
    abs_base = os.path.abspath(image_dir)
    
    for f in files:
        key = f"CAM_FRONT/{f}"
        
        item = {
            "id": key,
            "scene_name": "kanazawa", 
            "image_paths": [os.path.join(abs_base, f)],
            "text": "", # Placeholder, dense caption will come from VLM output later
            "meta": {}
        }
        samples.append(item)
        
    # 3. Save
    with open(output_path, 'w') as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
            
    print(f"Saved {len(samples)} items to {output_path}")

if __name__ == "__main__":
    main()
