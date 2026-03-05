
import json
import os
import argparse
from tqdm import tqdm

def merge_dataset(train_jsonl, captions_json, elements_json, output_jsonl):
    print(f"Loading base dataset from {train_jsonl}...")
    base_data = []
    with open(train_jsonl, 'r') as f:
        for line in f:
            base_data.append(json.loads(line))

    print(f"Loading captions from {captions_json}...")
    with open(captions_json, 'r') as f:
        captions_map = json.load(f) # Key: filename, Value: caption string

    print(f"Loading elements from {elements_json}...")
    with open(elements_json, 'r') as f:
        elements_map = json.load(f) # Key: filename, Value: list of strings
        
    print(f"Merging {len(base_data)} items...")
    merged_count = 0
    with open(output_jsonl, 'w') as f:
        for item in tqdm(base_data):
            # item['image_paths'] is a list
            paths = item.get('image_paths') or item.get('paths')
            if paths and len(paths) > 0:
                path = paths[0]
            else:
                path = item.get('path') or item.get('image_path')
            
            if not path:
                print(f"Warning: No path found for item: {item}")
                continue

            # Key matching logic (same as before)
            key_candidates = []
            if 'id' in item:
                key_candidates.append(item['id'])
            
            basename = os.path.basename(path)
            key_candidates.append(basename)
            
            if 'CAM' in path:
                parts = path.split('/')
                if len(parts) >= 2:
                    key_candidates.append(f"{parts[-2]}/{parts[-1]}")

            # 1. Merge Caption (Text)
            found_caption = ""
            for k in key_candidates:
                if k in captions_map:
                    found_caption = captions_map[k]
                    break
            
            # If base has empty text, fill it
            if found_caption:
                item['text'] = found_caption

            # 2. Merge Elements
            matched_elements = []
            found_elem = False
            for k in key_candidates:
                if k in elements_map:
                    matched_elements = elements_map[k]
                    found_elem = True
                    break
            
            if found_elem:
                item['elements'] = matched_elements
            else:
                item['elements'] = []
            
            # Count if we have both?
            if found_caption or found_elem:
                merged_count += 1
                
            f.write(json.dumps(item) + "\n")
            
    print(f"Done. Modified {merged_count} / {len(base_data)} items. Saved to {output_jsonl}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonl", type=str, required=True, help="Path to base train.jsonl (paths)")
    parser.add_argument("--captions_json", type=str, required=True, help="Path to VLM captions JSON")
    parser.add_argument("--elements_json", type=str, required=True, help="Path to refined elements JSON")
    parser.add_argument("--output_jsonl", type=str, required=True, help="Output path")
    args = parser.parse_args()
    
    merge_dataset(args.train_jsonl, args.captions_json, args.elements_json, args.output_jsonl)
