import os
import json
import torch
import numpy as np
from tqdm import tqdm
import argparse
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from models.qwen_encoder import QwenTextEncoder
# from models.image_encoder import ImageEncoder # Optional if using visual improvements

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--captions_file", type=str, required=True, help="Path to captions_inclusive.json")
    parser.add_argument("--output_file", type=str, default="offline_index.pt", help="Path to save embeddings")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-1.5B-Instruct")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained model checkpoint")
    args = parser.parse_args()

    # キャプション要素の読み込み (LLM Refinementの出力)
    input_file = args.captions_file
    
    print(f"Loading elements from {input_file}...")
    data = {}
    if input_file.endswith(".jsonl"):
        with open(input_file, 'r') as f:
            for line in f:
                item = json.loads(line)
                # Basenameをキーとして使用（Viewerのロジックと一致させるため）
                # 'path', 'image_path', 'image_paths' のいずれかを取得
                paths = item.get('image_paths') or item.get('paths')
                path = None
                if paths and len(paths) > 0:
                    path = paths[0]
                else:
                    path = item.get('path') or item.get('image_path')
                
                if path:
                    key = os.path.basename(path)
                    if 'elements' in item:
                        data[key] = item['elements']
    else:
        # Standard JSON dict
        with open(input_file, "r") as f:
            data = json.load(f) # {"key": ["element1", "element2", ...]}

    print(f"Loaded {len(data)} items.")
    # モデルの初期化
    print("Initializing CARIMScorer (Text-to-Text)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    from models.carim_scorer import CARIMScorer
    
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading trained checkpoint from {args.checkpoint}...")
        # 学習済みモデルは Projection Layer を使用
        # train.py で embed_dim=256 を使用していると仮定
        # (train.py の設定を確認: model = CARIMScorer(..., embed_dim=256, use_projection=True))
        # したがってここでも embed_dim=256, use_projection=True を指定する必要があります。
        model = CARIMScorer(text_encoder_name=args.model_name, embed_dim=256, use_projection=True).to(device)
        
        # Load State Dict
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state, strict=False)
        print("Checkpoint loaded.")
    else:
        print("No checkpoint provided. Using Zero-Shot (Raw Qwen Embeddings).")
        # Zero-Shot Text-to-Text の場合は生の Qwen 埋め込みを使用 (ランダム射影なし)
        model = CARIMScorer(text_encoder_name=args.model_name, embed_dim=1536, use_projection=False).to(device)
        
    model.eval()
    
    # Tokenizer for elements
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    keys = list(data.keys())
    valid_keys = []
    
    print("Encoding elements...")
    
    MAX_ELEMENTS = 32 # 固定長パディングサイズ
    all_element_embs = []
    all_element_masks = []
    
    batch_size = 16 # process scenes in batch
    
    for i in tqdm(range(0, len(keys), batch_size)):
        batch_keys = keys[i : i + batch_size]
        
        # 要素文字列のバッチを準備
        # エンコードのために平坦化が必要か？
        # シーンごとの要素数が可変なためリストが不揃いになる
        # 戦略: 単純化のため、効率が許せばシーンごとにエンコードを行う（オフライン処理なので許容）
        # しかしバッチ処理の方が良い。
        
        # ロジックの単純化のため、ここではシーンごとの反復処理とする（バッチサイズ分ループ）
        
        for k in batch_keys:
            elements = data[k] # List of strings
            if not isinstance(elements, list):
                # 生の文字列ならリスト化
                elements = [str(elements)]
            
            # 要素リストの切り捨てまたはパディング
            if len(elements) > MAX_ELEMENTS:
                elements = elements[:MAX_ELEMENTS]
            
            # 要素のエンコード
            # トークナイザへの入力: 文字列リスト
            if len(elements) == 0:
                elements = ["empty"]
                
            inputs = tokenizer(elements, padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
            
            with torch.no_grad():
                # [num_elements, L, D] (Qwen はシーケンスを出力)
                token_embs = model.encode_text(inputs.input_ids, inputs.attention_mask)
                
                # 平均プーリング (各要素ごとの文埋め込み)
                # mask: [N, L]
                mask_expanded = inputs.attention_mask.unsqueeze(-1).float() # [N, L, 1]
                sum_emb = (token_embs * mask_expanded).sum(dim=1) # [N, D]
                sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9) # [N, 1]
                embs = sum_emb / sum_mask # [N, D]
            
            # Pad to MAX_ELEMENTS
            num_e = embs.shape[0]
            padded_emb = torch.zeros(MAX_ELEMENTS, embs.shape[1], device=device) # [32, D]
            mask = torch.zeros(MAX_ELEMENTS, device=device) # [32]
            
            padded_emb[:num_e] = embs
            mask[:num_e] = 1.0
            
            all_element_embs.append(padded_emb.cpu())
            all_element_masks.append(mask.cpu())
            valid_keys.append(k)

    if not all_element_embs:
        print("No embeddings generated. Exiting.")
        return

    all_element_embs = torch.stack(all_element_embs, dim=0) # [N, 32, D]
    all_element_masks = torch.stack(all_element_masks, dim=0) # [N, 32]

    # Save
    print(f"Saving index to {args.output_file}...")
    torch.save({
        "keys": valid_keys,
        "embeddings": all_element_embs,
        "masks": all_element_masks
    }, args.output_file)
    print("Done.")

if __name__ == "__main__":
    main()
