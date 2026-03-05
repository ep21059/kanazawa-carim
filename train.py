# train.py
import torch
import os
import sys
sys.path.append("/workspace")

from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torchvision import transforms
from tqdm import tqdm

# from models.image_encoder import ImageEncoder
# from models.qwen_encoder import QwenTextEncoder
# from models.carim_model import CARIMModel
# from datasets.scene_dataset import SceneTextMatchDataset
from src_datasets.nuscenes_vlm.dataset import NuScenesVLMDataset

from losses.itc_loss import recall_at_k

def scene_collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    # images = torch.stack(images) # No images needed for Text-to-Text check
    
    texts = []
    elements = []
    for b in batch:
        texts.append(b["text"])
        elements.append(b["elements"])

    return {
        "text": texts, # トークナイザラッパーで単数形のキーが使われることが多いですが、ループ内では batch["texts"] を使用
        "texts": texts, # 安全のため保持（メインループで使用）
        "elements": elements
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_path", type=str, default="/home/ryoc1220/carim_ver1/datasets/nuscenes_vlm/processed/train.jsonl")
    parser.add_argument("--elements_path", type=str, default="/home/ryoc1220/carim_ver1/datasets/nuscenes_vlm/captions_elements.json")
    parser.add_argument("--epochs", type=int, default=3, help="エポック数")
    parser.add_argument("--batch_size", type=int, default=4, help="バッチサイズ (対照学習のため大きめを推奨)") 
    parser.add_argument("--save_path", type=str, default="runs/carim_text_model.pt", help="モデル保存パス")
    parser.add_argument("--lr", type=float, default=1e-4, help="学習率")
    parser.add_argument("--num_workers", type=int, default=4, help="データローダーのワーカー数")
    parser.add_argument("--pretrained", type=str, default=None, help="追加学習用の事前学習済み重みパス")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Qwenのトークナイザは通常エンコーダ内で処理されますが、ここでも collate/tokenization のために必要です
    # エンコーダインスタンスからロードするか、事前学習済みパスからロードします

    qwen_model_name = "Qwen/Qwen2-1.5B-Instruct"
    try:
        tokenizer = AutoTokenizer.from_pretrained(qwen_model_name, trust_remote_code=True)
    except:
        print(f"Failed to load tokenizer from {qwen_model_name}, assuming download in progress or using fallback.")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") # Fallback (will fail later if mismatched)

    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    print(f"Loading dataset from {args.jsonl_path}")
    dataset = NuScenesVLMDataset(
        jsonl_path=args.jsonl_path,
        elements_path=args.elements_path,
        image_transform=image_transform,
        max_frames=1, 
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=scene_collate_fn,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # グローバルモデルではなく CARIMScorer を使用
    from models.carim_scorer import CARIMScorer
    # 注意: Text-to-Text 学習には Projection Layer の有効化が必要
    model = CARIMScorer(text_encoder_name=qwen_model_name, embed_dim=256, use_projection=True).to(device)
    
    # DataParallel disabled for stability
    # if torch.cuda.device_count() > 1: ...
        
    # Single GPU mode
    raw_model = model

    if args.pretrained and os.path.exists(args.pretrained):
        print(f"Loading pretrained weights from {args.pretrained} for fine-tuning...")
        state_dict = torch.load(args.pretrained, map_location=device)
        model.load_state_dict(state_dict, strict=False)

    # logit_scale = torch.nn.Parameter(torch.tensor(4.6052)).to(device) # initial temperature ~ 100
    # To be safe with device placement:
    logit_scale = torch.nn.Parameter(torch.tensor(4.6052, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer.add_param_group({"params": [logit_scale]})

    # ANI (Adaptive Negative Injection) の初期化
    from scripts.ani_utils import AdaptiveNegativeInjector
    print("Initializing ANI Injector...")
    ani_injector = AdaptiveNegativeInjector(
        elements_path=args.elements_path,
        # raw_model を渡す (フィルタリング等で encode_text を使う可能性があるため)
        # 理想的には ANI も効率化すべきだが、バッチ前処理なので許容
        model=raw_model, 
        tokenizer=tokenizer,
        device=device
    )

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        steps = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for step, batch in enumerate(pbar):
            if batch is None:
                continue
            
            optimizer.zero_grad()
            
            # --- 1. ポジティブペアの処理 ---
            queries = batch["texts"]
            elements_batch = batch["elements"] # 文字列リストのリスト
            
            # Checks
            if not queries or not elements_batch:
                continue

            # クエリ（元の詳細キャプション）をエンコード
            q_inputs = tokenizer(queries, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            
            # DEBUG: Check IDs
            if q_inputs.input_ids.max() >= 151936 or q_inputs.input_ids.min() < 0:
                print(f"CRITICAL: Query ID Out of Bounds! Max={q_inputs.input_ids.max()}, Min={q_inputs.input_ids.min()}")
                continue # Skip this batch
                
            # エンコード (DataParallelなし、直接呼び出しでもOKだが、forward=encode_text なのでそのまま)
            query_emb = model(q_inputs.input_ids.long(), q_inputs.attention_mask)
            
            # 2. 要素(Keys)のエンコード
            # マルチGPUではなくても、バッチ化してエンコードするのは効率が良いのでロジックは維持
            B = len(queries)
            max_elements = 32
            
            # 大規模バッチエンコーディングのために要素を平坦化
            flat_elements = []
            flat_indices = [] # サンプル所属IDを追跡
            
            for b_idx in range(B):
                elems = elements_batch[b_idx]
                if not elems: 
                    elems = ["empty"]
                if len(elems) > max_elements:
                    elems = elems[:max_elements]
                
                flat_elements.extend(elems)
                # Pad to max_elements conceptually (we will reconstruct tensor later)
                # To reconstruct [B, max_elements, D], we need to map flat outputs.
                # Actually, simpler: just prepare [B, max_elements] strings and tokenize?
                # No, padding strings is messy.
                # Let's flatten, encode, then scatter back to [B, max_elements, D]
                
                # Store (b_idx, elem_idx) for each flattened item?
                # Or just keep list of lengths.
            
            # 全要素を一度にトークナイズ
            e_inputs = tokenizer(flat_elements, padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
            
            # DEBUG
            if e_inputs.input_ids.max() >= 151936 or e_inputs.input_ids.min() < 0:
                print(f"CRITICAL: Element ID Out of Bounds! Max={e_inputs.input_ids.max()}, Min={e_inputs.input_ids.min()}")
                # Identify culprit?
                # for ii, row in enumerate(e_inputs.input_ids):
                #    if row.max() >= 151936: print(f"Bad Element: {flat_elements[ii]}")
                continue
            
            # 大規模並列エンコーディング
            e_outputs_flat = model(e_inputs.input_ids.long(), e_inputs.attention_mask)
            
            # [B, max_elements, D] の形状に再構築
            # e_outputs_flat は [Total_Elems, D] (encode_text は [B, Seq, D] を返すが、ここではプーリングが必要？)
            # CARIMScorer.encode_text は [B, L, D] を返すため、ここで Mean Pooling を行う必要がある。
            
            # フラットバッチ上でのプーリング
            mask_expanded = e_inputs.attention_mask.unsqueeze(-1).float()
            sum_emb = (e_outputs_flat * mask_expanded).sum(dim=1)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
            pooled_flat = sum_emb / sum_mask # [Total_Elems, D]
            
            # Scatter to [B, max_elements, D]
            element_embs = torch.zeros(B, max_elements, pooled_flat.shape[1], device=device)
            element_masks = torch.zeros(B, max_elements, device=device)
            
            current_idx = 0
            for b_idx in range(B):
                elems = elements_batch[b_idx]
                # Same truncation logic as above to match counts
                if not elems: elems = ["empty"]
                num = min(len(elems), max_elements)
                
                element_embs[b_idx, :num, :] = pooled_flat[current_idx : current_idx+num]
                element_masks[b_idx, :num] = 1.0
                current_idx += num
                
            
            # 3. 類似度行列の計算 (B x B)
            # raw_model.compute_similarity (軽量な内積計算) を使用
            scores = torch.zeros(B, B, device=device)
            
            for j in range(B): # For each candidate scene j
                target_e_emb = element_embs[j].unsqueeze(0).expand(B, -1, -1)
                target_e_mask = element_masks[j].unsqueeze(0).expand(B, -1)
                col_scores = raw_model.compute_similarity(query_emb, target_e_emb, q_inputs.attention_mask, target_e_mask)
                scores[:, j] = col_scores

            # 4. 標準的な対照学習損失 (Contrastive Loss)
            logits = scores * logit_scale.exp()
            labels = torch.arange(B, device=device)
            loss_i2t = torch.nn.functional.cross_entropy(logits, labels)
            loss_t2i = torch.nn.functional.cross_entropy(logits.t(), labels)
            loss_contrastive = (loss_i2t + loss_t2i) / 2
            
            # --- ANI (Adaptive Negative Injection) ロジック ---
            syn_queries = ani_injector.generate_synthetic_queries(elements_batch)
            syn_inputs = tokenizer(syn_queries, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            # 並列エンコード
            if syn_inputs.input_ids.max() >= 151936 or syn_inputs.input_ids.min() < 0:
                print(f"CRITICAL: ANI SynQuery ID Out of Bounds! Max={syn_inputs.input_ids.max()}")
                continue
            
            syn_q_emb = model(syn_inputs.input_ids.long(), syn_inputs.attention_mask)
            
            # L_neg: ネガティブ混入クエリのスコアを最小化
            ani_scores = raw_model.compute_similarity(syn_q_emb, element_embs, syn_inputs.attention_mask, element_masks)
            loss_ani = torch.mean(ani_scores ** 2)
            
            # L_pos: ポジティブサブセットクエリのスコアを最大化
            pos_queries = ani_injector.generate_positive_queries(elements_batch)
            pos_inputs = tokenizer(pos_queries, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            if pos_inputs.input_ids.max() >= 151936 or pos_inputs.input_ids.min() < 0:
                 print(f"CRITICAL: ANI PosQuery ID Out of Bounds! Max={pos_inputs.input_ids.max()}")
                 continue
            # 並列エンコード
            pos_q_emb = model(pos_inputs.input_ids.long(), pos_inputs.attention_mask)
            
            pos_scores = raw_model.compute_similarity(pos_q_emb, element_embs, pos_inputs.attention_mask, element_masks)
            
            loss_syn_pos = torch.mean((1.0 - pos_scores) ** 2)
            
            loss = loss_contrastive + loss_ani + loss_syn_pos
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            steps += 1
            
            if step % 10 == 0:
                print(f"Epoch {epoch} Step {step} | Loss {loss.item():.4f} (Cont: {loss_contrastive.item():.4f}, Neg: {loss_ani.item():.4f}, Pos: {loss_syn_pos.item():.4f})")


        if steps > 0:
             print(f"Epoch {epoch} | Avg Loss={total_loss/steps:.4f} | Total skipped={dataset.skipped_count}")
        else:
             print(f"Epoch {epoch} | No valid batch processed | Total skipped={dataset.skipped_count}")

    # DataParallel使用時は、元モデルのステート辞書を保存（キーの 'module.' 接頭辞回避）
    # 保存
    torch.save(model.state_dict(), args.save_path)
    print(f"Model saved to {args.save_path}")

if __name__ == "__main__":
    main()
