
import torch
import random
import numpy as np
from tqdm import tqdm

class AdaptiveNegativeInjector:
    def __init__(self, elements_path, model, tokenizer, device="cuda"):
        """
        Manages Negative Pool and Injection Logic.
        """
        self.device = device
        self.tokenizer = tokenizer
        self.model = model  # Shared model instance to compute embeddings
        
        # 1. Build Global Negative Pool
        import json
        with open(elements_path, 'r') as f:
            data = json.load(f)
        
        all_elements = set()
        for k, v in data.items():
            if isinstance(v, list):
                for e in v:
                    all_elements.add(e.strip().lower())
        
        self.negative_pool = list(all_elements)
        print(f"[ANI] Global Negative Pool initialized with {len(self.negative_pool)} unique elements.")
        
        # Pre-compute embeddings for semantic filtering?
        # Too heavy for 1.5B model on thousands of elements in one go.
        # We will compute on-the-fly or cache in chunks if needed.
        # For now, let's just cache them if pool is small (<10k).
        
        self.pool_embeddings = None
        if len(self.negative_pool) < 20000:
            self.compute_pool_embeddings()
            
    def compute_pool_embeddings(self):
        print("[ANI] Pre-computing embeddings for negative pool...")
        bs = 256
        embs = []
        with torch.no_grad():
            for i in tqdm(range(0, len(self.negative_pool), bs)):
                batch = self.negative_pool[i:i+bs]
                inputs = self.tokenizer(batch, padding=True, truncation=True, max_length=16, return_tensors="pt").to(self.device)
                
                # We need simple pooled embeddings for semantic comparison
                out = self.model.encode_text(inputs.input_ids, inputs.attention_mask)
                # Mean Pool
                mask_exp = inputs.attention_mask.unsqueeze(-1).float()
                sum_emb = (out * mask_exp).sum(dim=1)
                sum_mask = mask_exp.sum(dim=1).clamp(min=1e-9)
                pooled = sum_emb / sum_mask
                embs.append(pooled.cpu())
        
        self.pool_embeddings = torch.cat(embs, dim=0).to(self.device) # [N_pool, D]
        self.pool_embeddings = torch.nn.functional.normalize(self.pool_embeddings, dim=-1)
        print("[ANI] Embeddings cached.")

    def select_negatives(self, positive_elements, n_neg=1):
        """
        Selects N negatives that are semantically distinct from positive_elements.
        """
        if not positive_elements:
            return random.sample(self.negative_pool, n_neg)
            
        # Encode Positives
        with torch.no_grad():
            inputs = self.tokenizer(positive_elements, padding=True, truncation=True, max_length=16, return_tensors="pt").to(self.device)
            out = self.model.encode_text(inputs.input_ids, inputs.attention_mask)
            mask_exp = inputs.attention_mask.unsqueeze(-1).float()
            sum_emb = (out * mask_exp).sum(dim=1)
            sum_mask = mask_exp.sum(dim=1).clamp(min=1e-9)
            pos_emb = sum_emb / sum_mask # [M_pos, D]
            pos_emb = torch.nn.functional.normalize(pos_emb, dim=-1)
            
        # Calc Sim with Pool
        # pos_emb: [M, D] vs pool: [N, D]
        # Max Sim per pool item
        sims = torch.mm(self.pool_embeddings, pos_emb.t()) # [N, M]
        max_sims = sims.max(dim=1)[0] # [N]
        
        # Filter: Sim < 0.65 (Threshold)
        valid_indices = torch.where(max_sims < 0.65)[0]
        
        if len(valid_indices) < n_neg:
            # Fallback: take lowest sims
            valid_indices = torch.argsort(max_sims)[:n_neg]
            
        selected_idx = valid_indices[torch.randperm(len(valid_indices))[:n_neg]]
        
        return [self.negative_pool[i] for i in selected_idx.cpu().numpy()]

    def generate_synthetic_queries(self, batch_positive_elements):
        """
        Generates Hard and Easy Negative queries for a batch.
        """
        syn_queries = []
        
        for elems in batch_positive_elements:
            # 20% Pure Negative (Balanced Sampling)
            # 40% Hard Negative (Pos + 1 Neg)
            # 40% Easy Negative (Subset Pos + Negs)
            
            rand_val = random.random()
            
            if rand_val < 0.2:
                # Pure Negative (BS)
                # Select 2 negatives
                negs = self.select_negatives(elems, n_neg=2)
                q_list = negs
            elif rand_val < 0.6:
                # Hard: All Pos + 1 Neg
                neg = self.select_negatives(elems, n_neg=1)[0]
                q_list = elems + [neg]
            else:
                # Easy: Subset Pos + Neg
                neg = self.select_negatives(elems, n_neg=1)[0]
                if len(elems) > 0:
                    k = random.randint(1, len(elems))
                    q_list = random.sample(elems, k) + [neg]
                else:
                    q_list = [neg]
            
            random.shuffle(q_list)
            syn_queries.append(", ".join(q_list))
            
        return syn_queries

    def generate_positive_queries(self, batch_positive_elements):
        """
        Generates Synthetic Positive queries (Subsets of elements).
        Covers L_pos explicitly.
        """
        pos_queries = []
        for elems in batch_positive_elements:
            if not elems:
                pos_queries.append("empty")
                continue
                
            # Random subset (non-empty)
            k = random.randint(1, len(elems))
            q_list = random.sample(elems, k)
            random.shuffle(q_list)
            pos_queries.append(", ".join(q_list))
        return pos_queries
