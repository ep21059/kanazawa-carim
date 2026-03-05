import streamlit as st
import torch
import torch.nn.functional as F
import os
import sys
import json
import glob
import time
from transformers import AutoTokenizer

# Ensure models module is visible
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.carim_scorer import CARIMScorer

# --- ヘルパー関数 ---
def get_scene_id(filename):
    """
    Kanazawaファイル名からScene IDを抽出。
    全フレームが同じプレフィックスを持つため、タイムスタンプを利用して
    およそ300フレーム（約30秒）ごとの仮想的なシーンIDを生成します。
    ex: 20250127_151151_367912100_1.jpg -> 20250127_151151_scene001
    """
    basename = os.path.basename(filename)
    parts = basename.split("_")
    if len(parts) >= 3:
        prefix = f"{parts[0]}_{parts[1]}"
        try:
            ts = int(parts[2])
            # Assuming TS increments by ~100 per frame. Grouping by roughly 30000 units (300 frames)
            chunk_id = ts // 30000
            return f"{prefix}_scene{chunk_id:04d}"
        except:
            return prefix
    elif len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return "unknown_scene"

def get_timestamp(filename):
    """
    Kanazawaファイル名からタイムスタンプを抽出。
    """
    basename = os.path.basename(filename)
    parts = basename.split("_")
    if len(parts) >= 3:
        try:
            return int(parts[2])
        except:
            return 0
    return 0

def get_metadata(filename, caption):
    """
    フィルタリング用のメタデータを抽出。
    """
    scene_id = get_scene_id(filename)
    location = "Kanazawa"
    
    time_of_day = "Daytime"
    try:
        parts = scene_id.split("_")
        if len(parts) >= 2:
            hour = int(parts[1][:2])
            if 5 <= hour < 12: time_of_day = "Morning"
            elif 12 <= hour < 17: time_of_day = "Daytime"
            elif 17 <= hour < 20: time_of_day = "Evening"
            else: time_of_day = "Night"
    except:
        pass
        
    # 3. 天候 (キャプションから推定)
    # "no signs of rain" などの否定表現を考慮したロジック
    weather = "Unknown"
    lower_cap = caption.lower()
    
    def has_keyword(text, keywords):
        for k in keywords:
            if k in text:
                # Check for negation
                # Find index
                idx = text.find(k)
                # Check preceding 15 chars for "no ", "not "
                context = text[max(0, idx-15):idx]
                if "no " in context or "not " in context or "free of " in context:
                    continue
                return True
        return False

    if has_keyword(lower_cap, ["rain", "drizzle", "wet ground"]):
        weather = "Rainy"
    elif has_keyword(lower_cap, ["snow", "icy"]):
        weather = "Snowy"
    elif has_keyword(lower_cap, ["sunny", "clear sky", "bright"]):
        weather = "Sunny"
    elif has_keyword(lower_cap, ["cloud", "overcast", "gray sky"]):
        weather = "Cloudy"
    elif "night" in lower_cap and "day" not in lower_cap: 
        weather = "Night"
    
    return {
        "Location": location,
        "Time": time_of_day,
        "Weather": weather
    }


# 動画プレイヤー用のヘルパー関数
def render_video_player(sid, scene_map, unique_id, highlight_path=None):
    """
    指定されたScene IDの動画プレイヤーを描画します。
    sid: Scene ID
    scene_map: 全シーンのフレーム情報マップ
    unique_id: UIキー用の識別子 (例: 'browse_0', 'rank_1')
    highlight_path: (任意) ハイライト/開始位置とする特定フレームのパス
    """
    scene_frames = scene_map.get(sid, [])
    
    if not scene_frames:
        st.warning("No frames found for this scene.")
        return

    # Determine start index
    start_idx = 0
    if highlight_path:
        for j, (ts, p) in enumerate(scene_frames):
            if p == highlight_path:
                start_idx = j
                break
    else:
        # Default to middle frame if no highlight
        start_idx = len(scene_frames) // 2

    # Video Player Controls
    # Adjust columns to tighten the play button position
    col_play, col_slider = st.columns([0.08, 0.92])
    
    # Session state key for playback
    vid_key = f"play_{unique_id}_{sid}"
    if vid_key not in st.session_state:
        st.session_state[vid_key] = False
        
    with col_play:
        # Spacer to align with slider (approx)
        st.write("") 
        btn_label = "⏹" if st.session_state[vid_key] else "▶"
        if st.button(btn_label, key=f"btn_{unique_id}_{sid}"):
            st.session_state[vid_key] = not st.session_state[vid_key]
            st.rerun()

    # Context Window Logic (show +/- 10 seconds around the highlight/start)
    if highlight_path:
        # Windowed
        hit_ts = scene_frames[start_idx][0]
        context_window = 10_000_000 # 10s
        w_start, w_end = hit_ts - context_window, hit_ts + context_window
        subset_frames = [f for f in scene_frames if w_start <= f[0] <= w_end]
        
        # Re-calc relative index
        rel_idx = 0
        for j, (ts, p) in enumerate(subset_frames):
            if p == highlight_path:
                rel_idx = j
                break
    else:
        # Full Scene
        subset_frames = scene_frames
        rel_idx = start_idx

    with col_slider:
        caption_text = f"Timeline: {len(subset_frames)} frames"
        if highlight_path:
             caption_text += f" (Frame {rel_idx+1} Match)"
        st.caption(caption_text)
        
        def fmt_frame(idx):
            if highlight_path and idx == rel_idx:
                return f"Frame {idx+1} (⭐)"
            return f"Frame {idx+1}"
            
        frame_pos = st.select_slider(
            f"Seek_{unique_id}", 
            options=range(len(subset_frames)),
            value=rel_idx if rel_idx < len(subset_frames) else 0,
            format_func=fmt_frame,
            key=f"slider_{unique_id}_{sid}",
            label_visibility="collapsed" 
        )

    img_container = st.empty()
    
    if st.session_state[vid_key]:
        # Playback
        for f_idx in range(len(subset_frames)):
            f_path = subset_frames[f_idx][1]
            msg = f"Frame {f_idx+1}/{len(subset_frames)} (Playing)"
            if highlight_path and subset_frames[f_idx][1] == highlight_path:
                 msg += " ⭐ Match"
            img_container.image(f_path, caption=msg, use_container_width=True)
            time.sleep(0.1) # 10 FPS for smoother playback
        
        st.session_state[vid_key] = False
        st.rerun()
    else:
        # Static
        current_path = subset_frames[frame_pos][1]
        msg = f"Frame {frame_pos+1}/{len(subset_frames)} (Paused)"
        if highlight_path and current_path == highlight_path:
            msg += " (⭐ Match)"
        img_container.image(current_path, caption=msg, use_container_width=True)
        return current_path

    return None

# --- Cache & Resources ---
@st.cache_resource
def load_resources(index_path, model_name, checkpoint_path):
    """
    モデル、インデックス、メタデータをロードします。
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}...")
    
    # Initialize Text-to-Text Model
    from models.carim_scorer import CARIMScorer
    
    if True: # 強制チェックロジック
        print(f"DEBUG: Attempting load from {checkpoint_path}", flush=True)
        if os.path.exists(checkpoint_path) and checkpoint_path != "":
            print("DEBUG: Path exists. Initializing with Projection=True", flush=True)
            model = CARIMScorer(text_encoder_name=model_name, embed_dim=256, use_projection=True)
            try:
                state = torch.load(checkpoint_path, map_location="cpu")
                model.load_state_dict(state, strict=False)
                print("DEBUG: Checkpoint loaded successfully.", flush=True)
            except Exception as e:
                print(f"DEBUG: Warning: Load failed {e}", flush=True)
        else:
            print(f"DEBUG: Path {checkpoint_path} NOT found. Fallback to ZeroShot.", flush=True)
            model = CARIMScorer(text_encoder_name=model_name, embed_dim=1536, use_projection=False)
            
    # Verify Model
    print(f"DEBUG: Model use_projection={model.use_projection}", flush=True)
    if hasattr(model, "text_proj"):
         print(f"DEBUG: Model has text_proj: {model.text_proj}", flush=True)
    else:
         print("DEBUG: Model has NO text_proj", flush=True)
         
    model = model.to(device).eval()

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


    # インデックスのロード
    keys = []
    embeddings = None
    masks = None
    resolved_paths = []
    
    if os.path.exists(index_path):
        print(f"Loading index from {index_path}")
        data = torch.load(index_path, map_location="cpu")
        keys = data["keys"]
        embeddings = data["embeddings"].to(device) # [N, M, D]
        masks = data["masks"].to(device) # [N, M]
        
        # パスの解決
        base_dirs = [
             "/workspace/datasets/kanazawa/20250127_151151",
             "datasets/kanazawa_scene",
             "."
        ]
        
        for k in keys:
            found_path = None
            if os.path.exists(k):
                found_path = os.path.abspath(k)
            else:
                for b in base_dirs:
                    p = os.path.join(b, k)
                    if os.path.exists(p):
                        found_path = os.path.abspath(p)
                        break
                    # Try basename
                    p_base = os.path.join(b, os.path.basename(k))
                    if os.path.exists(p_base):
                        found_path = os.path.abspath(p_base)
                        break
            resolved_paths.append(found_path)
                    
    else:
        print(f"Index not found: {index_path}")
    
    # 動画再生用のシーンマップ
    scene_map = {}
    
    # Scan directory for full scene playback
    print("Scanning directory for full scene playback...")
    search_dir = "/workspace/datasets/kanazawa/20250127_151151"
    if os.path.exists(search_dir):
        all_files = glob.glob(os.path.join(search_dir, "*.jpg"))
        for f in all_files:
            abs_f = os.path.abspath(f)
            sid = get_scene_id(abs_f)
            ts = get_timestamp(abs_f)
            if sid not in scene_map: scene_map[sid] = []
            scene_map[sid].append( (ts, abs_f) )
            
    # Add resolved paths
    for p in resolved_paths:
        if p and os.path.exists(p):
            sid = get_scene_id(p)
            ts = get_timestamp(p)
            if sid not in scene_map: scene_map[sid] = []
            scene_map[sid].append( (ts, p) )

    # Sort frames
    for sid in scene_map:
        unique_frames = {}
        for ts, p in scene_map[sid]:
            unique_frames[p] = ts
        scene_map[sid] = sorted([(ts, p) for p, ts in unique_frames.items()], key=lambda x: x[0])
        
    return model, keys, embeddings, masks, scene_map, resolved_paths, tokenizer

def main():
    st.set_page_config(layout="wide", page_title="CARIM: Context-Aware Retrieval with Video Playback")
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_path", type=str, default="datasets/kanazawa_scene/processed/text_index.pt")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-1.5B-Instruct")
    parser.add_argument("--checkpoint_path", type=str, default="runs/carim_kanazawa_finetuned.pt")
    parser.add_argument("--jsonl_path", type=str, default="datasets/kanazawa_scene/processed/train_full.jsonl")
    
    try:
        args = parser.parse_args()
    except SystemExit:
        args = argparse.Namespace(index_path="datasets/kanazawa_scene/processed/text_index.pt", model_name="Qwen/Qwen2-1.5B-Instruct", checkpoint_path="runs/carim_kanazawa_finetuned.pt")

    print(f"DEBUG: Args parsed: {args}", flush=True)
    
    # Resolve Paths Absolute
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    def resolve(p):
        if os.path.isabs(p): return p
        return os.path.join(project_root, p)
        
    ckpt_path = resolve(args.checkpoint_path)
    idx_path = resolve(args.index_path)
    jsonl_path = resolve(args.jsonl_path)
    
    print(f"DEBUG: Resolved Endpoint: {ckpt_path}, Exists: {os.path.exists(ckpt_path)}", flush=True)
    
    if not os.path.exists(ckpt_path):
        st.error(f"Checkpoint not found at resolved path: {ckpt_path}")
        # Stop execution to prevent ZeroShot fallback confusion
        st.stop()

    st.title("文脈認識型動画検索ビューアー")
    st.caption("Context-Aware Retrieval")
    
    # Load Resources
    try:
        model, keys, embeddings, masks, scene_map, resolved_paths, tokenizer = load_resources(idx_path, args.model_name, ckpt_path)
    except Exception as e:
        st.error(f"Failed to load resources: {e}")
        return

    if embeddings is None:
        st.warning(f"Index file '{args.index_path}' not found.")
        return
        
    # キャプションの読み込み
    captions = {}
    caption_lookup = {}
    if os.path.exists("datasets/kanazawa_scene/captions_inclusive.json"):
         with open("datasets/kanazawa_scene/captions_inclusive.json", "r") as f:
             captions = json.load(f)
             # Create lookup by basename for easier matching with absolute paths
             for k, v in captions.items():
                 caption_lookup[os.path.basename(k)] = v

    # Load Elements Data for Visualization
    # Load Elements Data for Visualization
    # Use backup file to avoid race condition with running Refinement Job
    elements_path = "datasets/kanazawa_scene/captions_elements.json"
    if not os.path.exists(elements_path):
        # Fallback if backup doesn't exist (e.g. fresh run)
        elements_path = "datasets/kanazawa_scene/captions_elements.json"

    elements_data = {}
    if os.path.exists(elements_path):
        try:
            with open(elements_path, "r") as f:
                elements_data = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Failed to load {elements_path}. It might be corrupted or writing.")
            elements_data = {}

    # --- Sidebar Filters ---
    st.sidebar.header("🔍 Filters & Modes")
    
    mode = st.sidebar.radio("モード (Mode)", ["Search (検索)", "Browse All (全シーン確認)"])
    
    # --- システム設定 (変更可能) ---
    with st.sidebar.expander("⚙️ 設定", expanded=False):
        # 変更された場合、再読み込みが必要になる場合がありますが、現状は入力のみ受け付けます
        new_model_name = st.text_input("モデル名 (Model Name)", args.model_name)
        new_index_path = st.text_input("インデックスパス (Index Path)", args.index_path)
        new_ckpt_path = st.text_input("チェックポイント (Checkpoint)", args.checkpoint_path)
    
    st.sidebar.markdown("---")
    st.sidebar.header("🔧 Debug Info")
    st.sidebar.info(f"Use Projection: {model.use_projection}")
    st.sidebar.info(f"Text Proj Layer: {'Yes' if hasattr(model, 'text_proj') else 'No'}")
    if hasattr(model, 'text_proj'):
         st.sidebar.info(f"Proj Out Dim: {model.text_proj.out_features}")
    if embeddings is not None:
         st.sidebar.info(f"Index Shape: {embeddings.shape}")
        
    # --- フィルタ (日本語表示 -> 英語ロジック) ---
    weather_map = {"晴れ": "Sunny", "曇り": "Cloudy", "雨": "Rainy"}
    time_map = {"朝": "Morning", "昼": "Daytime", "夕方": "Evening", "夜": "Night"}
    
    # 選択肢は日本語で表示
    selected_weather_jp = st.sidebar.multiselect("天気", list(weather_map.keys()), default=[])
    selected_time_jp = st.sidebar.multiselect("時間帯", list(time_map.keys()), default=[])
    
    # ロジック用に英語リストに変換
    f_weather = [weather_map[w] for w in selected_weather_jp]
    f_time = [time_map[t] for t in selected_time_jp]
    


    # ==========================
    #      BROWSE MODE
    # ==========================
    query = None



    if mode == "全シーン確認":

        st.header(f"全 {len(scene_map)} シーン一覧")
        
        page_size = 5 # Reduce page size as video players are heavy
        scene_ids = sorted(list(scene_map.keys()))
        total_pages = max(1, (len(scene_ids) - 1) // page_size + 1)
        page = st.sidebar.number_input("ページ", min_value=1, max_value=total_pages, value=1)
        
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        current_scenes = scene_ids[start_idx:end_idx]
        
        # Grid Layout (2 Columns)
        rows = [current_scenes[i:i+2] for i in range(0, len(current_scenes), 2)]
        
        for row_scenes in rows:
            cols = st.columns(2)
            
            for i, sid in enumerate(row_scenes):
                with cols[i]:
                    frames = scene_map[sid]
                    if not frames: continue
                    
                    # Card Container
                    with st.container():
                        # Header
                        st.markdown(f"""
                        <div style="background: rgba(255, 255, 255, 0.05); border-radius: 8px 8px 0 0; padding: 10px; border: 1px solid rgba(255, 255, 255, 0.1);">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <h4 style="margin: 0; font-family: monospace; font-size: 1rem;">{sid}</h4>
                                <span style="font-size: 0.8em; background: #eee; color: #444; padding: 2px 8px; border-radius: 4px; border: 1px solid #ddd;">{len(frames)} Frames</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Body (Stacked: Video -> Info)
                        # Video
                        current_path_browse = render_video_player(sid, scene_map, unique_id=f"browse_{sid}")
                        
                        # Info
                        if current_path_browse:
                            target_path = current_path_browse
                        else:
                            target_path = frames[len(frames)//2][1]
                            
                        b_basename = os.path.basename(target_path)
                        b_caption = caption_lookup.get(b_basename, "No caption.")
                        b_meta = get_metadata(target_path, b_caption)
                        
                        # Metadata Badges
                        st.markdown(f"""
                        <div style="padding: 10px; background: rgba(0, 0, 0, 0.2); border-radius: 0 0 8px 8px; border: 1px solid rgba(255, 255, 255, 0.1); border-top: none;">
                            <div style="display: flex; gap: 10px; font-size: 0.9em; margin-bottom: 5px;">
                                <span>📍 {b_meta['Location']}</span>
                                <span>☀ {b_meta['Weather']}</span>
                                <span>🕒 {b_meta['Time']}</span>
                            </div>
                            <div style="font-size: 0.8em; color: #aaa; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" title="{b_caption}">
                                {b_caption[:60]}...
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Spacer
                        st.write("")
        st.stop()

    # ==========================
    #      SEARCH MODE
    # ==========================
    st.sidebar.markdown("---")
    
    # query = st.text_input("Search Query", placeholder="e.g. A pedestrian crossing the street at night")
    
    # Text Input (Press Enter to Search)
    query = st.text_input("検索条件 (英語で入力)", placeholder="A pedestrian crossing the street at night.")

    # Filters
    st.sidebar.subheader("フィルタ設定")
    
    all_weather = ["rain", "night", "day", "cloudy"] # Example
    f_weather = st.sidebar.multiselect("天候 (Weather)", all_weather)
    
    all_times = ["Day", "Night"]
    f_time = st.sidebar.multiselect("時間帯 (Time)", all_times)
    
    min_score = st.sidebar.slider("最小スコア", 0.0, 1.0, 0.0)
    
    # クエリが入力された場合のみ検索ロジックを実行
    if query:
        device = next(model.parameters()).device
        st.write(f"Searching for: **{query}**")
        
        scores = None
        
        # Check Cache
        if query == st.session_state.get("last_query", "") and "cached_scores" in st.session_state:
            scores = st.session_state["cached_scores"]
        else:
            # 1. Query Processing
            with torch.no_grad():
                inputs = tokenizer([query], return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
                query_emb = model.encode_text(inputs["input_ids"], inputs["attention_mask"]) # (1, Lq, D)

            # 2. Similarity Computation
            with torch.no_grad():
                N = embeddings.shape[0]
                query_expanded = query_emb.expand(N, -1, -1)
                query_mask_expanded = inputs["attention_mask"].expand(N, -1)
                
                scores_list = []
                chunk_size = 128
                
                for i in range(0, N, chunk_size):
                    q_chunk = query_expanded[i:i+chunk_size]
                    q_mask_chunk = query_mask_expanded[i:i+chunk_size]
                    e_chunk = embeddings[i:i+chunk_size]
                    e_mask_chunk = masks[i:i+chunk_size]
                    
                    # Debug Shapes
                    # print(f"DEBUG: q_chunk: {q_chunk.shape}, e_chunk: {e_chunk.shape}")
                    try:
                        s = model.compute_similarity(q_chunk, e_chunk, q_mask_chunk, e_mask_chunk)
                    except RuntimeError as e:
                        st.error(f"RuntimeError: {e}")
                        st.write(f"Query Shape: {q_chunk.shape}")
                        st.write(f"Index Shape: {e_chunk.shape}")
                        st.write(f"Model Embed Dim: {model.text_proj.out_features if hasattr(model, 'text_proj') else 'N/A'}")
                        st.stop()
                    scores_list.append(s)
                    
                scores = torch.cat(scores_list, dim=0)
            
            # Update Cache
            st.session_state["last_query"] = query
            st.session_state["cached_text_emb"] = query_emb # Cache query emb for element analysis if needed
            st.session_state["cached_inputs"] = inputs
            st.session_state["cached_scores"] = scores
            
        # 3. Filtering & Deduplication
        sorted_indices = torch.argsort(scores, descending=True).cpu().numpy()
        
        results = []
        seen_scenes = set()
        
        for idx in sorted_indices:
            path = resolved_paths[idx]
            if path is None: continue
            
            key = keys[idx]
            score = scores[idx].item()
            meta = get_metadata(key, captions.get(key, ""))
            
            if f_weather and meta["Weather"] not in f_weather: continue
            if f_time and meta["Time"] not in f_time: continue
            
            sid = get_scene_id(key)
            if sid in seen_scenes: continue
            
            seen_scenes.add(sid)
            results.append({ "score": score, "path": path, "key": key, "sid": sid, "meta": meta })
            
            if len(results) >= 5: break
                
        # 4. Display Results
        st.subheader(f"上位 {len(results)} 件")
        
        for i, res in enumerate(results):
            with st.container():
                st.markdown(f"""
                <div class="result-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                        <h3 style="margin: 0;">Top-{i+1} <span style="font-size: 0.8em; color: #aaa; font-weight: normal;">(Relevance: {res['score']:.4f})</span></h3>
                    </div>
                </div>
                """, unsafe_allow_html=True) 
                
                c1, c2 = st.columns([1.5, 1])

                with c1:
                    # Render Video Player with Highlight
                    current_frame_path = render_video_player(res["sid"], scene_map, unique_id=f"rank_{i}", highlight_path=res["path"])
                    
                with c2:
                    # Dynamic Details based on current frame from player
                    # render_video_player returns currently displayed path (if static) or None (if playing/init)
                    # Use fallback if None
                    if not current_frame_path: current_frame_path = res["path"]
                    
                    is_match = (current_frame_path == res["path"])
                    c_basename = os.path.basename(current_frame_path)
                    current_caption = caption_lookup.get(c_basename, "No caption.")
                    current_meta = get_metadata(current_frame_path, current_caption)

                    st.markdown(f"**場所**: `{current_meta['Location']}` &nbsp;&nbsp; **天候**: `{current_meta['Weather']}` &nbsp;&nbsp; **時間**: `{current_meta['Time']}`")
                    
                    # --- Compute Top Factor Logic FIRST ---
                    elems = elements_data.get(res["key"], [])
                    top_factor_html = ""
                    sorted_e_idx = []
                    
                    if elems:
                        elems = list(dict.fromkeys(elems)) # Deduplicate
                        
                        # Retrieve cached query info
                        if "cached_text_emb" in st.session_state:
                             q_emb_for_vis = st.session_state["cached_text_emb"]
                             q_inputs_for_vis = st.session_state["cached_inputs"]
                        else:
                             q_emb_for_vis = query_emb 
                             q_inputs_for_vis = inputs

                        with torch.no_grad():
                            e_inputs = tokenizer(elems, padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
                            e_outs = model.encode_text(e_inputs["input_ids"], e_inputs["attention_mask"])
                            mask_exp = e_inputs["attention_mask"].unsqueeze(-1).float()
                            e_vecs = (e_outs * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1e-9)
                            
                            q_ex = q_emb_for_vis.expand(len(elems), -1, -1)
                            q_mask_ex = q_inputs_for_vis["attention_mask"].expand(len(elems), -1)
                            e_vecs_unsqueezed = e_vecs.unsqueeze(1)
                            e_mask_dummy = torch.ones(len(elems), 1).to(device)
                            
                            elem_scores = model.compute_similarity(q_ex, e_vecs_unsqueezed, q_mask_ex, e_mask_dummy)
                            sorted_e_idx = torch.argsort(elem_scores, descending=True).cpu().numpy()
                            
                            if len(sorted_e_idx) > 0:
                                top_idx = sorted_e_idx[0]
                                top_elem_txt = elems[top_idx]
                                top_score = elem_scores[top_idx].item()
                                top_factor_html = (
                                    f"<div style='margin-top: 10px; background-color: rgba(79, 195, 247, 0.1); "
                                    f"border: 1px solid #4fc3f7; border-radius: 8px; padding: 8px; color: #4fc3f7; font-weight: bold; font-size: 0.9em;'>"
                                    f"🏆 Top Factor: <span style='color: gray;'>{top_elem_txt}</span> <span style='font-size:0.8em; opacity:0.8;'>({top_score:.2f})</span>"
                                    f"</div>"
                                )


                    # Layout: 2 Columns for Matches & Top Factor
                    rc1, rc2 = st.columns([1, 1])
                    with rc1:
                         if is_match:
                            st.markdown(f"<div style='margin-top: 10px; color: #ef5350; font-weight: bold;'>一致フレーム (Matched)</div>", unsafe_allow_html=True)
                            st.image(res["path"], width=240)
                         else:
                            st.markdown(f"<div style='margin-top: 10px; color: #94a3b8;'>ℹ️ コンテキスト (前後フレーム)</div>", unsafe_allow_html=True)
                    
                    with rc2:
                         if top_factor_html:
                             st.markdown(top_factor_html, unsafe_allow_html=True)

                    
                    # Elements Visualization
                    st.markdown("---")
                    
                    if elems and len(sorted_e_idx) > 0:
                         # Token Heatmap logic inside Expander
                        with st.expander("🧠 トークン別ヒートマップ (詳細)"):
                                # Calculate detailed interactions for the top element
                                input_ids = q_inputs_for_vis["input_ids"][0] # full seq
                                attn_mask = q_inputs_for_vis["attention_mask"][0].bool()
                                
                                top_idx = sorted_e_idx[0]
                                q_tokens_emb = q_emb_for_vis[0]
                                target_elem_vec = e_vecs[top_idx]
                                token_interactions = torch.matmul(q_tokens_emb, target_elem_vec)
                                valid_interactions = token_interactions[attn_mask]
                                
                                input_ids = input_ids[attn_mask]
                                tokens = tokenizer.convert_ids_to_tokens(input_ids)
                                
                                expl_html = "<div style='display: flex; flex-wrap: wrap; gap: 4px;'>"
                                for t, s in zip(tokens, valid_interactions):
                                    s_val = s.item()
                                    intensity = max(0, min(1, (s_val - 0.1) * 2.5)) 
                                    bg_color = f"rgba(255, 0, 0, {intensity*0.3:.2f})"
                                    border_col = "rgba(255, 0, 0, 0.5)" if s_val > 0.3 else "#eee"
                                    font_weight = "bold" if s_val > 0.3 else "normal"
                                    
                                    t_clean = tokenizer.decode([tokenizer.convert_tokens_to_ids(t)]).strip()
                                    if not t_clean: t_clean = t
                                    
                                    expl_html += f"<div style='border: 1px solid {border_col}; background-color: {bg_color}; padding: 2px 6px; border-radius: 4px; text-align: center;'><div style='font-size: 0.8em; color: #555;'>{t_clean}</div><div style='font-weight: {font_weight}; font-size: 0.9em; color: #000;'>{s_val:.2f}</div></div>"
                                expl_html += "</div>"
                                st.markdown(expl_html, unsafe_allow_html=True)
                        
                        # All Elements (Collapsible)
                        with st.expander("🔍 全抽出要素 (All Elements)", expanded=False):
                            html = "<div style='line-height: 2.0;'>"
                            for ei in sorted_e_idx:
                                s_val = elem_scores[ei].item()
                                txt = elems[ei]
                                
                                if s_val > 0.25:
                                    style = "background-color: #ffebee; color: #c62828; border: 1px solid #ef5350; font-weight: bold;"
                                elif s_val > 0.10:
                                    style = "background-color: #fff8e1; color: #f57f17; border: 1px solid #ffb74d;"
                                else:
                                    style = "background-color: #f5f5f5; color: #9e9e9e; border: 1px solid #e0e0e0;"
                                    
                                html += f"<span style='{style} padding: 4px 10px; border-radius: 16px; margin-right: 8px; display: inline-block; font-size: 0.9em;'>{txt} <small>({s_val:.2f})</small></span>"
                            html += "</div>"
                            st.markdown(html, unsafe_allow_html=True)
                    else:
                        st.text("No elements index for this specific frame.")

                    with st.expander("Caption"):
                        st.write(current_caption)
                
                st.divider()


if __name__ == "__main__":
    main()
