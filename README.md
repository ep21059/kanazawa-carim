# CARIM: Kanazawa Dataset Fine-Tuning Pipeline

このディレクトリ（`kanazawa_ver`）は、CARIMモデルを「金沢データセット（Kanazawa Dataset）」向けに特化させるため、パイプラインと実行環境を独立させたバージョンです。

## ディレクトリ構成
- `scripts/`: データセット構築、キャプション生成、要素抽出、インデックス生成などを行うPythonスクリプト群です。（元実装からKanazawa用にパスを最適化済み）
- `slurm/`: クラスター環境(Slurm)で各種処理をバックグラウンド実行するためのバッチスクリプト群です。
- `app.py`: 金沢データセット専用の検索・可視化Streamlitアプリケーションです。金沢データのタイムスタンプ仕様に合わせたシーン分割ロジック（約300フレームごとに分割）が実装されています。
- `train.py`: 追加学習（Fine-Tuning）用のエントリポイントです。

## パイプライン実行ログ
今回の追加学習に用いた全実行フローは以下の通りです。

### 1. キャプション生成と要素抽出
金沢データ（30,092枚）に対して大規模視覚言語モデルで情景説明と主要要素を付与しました。
```bash
sbatch slurm/generate_kanazawa.sbatch
sbatch --dependency=afterok:<JOBID> slurm/refine_kanazawa.sbatch
sbatch --dependency=afterok:<JOBID> slurm/merge_kanazawa.sbatch
```

### 2. 追加学習 (Fine-Tuning)
NuScenes等で事前学習された重みを初期値として、金沢データセットでの対照学習（Contrastive Learning）を実行しました。
```bash
sbatch slurm/train_kanazawa.sbatch
```
- **出力モデル重み**: `runs/carim_kanazawa_finetuned.pt`

### 3. 検索用インデックスの生成
学習済みモデルを用いて、全30,092枚の要素埋め込み（インデックス）を作成しました。
```bash
sbatch slurm/index_kanazawa.sbatch
```
- **出力インデックス**: `datasets/kanazawa_scene/processed/text_index.pt`

### 4. アプリケーション実行
StreamlitアプリケーションをSlurm上でホストし、ローカルWebブラウザから検索エンジンにアクセスします。
```bash
sbatch slurm/run_app_kanazawa.sbatch
```
SSHポートフォワーディングを使用して `http://localhost:8501` にアクセスします。

## 詳細レポート
学習の詳しい条件、所要時間、結果等については `REPORT.md` を参照してください。
