# VLM2Rec — Anonymous Code Release

This repository is the official implementation of *VLM2Rec: Vision-Language Model
Embeddings for Sequential Recommendation* (anonymous submission).

It contains:

- A **VLM2Rec encoder** (LoRA-finetuned Qwen2.5-VL) trained with the
  *Weak-modality Penalized Contrastive Loss* (WPCL) and an optional
  *Cross-modal Relational Topology Regularization* auxiliary loss (CRTR).
- A pipeline to **generate item embeddings** from the trained encoder.
- A pluggable **downstream sequential-recommendation backbone** initialized
  with the generated item embeddings and trained / evaluated end-to-end.
  SASRec is included out of the box; other backbones (FMLP, GRU4Rec,
  DiffuRec, …) can be added by registering them in
  `cf_baselines/model/__init__.py`.

Baselines and ablation variants used in the paper are **not** included in this
release.

## Directory layout

```
VLM2Rec/
├── README.md
├── requirements.txt
│
│   # Stage 1: encoder training + item embedding generation
├── main.py                  # train + evaluate pipeline
├── train.py                 # encoder training (WPCL + CRTR)
├── evaluate.py              # item embedding + sequence evaluation
├── models.py                # VLEmbedder (Qwen2.5-VL + LoRA)
├── dataset.py               # SRDataset
├── loss.py                  # WPCL, CRTR
├── metrics.py               # Hit@k, NDCG@k
├── utils.py                 # collators, item embedding generation
│
├── datasets/                # data download / preprocessing helpers
│   ├── download_data.sh
│   ├── download_meta.sh
│   ├── download_img_pool.py
│   ├── build_item_meta.py
│   └── preprocess_data.py
│
└── cf_baselines/            # Stage 2: backbone recommender on top of VLM2Rec item embeddings
    ├── main.py              # backbone train + evaluate
    ├── trainer.py
    ├── dataset.py
    ├── metrics.py
    ├── utils.py
    ├── run_backbone.py      # convenience launcher (sweeps + run)
    └── model/
        ├── __init__.py         # backbone registry (MODEL_DICT)
        ├── abstract_model.py   # ID + frozen pretrained-emb fusion
        ├── sasrec.py           # currently shipped backbone
        └── sasrec_modules.py
```

## Requirements

```bash
pip install -r requirements.txt
```

Tested with PyTorch 2.7, transformers 4.55, peft 0.17 on a single RTX3090
GPU. Qwen2.5-VL weights are downloaded automatically by `transformers` and
cached under `./pretrained_vlm` (see `--vlm_cache_dir`).

## Data

Helper scripts under `datasets/` download the raw Amazon Reviews / MovieLens
files and turn them into the layout consumed by Stage 1:

```
./datasets/processed/<DATASET>.txt                 # user_id item_id_1 item_id_2 ...
./datasets/processed/<DATASET>_item_meta.csv        # columns: sid,title,imgurl
./datasets/processed/<DATASET>_image_pool/<sid>.jpg # cached item images
```

`<DATASET>` ∈ {`ml-1m`, `Beauty`, `Toys_and_Games`, `Sports`, `Clothing`}.

A typical preparation flow is:

```bash
cd datasets
bash download_data.sh                   # raw interaction logs
bash download_meta.sh                   # raw item metadata
python preprocess_data.py <DATASET>     # interactions → <DATASET>.txt
python build_item_meta.py <DATASET>     # metadata → <DATASET>_item_meta.csv
python download_img_pool.py <DATASET>   # cache item images locally
```

## Stage 1 — VLM2Rec encoder + item embeddings

Run from the repo root. The pipeline trains the encoder and then generates
item embeddings:

```bash
python main.py \
    --model_name Qwen/Qwen2.5-VL-3B-Instruct \
    --data_name ml-1m \
    --data_path ./datasets/processed \
    --loss_type wpcl \
    --aux_loss crtr \
    --temp_cf 0.05 --temp_aux 0.1 --lmd 0.1 \
    --epochs 3 --batch_size 8 --lr 1e-5 \
    --gpu_id 0 --id_time 001
```

`main.py` runs training (`train.py`) followed by evaluation (`evaluate.py`),
which encodes every catalog item with the trained encoder and stores fused /
text / image embeddings under:

```
./results/vlm_item_emb/<DATASET>/<save_desc>/epoch_<E>/<MODEL>/item_emb_*.pt
```

The Hit@k / NDCG@k of the encoder itself (in-batch retrieval against sampled
negatives) is written to `./results/evaluation/<DATASET>/<save_desc>_eval.csv`.

To run only one of the two phases, pass `--phase {embed, eval}`.

## Stage 2 — Backbone recommender on VLM2Rec item embeddings

Copy or symlink the directory of `.pt` files generated above into the layout
expected by the backbone:

```
./datasets/item_emb/<DATASET>/vlm2rec/item_emb_*.pt
```

Then train and evaluate the backbone:

```bash
cd cf_baselines
python run_backbone.py
```

`run_backbone.py` builds a small shell script under `search_scripts/` and
executes it. Each line corresponds to one configuration; for a single run the
underlying call is:

```bash
python main.py \
    --output_dir ./output/ \
    --data_dir ../datasets/processed/ \
    --item_emb_root ../datasets/item_emb/ \
    --model_name sasrec --emb_model vlm2rec \
    --data_name ml-1m \
    --lm_embed True --emb_fusion concat \
    --hidden_size 128 --batch_size 256 --lr 0.001 --epochs 300 \
    --num_negatives 0 --loss ce --do_eval False \
    --gpu_id 0 --seed 42
```

`--model_name` selects the backbone from `MODEL_DICT` in
`cf_baselines/model/__init__.py`. Add new backbones (e.g. `fmlp`, `gru4rec`,
`diffurec`) by implementing them as `AbstractModel` subclasses and
registering them there.

Results (Hit@k, NDCG@k for k ∈ {5, 10, 20}) are written to a `.log` file
under `./output/<emb_model>_<model_name>_<data_name>/`.

## License

Released under the MIT license for review purposes. See `LICENSE` once added in
the camera-ready version.
