# Joint KVCluster: Clustering Keys and Values for Query-Aware KV Cache Retrieval

Code for the paper

> **Clustering Keys and Values for Query-Aware KV Cache Retrieval in LLM Inference**
> Hong Duc Dang, Trung Tin Nguyen, Ly Vu.
> Institute of Information and Communication Technology, Le Quy Don Technical University, Hanoi.
> WCCC 2026.

Cluster-based KV cache retrieval (Squeezed Attention, ClusterKV) groups the cached prefix tokens
by their **Key** vectors and, at each decoding step, reads only the clusters that best match the
query. **Joint KVCluster** clusters the tokens in a joint **Key–Value** space instead:

```
Z_i = Normalize([ K_i/||K_i|| ; β · V_i/||V_i|| ])
```

A single weight `β` sets the Value share of the clustering similarity (`β²/(1+β²)`); `β = 0`
recovers Squeezed Attention exactly. Only the offline clustering changes. The retrieval score,
the calibrated threshold and the attention computation are those of Squeezed Attention.

The folder `value_aware_squeezed_project/scripts/` holds the scripts that produce every number
in the paper: running all methods, choosing `β` on held-out validation data, the statistics, the
mechanism analysis and the runtime table.

---

## 1. Hardware and setting used in the paper

| Item | Value |
|---|---|
| GPU | 1 × NVIDIA Quadro RTX 5000, **16 GB VRAM** (Turing, sm_75) |
| Model | LLaMA-2-7B-32K, 4-bit **AWQ** |
| Attention | **eager** (FlashAttention-2 is not supported on this GPU) |
| Input length | prompts truncated in the middle to **5,000 tokens** |
| Output length | greedy decoding, at most 128 new tokens (64 for MultiFieldQA-en) |
| Datasets | LongBench Qasper, NarrativeQA, MultiFieldQA-en (+ held-out validation sets) |
| Clusters | `K = ⌊0.05 (N − w)⌋`, observation window `w = 100` |
| Compression targets | `p ∈ {70, 80, 90}%` |
| Seeds | 3 K-means seeds (0, 1, 2) |

### Why 5,000 tokens

The original Squeezed Attention setting uses inputs of more than 31K tokens. At that length the
KV cache of a 7B model does not fit in 16 GB, so all methods (including the uncompressed All-KV
baseline) use the same 5,000-token limit.

The limit is read from **`SqueezedAttention-main/LongBench/config/model2maxlen.json`** by both
the offline clustering and the prediction step:

```json
"LLaMA-2-7B-32K": 5000,
```

On a GPU with more memory, set it back to `31500` to match the original Squeezed Attention setting.
Clustering and prediction must use the same value. If you change it, delete the old
`Clusters-*` folders so the clusters are rebuilt.

---

## 2. Folder layout

The scripts find all paths from their own location. No path needs editing if the layout is:

```
<root>/
├── SqueezedAttention-main/                 # REPO: fork of Squeezed Attention
│   ├── transformers/src/                   #   patched transformers (modeling_llama.py)
│   ├── offline_clustering.py               #   Key-only clustering (Squeezed / ClusterKV)
│   ├── run_offline_clustering.sh           #   runner for All-KV / Squeezed / ClusterKV
│   └── LongBench/                          #   pred.py, eval.py, config/, data_val/, outputs
└── value_aware_squeezed_project/           # VA_PROJECT
    ├── value_aware/                        #   joint K–V clustering library
    ├── patches/offline_clustering_value_aware.py
    └── scripts/                            #   THIS FOLDER
```

The model is read from `$HOME/models/LLaMA-2-7B-32K-AWQ`. Every default can be overridden with an
environment variable:

| Variable | Default | Meaning |
|---|---|---|
| `REPO` | `<root>/SqueezedAttention-main` | Squeezed Attention fork |
| `VA_PROJECT` | `<root>/value_aware_squeezed_project` | this project |
| `AWQ_MODEL_PATH` | `$HOME/models/LLaMA-2-7B-32K-AWQ` | 4-bit AWQ checkpoint |

```bash
AWQ_MODEL_PATH=/data/models/llama2-7b-32k-awq bash chay_toan_bo_valid_test.sh
```

---

## 3. Installation

```bash
conda create -n jkv python=3.10 -y && conda activate jkv
pip install torch                       # a CUDA build matching your driver
pip install -r value_aware_squeezed_project/requirements.txt
pip install autoawq "huggingface-hub==0.25.0" "datasets<4.0" numpy openpyxl
```

- `huggingface-hub==0.25.0`: the patched transformers 4.40 in `REPO/transformers/src` does not
  import with `huggingface-hub ≥ 1.0`.
- `datasets<4.0`: LongBench loads its data with a dataset script, which `datasets 4.x` removed.
- The scripts put `VA_PROJECT`, `REPO/transformers/src` and `REPO` on `PYTHONPATH` themselves.
  Do **not** `pip install transformers` over the patched copy.

Place the AWQ checkpoint of LLaMA-2-7B-32K at `$HOME/models/LLaMA-2-7B-32K-AWQ` (or set
`AWQ_MODEL_PATH`).

### Quick check (a few minutes)

```bash
cd value_aware_squeezed_project/scripts
SQ_LIMIT=2 SQ_SELECT_MODE=threshold BETA=0.3 GAMMA=0 \
  SQ_DATASETS="qasper" SQ_PERCENTILES="0.7" bash run_joint_kvcluster.sh
```

`SQ_LIMIT=2` clusters and answers only the first 2 questions and writes into separate `-lim2`
folders, so it never mixes with real results.

---

## 4. Reproducing the paper

All commands run from `value_aware_squeezed_project/scripts/`. GPU runs are long, so run them in
`tmux`. Every runner skips results that are already complete and redoes files that were cut off,
so you can stop and restart at any time.

### Step 1. Validation sets (Sec. V-A, no GPU)

```bash
python3 tao_tap_valid.py --kiem_tra   # preview
python3 tao_tap_valid.py              # build qasper_val, narrativeqa_val
```

- `qasper_val`, `narrativeqa_val`: 200 questions each from the official validation splits,
  documents that overlap the LongBench test sets removed. Written to `LongBench/data_val/`.
- `multifieldqa_en`: no validation split. `tach_mfqa.py` splits its 150 questions once, at random,
  into 50 for validation (`multifieldqa_en_val`) and 100 for testing (`multifieldqa_en_test`).
  It is called automatically in Step 2.

### Step 2. All methods, 3 seeds, validation + test (Tables I and II)

```bash
KIEM_TRA=1 bash chay_toan_bo_valid_test.sh                    # count what is missing, no GPU
bash chay_toan_bo_valid_test.sh 2>&1 | tee chay_toan_bo.log  # run everything
```

Stages (choose with `GIAI_DOAN="..."`):

| Stage | What it does |
|---|---|
| `chuan_bi` | validation sets and LongBench config |
| `goc` | All-KV, Squeezed Attention (threshold rule), ClusterKV (budget rule) |
| `jkv` | Joint KVCluster on the grid `BETAS × GAMMAS`, both rules (`run_joint_kvcluster.sh`) |
| `tong_hop` | choose `(β, γ)` on validation, report on test, paired bootstrap (no GPU) |

The default grid is `BETAS="0.05 0.1 0.2 0.3 0.5 1.0 1.2"`, `GAMMAS="0 0.1"`, `MODES="threshold budget"`.
To run only the configurations in Table II (threshold rule):

```bash
MODES=threshold BETAS="0.05 0.3 1.2" GAMMAS="0 0.1" bash chay_toan_bo_valid_test.sh
MODES=threshold BETAS="0" GAMMAS="0.1" GIAI_DOAN="jkv" bash chay_toan_bo_valid_test.sh   # row β=0, γ=0.1
```

Analysis only, from existing predictions:

```bash
GIAI_DOAN="tong_hop" bash chay_toan_bo_valid_test.sh
```

Results go to `scripts/ket_qua_cuoi/`:

| File | Content |
|---|---|
| `heldout_val.*` | `(β, γ)` chosen on validation, test F1, paired bootstrap 95% CI (Table I, Table II, Sec. V-B) |
| `heldout_chia.*` | check: choose on 30% of test questions, report on 70% (20 splits) |
| `xep_hang_VALID.xlsx`, `xep_hang_TEST.xlsx` | every configuration, mean ± std over seeds, wins vs. baseline |
| `ket_qua_valid_test.xlsx` | per-run scores |
| `kept_pc.*` | fraction of tokens actually kept, from `[SQDBG]` logs |

### Step 3. Measured sparsity and the budget rule on validation (Sec. IV, Sec. V-A)

```bash
bash chay_du.sh test   # 2 questions per command, checks the pipeline
bash chay_du.sh A      # sparsity: first 40 questions, 1 seed, p = 70/80/90%
bash chay_du.sh B      # budget rule (ClusterKV vs. joint clustering) on validation, 3 seeds
python3 kept_pc.py --logs ../../SqueezedAttention-main/LongBench/logs_E1 --out ket_qua_cuoi/kept_pc
```

### Step 4. Within-cluster Value variance (Sec. V-B, "Effect on the clusters")

```bash
bash chay_phuong_sai.sh          # β = 0, 0.3, 1.2; first 20 documents per dataset; 1 seed
```

Clusters only (prefill + K-means, no generation), then `do_phuong_sai_cum.py` compares the
size-weighted within-cluster Value variance against Key-only clustering.

### Step 5. Answer self-repetition (Sec. V-B, no GPU)

```bash
python3 dem_lap.py --n 4 --seeds "0 1 2"
```

### Step 6. Runtime and memory (Table III)

```bash
bash chay_bang3.sh
```

1K-token context, batch size 1, 32 generated tokens, mean over 20 repetitions. The script
re-clusters both configurations at 1K tokens (to time the clustering), then measures All-KV,
Squeezed Attention and Joint KVCluster (`β = 0.3`), each in its own process. It is
self-contained: the Python code it needs is embedded at the end of the file and unpacked into
`ket_qua_bang3/_code/` at run time. Results go to `scripts/ket_qua_bang3/`
(`bang3_toc_do.json`, `bang3.tex`, `ket_qua_bang3.txt`). Settings (context length, number of
repetitions, `β`) are at the top of the file. Set `MAIN_TEX=path/to/main.tex` to fill Table III
in place.

---

## 5. Running one configuration

```bash
# Joint KVCluster, β = 0.3, threshold rule, seed 0
SQ_SELECT_MODE=threshold BETA=0.3 GAMMA=0 SQ_SEED=0 \
  SQ_DATASETS="qasper narrativeqa" SQ_PERCENTILES="0.7 0.8 0.9" bash run_joint_kvcluster.sh

# Several seeds of the same configuration
SQ_SELECT_MODE=threshold ONLY=e1 SEEDS="0 1 2" BETAS="0.3" GAMMAS="0" \
  DATASETS="qasper" bash run_repeat.sh
```

| Variable | Values | Meaning |
|---|---|---|
| `BETA` | `0` … `1.2` | Value weight `β` (`0` = Squeezed Attention clusters) |
| `GAMMA` | `0`, `0.1` | optional variance-based score boost (Table II); `0` = off |
| `SQ_SELECT_MODE` | `threshold` / `budget` | Squeezed Attention rule / ClusterKV rule |
| `SQ_PERCENTILES` | `"0.7 0.8 0.9"` | compression targets `p` |
| `SQ_SEED` | `0`, `1`, `2` | K-means seed |
| `SQ_DATASETS` | e.g. `"qasper_val narrativeqa"` | datasets |
| `SQ_LIMIT` | `0` (all) or `N` | test run on the first `N` questions |
| `SQ_DEBUG` | `1` / `0` | log the fraction of kept tokens (`[SQDBG]`) |

### Output folders

Inside `SqueezedAttention-main/`:

| Path | Method |
|---|---|
| `LongBench/pred_baseline/` | All-KV |
| `LongBench/pred_s<seed>_thr/` | Squeezed Attention |
| `LongBench/pred_s<seed>_bgt/` | ClusterKV |
| `LongBench/pred_E1_b<β>_g<γ>_s<seed>_thr/` | Joint KVCluster (threshold rule) |
| `LongBench/pred_E1_b<β>_g<γ>_s<seed>_bgt/` | joint clustering with the budget rule |
| `Clusters-E1-b<β>-g<γ>-s<seed>/` | Joint KVCluster clusters |

The `E1` prefix in folder names is kept from the first experiments so that existing results stay
readable; it means Joint KVCluster.

---

## 6. Scripts and the paper

| Paper item | Script(s) | GPU |
|---|---|---|
| Validation sets (Sec. V-A) | `tao_tap_valid.py`, `tach_mfqa.py` | no |
| All runs, 3 seeds, validation + test | `chay_toan_bo_valid_test.sh` → `run_joint_kvcluster.sh`, `kiem_tra_du.py` | yes |
| Table I, Table II, pooled gain +0.56 F1 [+0.05, +1.08] | `phan_tich_heldout.py`, `kiem_tra_cau_hinh_chung.py`, `xuat_ket_qua.py` | no |
| Measured sparsity (Sec. IV) | `chay_du.sh A` → `run_repeat.sh`; `kept_pc.py` | yes |
| Budget rule on validation (Sec. V-A) | `chay_du.sh B` → `run_repeat.sh`, `aggregate_runs.py` | yes |
| Within-cluster Value variance (Sec. V-B) | `chay_phuong_sai.sh` → `do_phuong_sai_cum.py` | yes |
| Answer self-repetition (Sec. V-B) | `dem_lap.py` | no |
| Table III, runtime and memory | `chay_bang3.sh` (self-contained) | yes |

---

## 7. Notes and limitations

- **No memory saving.** The prototype masks the skipped tokens in eager PyTorch and keeps the
  full KV cache resident. It reduces the KV entries read, not the memory footprint. Without a
  sparse kernel, both threshold methods decode about 1.5× slower than All-KV.
- **Threshold rule ≠ target sparsity.** Under the threshold rule, `p` only sets the calibration
  quantile, so fewer tokens are skipped than `p` suggests. Use `SQ_DEBUG=1` and `kept_pc.py` to
  measure the actual fraction.
- **Determinism.** Decoding is greedy. The only source of randomness is the K-means
  initialization (`SQ_SEED`). K-means on GPU uses `scatter_add_`, so re-clustering with the same
  seed can differ slightly (≤ ~0.2 F1 observed).

---

## Citation

```bibtex
@inproceedings{dang2026jointkvcluster,
  title     = {Clustering Keys and Values for Query-Aware {KV} Cache Retrieval in {LLM} Inference},
  author    = {Dang, Hong Duc and Nguyen, Trung Tin and Vu, Ly},
  booktitle = {Proceedings of the WCCC 2026},
  year      = {2026}
}
```

## Acknowledgements

Built on [Squeezed Attention](https://github.com/SqueezeAILab/SqueezedAttention) (Hooper et al.,
ACL 2025) and [LongBench](https://github.com/THUDM/LongBench) (Bai et al., ACL 2024).
