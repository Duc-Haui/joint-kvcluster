#!/bin/bash
_GOC="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"   # goc repo, tinh tu vi tri script
# ==============================================================================
# chay_bang3.sh — MOT FILE DUY NHAT de do so lieu TABLE III (runtime & memory)
#   Squeezed Attention (beta=0)  vs  Joint KVCluster (beta=0.3, gamma=0)
#   1K-token context, batch 1, 32 token sinh, 20 lan lap.
#
#   B1. Gom cum lai o 1K cho ca 2 cau hinh, ghi thoi gian tung context
#   B2. Do latency / VRAM / RAM: All-KV, Squeezed, Joint KVCluster
#       (moi phuong phap mot tien trinh rieng)
#   B3. Tinh trung binh, % thay doi, dien 6 dong so vao Table III cua main.tex
#
# Cach chay (tren server GPU):   bash chay_bang3.sh
# Ket qua nam trong $OUT_DIR: bang3_toc_do.json/.xlsx, bang3.tex, log.
# Cac script Python duoc NHUNG o cuoi file va tu giai nen vao $OUT_DIR/_code.
# ==============================================================================

# ============================ CAU HINH (SUA O DAY) ============================
SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # thu muc chua script nay
VA_PROJECT="${VA_PROJECT:-$(dirname "$SCRIPTS")}"
REPO="${REPO:-$(dirname "$VA_PROJECT")/SqueezedAttention-main}"
AWQ_MODEL_PATH="${AWQ_MODEL_PATH:-$HOME/models/LLaMA-2-7B-32K-AWQ}"
PYTHON="python3"
MODEL_NAME="LLaMA-2-7B-32K"

DS="qasper"        # tap lay prompt (latency khong phu thuoc noi dung tap)
NC=1024            # do dai ngu canh (token) — caption Table III ghi 1K
MAU=10             # so context gom cum; context 0 = khoi dong, bo khi lay TB
SEED=0
BETA=0.3           # cau hinh tot nhat (chon tren validation)
GAMMA=0            # Joint KVCluster khong dung boost
PC=5               # % cum
OBS=100            # observation window
PERCENTILE=0.7     # p dung khi do (giong bang do cu)
LAP=20             # so lan lap do (caption: mean over 20 repetitions)
WARMUP=5
SINH=32            # so token sinh (dong "End-to-end, 32 tokens")

# main.tex de dien bang. De trong ("") = chi in so + ghi bang3.tex, khong sua file.
MAIN_TEX="${MAIN_TEX:-$SCRIPTS/ket_qua_cuoi/main.tex}"
OUT_DIR="${OUT_DIR:-$SCRIPTS/ket_qua_bang3}"
GOM_LAI=1          # 1 = luon gom cum lai (can de co thoi gian); 0 = dung lai neu da co
# ==============================================================================

set -e
set -o pipefail
HAU_TO="ctx${NC}-bang3"
CODE="${OUT_DIR}/_code"
mkdir -p "$CODE"
export AWQ_TEMPERATURE=1.0
export SQ_SEED=$SEED
export PYTHONPATH="${VA_PROJECT}:${REPO}/transformers/src:${REPO}:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- giai nen cac script nhung o cuoi file -----------------------------------
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
giai_nen() {  # $1 = ten file
    awk -v f="$1" '$0=="#@@BEGIN "f{on=1;next} $0=="#@@END "f{on=0} on' "$SELF" > "$CODE/$1"
    [ -s "$CODE/$1" ] || { echo "❌ Khong giai nen duoc $1"; exit 1; }
}
giai_nen offline_clustering_value_aware.py
giai_nen do_toc_do.py
giai_nen dien_bang3.py

# ---- kiem tra duong dan ------------------------------------------------------
for d in "$REPO/LongBench" "$VA_PROJECT/value_aware" "$REPO/transformers/src" "$AWQ_MODEL_PATH"; do
    [ -d "$d" ] || { echo "❌ Khong thay thu muc: $d  (sua phan CAU HINH)"; exit 1; }
done

# ---- B1. gom cum -------------------------------------------------------------
cd "$REPO"
for CAP in "0 0" "${BETA} ${GAMMA}"; do
    read -r B G <<< "$CAP"
    OUT="${REPO}/Clusters-E1-b${B}-g${G}-s${SEED}-${HAU_TO}/${DS}/"
    if [ "$GOM_LAI" != "1" ] && [ -f "${OUT}/.hoan_tat" ] && [ -f "${OUT}/clustering_time.jsonl" ]; then
        echo "✅ [SKIP] da co cum + thoi gian: $OUT"; continue
    fi
    mkdir -p "$OUT"
    rm -f "${OUT}"/*.pt "${OUT}/clustering_time.jsonl" "${OUT}/.hoan_tat"
    echo "🔄 B1: gom cum b=${B} g=${G} o ${NC} token, ${MAU} context -> $OUT"
    $PYTHON "$CODE/offline_clustering_value_aware.py" "$MODEL_NAME" \
        --dataset "$DS" --percent_clusters $PC --observation_window $OBS \
        --alpha 1.0 --beta "$B" --gamma "$G" --seed "$SEED" \
        --max_length "$NC" --max_samples "$MAU" \
        --output_path "$OUT" --device 0 \
        --load_awq --awq_model_path "$AWQ_MODEL_PATH" 2>&1 | tee "${OUT_DIR}/gom_cum_b${B}_g${G}.log"
    N=$(ls "${OUT}"/centroids_tensor_dict_*.pt 2>/dev/null | wc -l)
    [ "$N" -gt 0 ] || { echo "❌ Khong co file cum trong $OUT"; exit 1; }
    touch "${OUT}/.hoan_tat"
done

# ---- B2. do latency / memory -------------------------------------------------
cd "$OUT_DIR"
echo "🔄 B2: do latency/memory (All-KV, Squeezed, Joint KVCluster)"
$PYTHON "$CODE/do_toc_do.py" --repo "$REPO" --awq "$AWQ_MODEL_PATH" \
    --va_project "$VA_PROJECT" --model_name "$MODEL_NAME" \
    --dataset "$DS" --ngu_canh "$NC" --beta "$BETA" --gamma "$GAMMA" \
    --seed "$SEED" --pc "$PC" --obs_window "$OBS" --percentile "$PERCENTILE" \
    --hau_to "$HAU_TO" --lap "$LAP" --warmup "$WARMUP" --sinh "$SINH" \
    --chi "allkv sq jkv" --out bang3_toc_do 2>&1 | tee "${OUT_DIR}/do_toc_do.log"

# ---- B3. dien bang -----------------------------------------------------------
echo "🔄 B3: tinh so va dien Table III"
MAIN_ARG=()
if [ -n "$MAIN_TEX" ]; then
    if [ -f "$MAIN_TEX" ]; then MAIN_ARG=(--main "$MAIN_TEX")
    else echo "⚠️  Khong thay $MAIN_TEX -> chi ghi bang3.tex"; fi
fi
$PYTHON "$CODE/dien_bang3.py" --toc_do bang3_toc_do.json \
    --cum_sq  "${REPO}/Clusters-E1-b0-g0-s${SEED}-${HAU_TO}/${DS}/clustering_time.jsonl" \
    --cum_jkv "${REPO}/Clusters-E1-b${BETA}-g${GAMMA}-s${SEED}-${HAU_TO}/${DS}/clustering_time.jsonl" \
    "${MAIN_ARG[@]}" 2>&1 | tee "${OUT_DIR}/ket_qua_bang3.txt"

echo ""
echo "✅ XONG. Ket qua: ${OUT_DIR}  (bang3.tex, bang3_toc_do.json, ket_qua_bang3.txt)"
exit 0

# ==============================================================================
# CAC SCRIPT NHUNG (khong sua phan duoi day tru khi biet minh lam gi)
# ==============================================================================
: <<'__CAC_SCRIPT_NHUNG__'
#@@BEGIN offline_clustering_value_aware.py
# SQ_VAL_PATCH (tao_tap_valid.py)
"""
Patched offline clustering script tương thích với repo gốc SqueezedAttention.

So với offline_clustering.py gốc:
1. Dùng `run_value_aware_clustering` từ value_aware package thay vì run_clustering gốc
2. Lưu thêm value_centroids và normalized_variance
3. Thêm CLI args: --alpha, --beta, --gamma

GIỮ NGUYÊN tất cả kỹ thuật offloading VRAM từ file gốc:
- Hook offload QKV sang CPU ngay lập tức
- Clustering chạy trên CPU
- del input_ids + torch.cuda.empty_cache() sau mỗi sample
- output_attentions=False
- sdpa attention backend
- Tên file output tương thích pred.py

Cách dùng:
    python offline_clustering_value_aware.py LLaMA-2-7B-32K \\
        --dataset 2wikimqa \\
        --output_path /tmp/clusters/2wikimqa/ \\
        --percent_clusters 5 \\
        --observation_window 100 \\
        --alpha 1.0 --beta 0.5 --gamma 0.3 \\
        --device 0
"""

import argparse
import json
import os
import sys
import time
import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "transformers", "src")))

class DummyRMSNorm: pass
class DummyModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"): raise AttributeError(name)
        return DummyRMSNorm
for mod_name in ['gemma2', 'phi3', 'qwen3', 'qwen2_moe', 'mamba', 'jamba']:
    full_name = f"transformers.models.{mod_name}.modeling_{mod_name}"
    mod = DummyModule(full_name)
    sys.modules[full_name] = mod
    sys.modules[f"transformers.models.{mod_name}"] = DummyModule(f"transformers.models.{mod_name}")

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

# Local imports - giả định chạy từ root của repo SqueezedAttention
from utils.modelutils import *  # noqa
from utils.datautils import *   # noqa
from utils.model_parse import parse_model, get_layers
from squeezedattention.utils import build_chat, truncate_fn

# Value-aware extension - đảm bảo `value_aware` package nằm trên PYTHONPATH
from value_aware.clustering import run_value_aware_clustering
from value_aware.threshold import run_value_aware_global_threshold


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("model", type=str, help="Model name (key trong model2path.json)")
    p.add_argument("--output_path", type=str, default="output/")
    p.add_argument(
        "--dataset", type=str, default="trec",
        choices=[
            "narrativeqa", "qasper", "qasper_val", "narrativeqa_val", "multifieldqa_en", "hotpotqa", "2wikimqa",
            "musique", "gov_report", "qmsum", "multi_news", "trec", "triviaqa",
            "samsum", "lcc", "repobench-p",
        ],
    )
    p.add_argument("--percent_clusters", type=int, default=5,
                   help="% centroids so với fixed context length")
    p.add_argument("--observation_window", type=int, default=100)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--load_4bit", action="store_true", help="Use 4-bit quantization")
    p.add_argument("--load_8bit", action="store_true", help="Use 8-bit quantization")
    p.add_argument("--load_awq", action="store_true", help="Load AWQ 4-bit model")
    p.add_argument("--awq_model_path", type=str, default="", help="Path to AWQ model")
    # Value-aware specific
    p.add_argument("--alpha", type=float, default=1.0,
                   help="Trọng số K trong joint K-V clustering")
    p.add_argument("--beta", type=float, default=0.5,
                   help="Trọng số V. 0 = tắt value-aware (về baseline gốc)")
    p.add_argument("--gamma", type=float, default=0.3,
                   help="Hệ số boost variance khi tính threshold/retrieve")
    p.add_argument("--kmeans_iters", type=int, default=100)
    p.add_argument("--max_length", type=int, default=0,
                   help="Do dai cat prompt khi gom cum. 0 = model2maxlen.json "
                        "(giong pred.py). Dat 1024 de gom cum cho phep do "
                        "latency/memory o 1K (do_toc_do.py --ngu_canh 1024).")
    p.add_argument("--max_samples", type=int, default=-1,
                   help="Giới hạn số sample. -1 = tất cả")
    p.add_argument("--seed", type=int,
                   default=int(os.environ.get("SQ_SEED", "0")),
                   help="Seed cho k-means init. Đổi seed -> cluster khác -> "
                        "dùng để chạy lặp lấy mean/std. Mặc định lấy từ $SQ_SEED.")
    return p.parse_args()


def main():
    args = parse_args()
    DEV = torch.device(f"cuda:{args.device}")

    # Load config
    model2path = json.load(open("LongBench/config/model2path.json", "r"))
    model2maxlen = json.load(open("LongBench/config/model2maxlen.json", "r"))
    model_path = model2path[args.model]
    max_length = args.max_length if args.max_length > 0 else model2maxlen[args.model]
    print(f"max_length = {max_length}")

    print(f"=== Value-Aware Squeezed Attention Offline Clustering ===")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Config: alpha={args.alpha}, beta={args.beta}, gamma={args.gamma}, seed={args.seed}")
    print(f"Centroids: {args.percent_clusters}% of context length")
    print(f"Output: {args.output_path}")
    print()

    # ─── Load model (GIỮ NGUYÊN config từ file gốc) ─────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    config = LlamaConfig.from_pretrained(model_path)
    config.return_qkv_states = True
    config._flash_attn_2_enabled = False    # ← giống file gốc (sdpa tiết kiệm VRAM hơn)
    config._attn_implementation = "sdpa"    # ← giống file gốc
    
    kwargs = {
        "config": config,
        "torch_dtype": torch.float16,
        "device_map": {"": DEV},
        "trust_remote_code": True,
        "attn_implementation": "sdpa",
    }
    
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=False,
        )
    elif args.load_8bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
        )

    if args.load_awq:
        # Lấy file config gốc đã chỉnh sửa từ thư mục AWQ
        custom_config = LlamaConfig.from_pretrained(args.awq_model_path)
        orig_config = LlamaConfig.from_pretrained(model_path)
        if hasattr(orig_config, "rope_scaling"): custom_config.rope_scaling = orig_config.rope_scaling
        if hasattr(orig_config, "max_position_embeddings"): custom_config.max_position_embeddings = orig_config.max_position_embeddings
        custom_config.return_qkv_states = True
        custom_config._flash_attn_2_enabled = False
        custom_config._attn_implementation = "sdpa"
        
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            args.awq_model_path,
            config=custom_config,
            torch_dtype=torch.float16,
            device_map={"": DEV},
            low_cpu_mem_usage=True,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            model_path,
            **kwargs
        )
    model.eval()

    model_type = parse_model(model)
    layers = get_layers(model, model_type)

    # Load LongBench dataset
    from datasets import load_dataset
    dataset = args.dataset
    dataset_name_prompt = dataset + "_prompt"
    data = (load_dataset('json', data_files=os.path.join('${_GOC}/SqueezedAttention-main/LongBench/data_val', dataset + '.jsonl'), split='train') if dataset.endswith('_val') else load_dataset("THUDM/LongBench", dataset, split="test"))

    dataset2prompt = json.load(open("LongBench/config/dataset2prompt.json", "r"))
    prompt_format = dataset2prompt[dataset]
    prompt_only_format = dataset2prompt[dataset_name_prompt]

    # Compute shared prefix lengths
    data_all = list(data)
    if args.max_samples > 0:
        data_all = data_all[: args.max_samples]
    shared_prefix_length = {}
    for i, sample in enumerate(data_all):
        prompt = prompt_format.format(**sample)
        prompt_only = prompt_only_format.format(**sample)
        prompt, sp_len = truncate_fn(
            prompt, prompt_only, tokenizer, max_length, dataset, DEV
        )
        shared_prefix_length[i] = sp_len
        assert sp_len > 0

    # ─── Hooks để collect K, V, Q (OFFLOAD SANG CPU NGAY) ────────────────────
    all_queries_layers = []
    all_keys_layers = []
    all_values_layers = []

    def hook(module, inp, out):
        _, qkv, _ = out
        queries, keys, values = qkv
        sp_len = shared_prefix_length[dataidx]

        queries = queries[:, :, :sp_len]
        keys = keys[:, :, :sp_len]
        values = values[:, :, :sp_len]

        all_queries_layers.append(queries)
        all_keys_layers.append(keys)
        all_values_layers.append(values)

    for layer in layers:
        layer.self_attn.register_forward_hook(hook)

    os.makedirs(args.output_path, exist_ok=True)

    # ─── Loop qua samples ────────────────────────────────────────────────────
    for dataidx, d in enumerate(tqdm(data_all)):
        # Seed k-means CO DINH theo (seed, dataidx) — GIONG HET repo goc
        # (offline_clustering.py:228). Khong co dong nay thi hai nhanh
        # khoi tao tu hai diem khac nhau va beta=0 KHONG suy bien ve goc.
        torch.manual_seed(args.seed * 1000003 + dataidx)
        all_queries_layers.clear()
        all_keys_layers.clear()
        all_values_layers.clear()

        prompt = prompt_format.format(**d)
        prompt_only = prompt_only_format.format(**d)
        prompt, _ = truncate_fn(
            prompt, prompt_only, tokenizer, max_length, dataset, DEV
        )
        input_ids = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids.to(DEV)

        print(f"dataidx: {dataidx} | length of input_ids: {len(input_ids[0])}")
        print(f"dataidx: {dataidx} | shared_prefix_length: {shared_prefix_length[dataidx]}")

        # Forward pass — GIỮ NGUYÊN setting từ file gốc
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids,
                do_sample=False,
                max_new_tokens=1,
                use_cache=False,            # ← giống file gốc (tiết kiệm VRAM)
                output_attentions=False,     # ← giống file gốc (False, không phải True)
            )

        # [QUAN TRỌNG] Free VRAM ngay sau hook capture — giống file gốc
        del generated_ids, input_ids
        torch.cuda.empty_cache()

        # Số centroid
        sp_len = shared_prefix_length[dataidx]
        percentage = ((args.percent_clusters * 1.0) / 100.0)
        num_clusters = int(percentage * (sp_len - args.observation_window))
        if num_clusters < 1:
            num_clusters = 1
        print(num_clusters)

        # ─── Báo cáo bộ nhớ (giống file gốc) ────────────────────────────────
        tokens_kept = args.observation_window + num_clusters
        kv_budget_percent = (tokens_kept / sp_len) * 100

        print("\n" + "=" * 50)
        print(f" BÁO CÁO NHANH - MẪU DỮ LIỆU {dataidx}")
        print(f"  - Tổng chiều dài ngữ cảnh gốc : {sp_len} tokens")
        print(f"  - Số tokens giữ lại (KV Cache) : {tokens_kept} tokens")
        print(f"  --> KV Budget sử dụng         : {kv_budget_percent:.2f}% (Nén được {100 - kv_budget_percent:.2f}%)")

        try:
            import psutil
            ram_used_gb = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)
            print(f" TIÊU THỤ BỘ NHỚ:")
            print(f"  - System RAM (CPU) đang dùng   : {ram_used_gb:.2f} GB")
        except ImportError:
            pass

        torch.cuda.empty_cache()
        vram_allocated_gb = torch.cuda.memory_allocated(DEV) / (1024 ** 3)
        vram_peak_gb = torch.cuda.max_memory_allocated(DEV) / (1024 ** 3)
        print(f"  - GPU VRAM đang dùng (Hiện tại): {vram_allocated_gb:.2f} GB")
        print(f"  - GPU VRAM Đỉnh (Peak)         : {vram_peak_gb:.2f} GB")
        print("=" * 50 + "\n")

        t0 = time.time()

        kc_dict, vc_dict, lbl_dict, vvar_dict, nvar_dict = run_value_aware_clustering(
            all_keys_layers,
            all_values_layers,
            num_clusters=num_clusters,
            observation_window=args.observation_window,
            alpha=args.alpha,
            beta=args.beta,
            num_iters=args.kmeans_iters,
            print_log=False,
            device=DEV,
            seed_base=args.seed,
        )

        global_threshold_dict = run_value_aware_global_threshold(
            keys_layers=all_keys_layers,
            queries_layers=all_queries_layers,
            key_centroids_dict=kc_dict,
            labels_dict=lbl_dict,
            normalized_variance_dict=nvar_dict,
            num_clusters=num_clusters,
            observation_window=args.observation_window,
            gamma=args.gamma,
            device=DEV,
        )
        torch.cuda.synchronize(DEV)  # cho GPU chay xong roi moi chot gio
        clustering_time = time.time() - t0
        # Ghi thoi gian (k-means + nguong) cua TUNG context, cho Table III.
        with open(os.path.join(args.output_path, "clustering_time.jsonl"),
                  "a", encoding="utf-8") as _f:
            _f.write(json.dumps({"dataidx": dataidx, "sp_len": sp_len,
                                 "num_clusters": num_clusters,
                                 "beta": args.beta, "gamma": args.gamma,
                                 "giay": clustering_time}) + "\n")

        # ─── Save kết quả (GPU) ──────────────────────────────────────────────
        # TÊN FILE PHẢI TƯƠNG THÍCH VỚI pred.py CỦA REPO GỐC
        os.makedirs(args.output_path, exist_ok=True)
        # Bỏ qua việc chuyển về CPU trước khi save để tăng tốc độ ghi file
        pass

        # File chính — tên giống hệt file gốc để pred.py đọc được
        torch.save(kc_dict, f'{args.output_path}/centroids_tensor_dict_{dataidx}_{num_clusters}.pt')
        torch.save(lbl_dict, f'{args.output_path}/centroids_labels_dict_{dataidx}_{num_clusters}.pt')
        torch.save(global_threshold_dict, f'{args.output_path}/global_threshold_{dataidx}_{num_clusters}.pt')

        # File bổ sung — value-aware specific (pred.py gốc bỏ qua, dùng cho online VA)
        torch.save(vc_dict, f'{args.output_path}/value_centroids_{dataidx}_{num_clusters}.pt')
        torch.save(vvar_dict, f'{args.output_path}/value_variance_{dataidx}_{num_clusters}.pt')
        torch.save(nvar_dict, f'{args.output_path}/normalized_variance_{dataidx}_{num_clusters}.pt')

        # ─── Cleanup ─────────────────────────────────────────────────────────
        n_layers = len(all_keys_layers)
        for _ in range(n_layers):
            del all_queries_layers[0]
            del all_keys_layers[0]
            del all_values_layers[0]
        torch.cuda.empty_cache()

        if dataidx == 0:
            print(f"  [first sample] clustering took {clustering_time:.2f}s, "
                  f"thresholds: {global_threshold_dict}")

    print(f"\nDone. Results saved to {args.output_path}")


if __name__ == "__main__":
    main()
#@@END offline_clustering_value_aware.py
#@@BEGIN do_toc_do.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
do_toc_do.py — do LATENCY va MEMORY cho 3 phuong phap, trong MOT lan chay.

Sinh ra so lieu cho hai hinh kieu Fig. 4 va Fig. 5 cua bai bao:
    Latency (ms):          Prefill (1K prompt)  /  Generation (per token)
    Memory Allocation (MB): VRAM Usage          /  RAM Usage

BA PHUONG PHAP
    allkv   All-KV      khong nen (use_centroids tat)
    sq      Squeezed    cum b0-g0, use_value_aware=0, SQ_SELECT_MODE=threshold
    vasq    VA-Squeezed cum b{B}-g{G}, use_value_aware=1, threshold

  Squeezed dung CHINH thu muc cum beta=0 gamma=0 voi VA TAT. Theo
  code (clustering.py:141-150) beta=0 cho phan hoach TRUNG voi Squeezed goc,
  nen khong can thu muc cum rieng cho moc — dung lai cum ban da co.

VI SAO MOI PHUONG PHAP MOT TIEN TRINH RIENG
  torch.cuda.max_memory_allocated() do dinh cua CA tien trinh. Nap nam mo
  hinh trong mot tien trinh thi so VRAM cua phuong phap sau bi nhiem phan
  bo con sot cua phuong phap truoc, va con de OOM. Script tu goi lai chinh
  no bang subprocess cho tung phuong phap, moi tien trinh ghi mot file JSON
  nho, roi gop lai.

DIEU KIEN DE SO CO NGHIA
  - Ca nam phuong phap do trong CUNG mot lan chay, tren CUNG mot may.
  - Co torch.cuda.synchronize() truoc va sau moi phep do. Thieu no la do
    thoi gian GUI lenh, khong phai thoi gian CHAY.
  - KHONG can chay full du lieu va KHONG can nhieu seed: latency/memory
    khong phu thuoc noi dung dataset, cung khong phu thuoc seed gom cum.
    Cai can nhieu la SO LAN LAP DO (--lap), lay mean ± std qua cac lan lap.

CACH DUNG
    # kiem tra duong dan truoc, KHONG nap mo hinh:
    python3 do_toc_do.py --kiem_tra

    # 0) gom cum o 1K (mot lan): bash gom_cum_1k.sh
    # 1) do that (mac dinh: 1K, All-KV cung nhan eager, 20 lan lap):
    python3 do_toc_do.py --lap 20

    # 2) (nen lam) quet nhieu do dai de thay xu huong ro hon:
    NGU_CANH="1024 2048 4096" bash gom_cum_1k.sh
    python3 do_toc_do.py --ngu_canh "1024 2048 4096" --lap 20

  CHI SO (dat ten theo chuan benchmark LLM)
    TTFT  time to first token  = mot lan prefill tren ca prompt
    TPOT  time per output token = (E2E - TTFT) / (so token sinh - 1)
    E2E   end-to-end            = generate() tron ven `sinh` token
    TPS   tokens per second     = 1000 / TPOT
    Moi chi so co mean, std, p50, p95 qua --lap lan do (sau --warmup lan
    khoi dong, co torch.cuda.synchronize truoc/sau moi lan).
    Bo nho: VRAM trong so (sau khi nap) + KV cache DO THAT (tong byte
    past_key_values) + phan tam = VRAM dinh pha sinh; RAM dinh lay mau
    RSS moi 10 ms trong luc do.

  CHI CAN 1 TAP, 1 SEED: latency/memory phu thuoc do dai prompt, khong phu
  thuoc noi dung tap hay seed gom cum. Mean ± std lay qua --lap lan do.

  HAI LOAI SO VRAM TRONG FILE KET QUA
    vram_gen_mb        DO THAT o pha sinh token. Nen KV KHONG lam giam so
                       nay (cache khong bi xoa, xem modeling_llama.py:650).
    vram_hieu_dung_mb  SO TINH = vram_gen_mb - percentile * KV_ly_thuyet,
                       dung cong thuc cua benchmark_memory.py cu / Fig. 5.

XUAT RA
    <out>.json   so lieu tho (moi phuong phap mot muc)
    <out>.xlsx   bang de ve bieu do bang ve_do_do.py
"""
import argparse
import json
import os
import subprocess
import sys
import time

CAU_HINH = {
    "repo":       "",          # SqueezedAttention-main/ (chua LongBench/)
    "model_name": "LLaMA-2-7B-32K",
    "awq":        "",          # duong dan model AWQ
    "dataset":    "multifieldqa_en",
    # Do dai prompt khi do. Bai bao bao cao o 1K -> mac dinh 1024.
    # modeling_llama.py tinh
    #     K = int(percent_clusters/100 * (shared_prefix_length - obs_window))
    # va nap centroids_tensor_dict_{idx}_{K}.pt, nen CUM PHAI DUOC GOM O DUNG
    # DO DAI NAY: chay gom_cum_1k.sh truoc (ghi vao thu muc ...-ctx1024).
    # 0 = dung max_length cua pred.py (config/model2maxlen.json, 5000) va
    # dung lai cum cua cac lan chay pred.
    # Co the cho NHIEU do dai de quet, vd "1024 2048 4096" (moi do dai can
    # cum rieng: NGU_CANH="1024 2048 4096" bash gom_cum_1k.sh).
    "ngu_canh":   "1024",
    "sinh":       32,          # so token sinh ra de do toc do giai ma
    "warmup":     5,
    "lap":        20,
    "mau":        3,           # so prompt khac nhau
    "beta":       "0.05",
    "gamma":      "0.1",
    "seed":       "0",
    "pc":         5,
    "percentile": "0.7",
    "obs_window": 100,
    # Hau to cua thu muc cum. Gom cum voi SQ_LIMIT=N thi run_joint_kvcluster.sh dat
    # SUFFIX="-limN" vao TEN THU MUC -> phai khai o day, neu khong script
    # tim nham thu muc khong ton tai.
    # "auto" = "-ctx<ngu_canh>" khi ngu_canh > 0 (thu muc cua gom_cum_1k.sh),
    #          ""               khi ngu_canh = 0 (cum cua run_joint_kvcluster.sh).
    "hau_to":     "auto",
    # value_aware_squeezed_project/ — de trong thi suy tu vi tri file nay
    # (script nam trong .../value_aware_squeezed_project/scripts/).
    "va_project": "",
    "out":        "do_toc_do",
}

# (khoa, ten hien thi, dung cum?, che do chon, cum b{B}-g{G}?, bat value-aware?)
# jkv = Joint KVCluster cua bai bao (beta>0, gamma=0): CHI doi phan hoach
#   (thu muc cum b{B}-g0), con buoc giai ma giong het Squeezed -> VA tat.
#   Voi gamma=0 boost (1+gamma*sigma)=1 nen VA bat hay tat cho cung ket qua;
#   tat de khong nap them value_centroids/normalized_variance khong dung toi.
PP = [
    ("allkv", "All-KV",          False, None,        False, False),
    ("sq",    "Squeezed Attn",   True,  "threshold", False, False),
    ("vasq",  "VA-Squeezed",     True,  "threshold", True,  True),
    ("jkv",   "Joint KVCluster", True,  "threshold", True,  False),
]


def cac_duong_dan_python(a):
    """Ba duong dan ma run_joint_kvcluster.sh:37 dat vao PYTHONPATH truoc khi chay.

        export PYTHONPATH="${VA_PROJECT}:${REPO}/transformers/src:${REPO}:..."

    Thieu bat ky cai nao la import vo nghia:
      VA_PROJECT            -> goi 'value_aware'
      REPO/transformers/src -> ban transformers DA VA (modeling_llama)
      REPO                  -> goi 'squeezedattention'
    """
    return [a.va_project,
            os.path.join(a.repo, "transformers", "src"),
            a.repo]


def duong_dan_cum(a, dung_va):
    b, g = (a.beta, a.gamma) if dung_va else ("0", "0")
    return os.path.join(a.repo,
                        f"Clusters-E1-b{b}-g{g}-s{a.seed}{a.hau_to}")


def kiem_tra(a):
    """Kiem moi thu CAN TRUOC KHI nap mo hinh. Loi o day re hon loi sau 5
    phut nap model."""
    loi = []
    lb = os.path.join(a.repo, "LongBench")
    if not os.path.isdir(lb):
        loi.append(f"Khong thay LongBench/: {lb}   (dung --repo)")
    if not os.path.exists(os.path.join(lb, "pred.py")):
        loi.append(f"Khong thay pred.py trong {lb}")
    if a.awq and not os.path.isdir(a.awq):
        loi.append(f"Khong thay model AWQ: {a.awq}")
    for d, vi_sao in zip(cac_duong_dan_python(a),
                         ("goi 'value_aware' (dung --va_project)",
                          "ban transformers DA VA",
                          "goi 'squeezedattention'")):
        if not os.path.isdir(d):
            loi.append(f"Khong thay {d}   <- can cho {vi_sao}")
    for dung_va in (False, True):
        d = duong_dan_cum(a, dung_va)
        con = os.path.join(d, a.dataset)
        if not os.path.isdir(con):
            loi.append(f"Khong thay thu muc cum: {con}")
        else:
            n_cum = len([f for f in os.listdir(con)
                         if f.startswith("centroids_tensor_dict_")])
            if n_cum == 0:
                loi.append(f"Thu muc cum rong (khong co centroids_*.pt): "
                           f"{con}")
            elif n_cum < a.mau:
                loi.append(f"Chi co {n_cum} prefix da gom cum trong {con}, "
                           f"nhung --mau = {a.mau}. Giam --mau xuong "
                           f"{n_cum}, hoac gom cum them.")
    print("=" * 66)
    if loi:
        print("THIEU:")
        for x in loi:
            print("  !! " + x)
        print("=" * 66)
        return False
    print("OK — du duong dan de chay. Cac thu muc cum:")
    for dung_va in (False, True):
        print("   " + duong_dan_cum(a, dung_va))
    print("=" * 66)
    return True


# --------------------------------------------------------------- worker
def chay_mot(a, khoa):
    """Do MOT phuong phap. Chay trong tien trinh rieng."""
    import gc
    import torch

    ten, dung_cum, che_do, cum_va, bat_va = next(
        (t, c, m, cv, v) for k, t, c, m, cv, v in PP if k == khoa)
    a.ngu_canh = int(a.ngu_canh)

    lb = os.path.join(a.repo, "LongBench")
    # Thu tu giong PYTHONPATH cua run_joint_kvcluster.sh: VA_PROJECT, transformers da va,
    # REPO — roi moi den LongBench. Phai dat TRUOC 'import pred', vi pred.py
    # keo theo transformers.models.llama.
    for d in reversed(cac_duong_dan_python(a) + [lb]):
        if d not in sys.path:
            sys.path.insert(0, d)
    cwd = os.getcwd()
    os.chdir(lb)                      # pred.py doc config/ bang duong dan tuong doi
    try:
        import pred as P
        from datasets import load_dataset

        if che_do:
            os.environ["SQ_SELECT_MODE"] = che_do
        os.environ.setdefault("SQ_DEBUG", "0")

        model2path = json.load(open("config/model2path.json"))
        d2p = json.load(open("config/dataset2prompt.json"))
        max_len = (a.ngu_canh or
                   json.load(open("config/model2maxlen.json"))[a.model_name])
        print(f"   max_length = {max_len}"
              + ("  (tu config/model2maxlen.json, giong pred.py)"
                 if not a.ngu_canh else "  (ep bang --ngu_canh)"))

        cfg = {
            "path_to_clusters_cosine": duong_dan_cum(a, cum_va) + "/"
                                       + a.dataset + "/",
            "use_centroids": dung_cum,
            "hierarchical_lookup": False,
            "percent_clusters": a.pc,
            "percent_clusters_l2": -1,
            "percentile": float(a.percentile),
            "percentile_lower": 0.7,
            "obs_window": a.obs_window,
            "value_aware_gamma": float(a.gamma) if bat_va else 0.0,
            "use_value_aware": bool(bat_va),
        }
        # ---- CUNG MOT NHAN ATTENTION cho moi phuong phap -----------------
        # pred.py:159:
        #     config._attn_implementation = "eager" if use_centroids else "sdpa"
        # Nghia la All-KV chay SDPA (nhan hop nhat, rat nhanh) con cac phuong
        # phap nen chay eager (attention viet tay, cham hon nhieu lan). So hai
        # cai do voi nhau la so HAI NHAN, khong phai so hai phuong phap.
        # --cung_nhan ep All-KV cung chay eager de phep so sanh cong bang.
        attn = "eager" if (dung_cum or a.cung_nhan) else "sdpa"
        if a.cung_nhan and not dung_cum:
            _goc_auto = P.AutoModelForCausalLM.from_pretrained
            _goc_llama = P.LlamaForCausalLM.from_pretrained

            def _ep_eager(goc):
                def f(*ar, **kw):
                    c = kw.get("config")
                    if c is not None:
                        c._attn_implementation = "eager"
                        c._flash_attn_2_enabled = False
                    return goc(*ar, **kw)
                return f

            class _ShimA:
                from_pretrained = staticmethod(_ep_eager(_goc_auto))

            class _ShimL:
                from_pretrained = staticmethod(_ep_eager(_goc_llama))

            P.AutoModelForCausalLM = _ShimA
            P.LlamaForCausalLM = _ShimL
            print("   [cung_nhan] ep All-KV chay eager de so cong bang")

        dev = torch.device("cuda:0")
        t0 = time.time()
        model, tok = P.load_model_and_tokenizer(
            model2path[a.model_name], a.model_name, dev, cfg,
            load_awq=bool(a.awq), awq_model_path=(a.awq or None))
        nap_giay = time.time() - t0
        torch.cuda.synchronize()
        # Bo nho TINH: trong so mo hinh (AWQ 4-bit) nam tren GPU sau khi nap.
        vram_trong_so = torch.cuda.memory_allocated() / (1024 ** 2)

        # ---- chuan bi vai prompt dai dung ~ngu_canh token ----------------
        data = load_dataset("THUDM/LongBench", a.dataset, split="test")
        mau_in = []
        for i, o in enumerate(data):
            if len(mau_in) >= a.mau:
                break
            o = dict(o)
            o["different_prefix_index"] = i
            tho = d2p[a.dataset].format(**o)
            khong_hoi = d2p[a.dataset + "_prompt"].format(**o)
            pr, spl = P.truncate_fn(tho, khong_hoi, tok, max_len,
                                     a.dataset, dev)
            # Kiem TRUOC khi chay: K suy ra tu spl phai co file cum tuong ung.
            # Khong co thi bao ngay kem K dang co, thay vi de torch.load nem
            # FileNotFoundError kho hieu o giua phep do.
            K = max(int(a.pc / 100.0 * (spl - a.obs_window)), 1)
            if dung_cum:
                f_cum = os.path.join(cfg["path_to_clusters_cosine"],
                                     f"centroids_tensor_dict_{i}_{K}.pt")
                if not os.path.exists(f_cum):
                    co = sorted({int(x.rsplit("_", 1)[-1][:-3])
                                 for x in os.listdir(
                                     cfg["path_to_clusters_cosine"])
                                 if x.startswith("centroids_tensor_dict_")})
                    raise SystemExit(
                        f"\n!! Khong co file cum cho prefix {i} voi K={K}.\n"
                        f"   Thu muc: {cfg['path_to_clusters_cosine']}\n"
                        f"   K dang co: {co}\n"
                        f"   K = pc/100 * (shared_prefix_length - obs_window),"
                        f" nen K khac nghia la PROMPT DAI KHAC luc gom cum.\n"
                        f"   Cach sua: BO --ngu_canh (dung max_length cua "
                        f"pred.py), HOAC gom cum lai o dung do dai muon do.")
            mau_in.append((tok(pr, truncation=False, return_tensors="pt")
                           .to(dev), spl, i))

        def dat(spl, idx):
            model.model.shared_prefix_length = spl
            model.model.different_prefix_index = idx

        def dong_bo():
            torch.cuda.synchronize()

        # ---- do PREFILL --------------------------------------------------
        def mot_prefill(inp, spl, idx):
            dat(spl, idx)
            with torch.no_grad():
                model(**inp, use_cache=True)

        for _ in range(a.warmup):
            inp, spl, idx = mau_in[0]
            mot_prefill(inp, spl, idx)
        dong_bo()

        # CHI do tren MOT prompt. Luan phien nhieu prompt thi std phan anh
        # chenh lech DO DAI giua cac prompt (latency ti le voi do dai), chu
        # khong phai nhieu cua phep do — ra thanh sai so to vo nghia.
        #
        # Chon prompt DAI NHAT, khong phai prompt dau tien. Ly do: muc dich
        # cua nen KV cache la cat chi phi attention, ma chi phi do ti le voi
        # do dai ngu canh. Do o prompt ngan la do dung cho ma phuong phap
        # KHONG duoc thiet ke cho — chi phi tra centroid co dinh se lan at,
        # va moi phuong phap deu ra gan bang nhau.
        mau_in.sort(key=lambda m: -m[1])
        inp0, spl0, idx0 = mau_in[0]
        print("   do dai (token) cua cac prompt ung vien: "
              + ", ".join(str(m[1]) for m in mau_in)
              + f"   -> do tren {spl0}")
        if spl0 < max_len * 0.8:
            print(f"   !! CANH BAO: prompt dai nhat ({spl0}) con xa tran "
                  f"max_length={max_len}.")
            print(f"      Tang --mau de quet them mau (can co cum cho tung "
                  f"prefix), hoac doi --dataset sang bo co tai lieu dai hon "
                  f"(narrativeqa dai 15K-25K).")

        # ---- KV cache DO THAT: tong byte cac tensor trong past_key_values
        # sau mot lan prefill. Day la bo nho cache that su ton tai, khong
        # phai cong thuc. (Nen KV o repo nay chi che attention, nen so nay
        # se BANG NHAU giua cac phuong phap — do la ket qua, khong phai loi.)
        def _byte_cache(c):
            if c is None:
                return 0
            if torch.is_tensor(c):
                return c.numel() * c.element_size()
            tong = 0
            for ten_ds in ("key_cache", "value_cache"):
                if hasattr(c, ten_ds):
                    tong += sum(_byte_cache(x) for x in getattr(c, ten_ds))
            if tong:
                return tong
            if isinstance(c, (list, tuple)):
                return sum(_byte_cache(x) for x in c)
            return 0

        dat(spl0, idx0)
        with torch.no_grad():
            _o = model(**inp0, use_cache=True)
        kv_do = _byte_cache(getattr(_o, "past_key_values", None)) / 1024 ** 2
        del _o
        torch.cuda.empty_cache()

        # ---- RAM DINH: lay mau RSS moi 10 ms trong suot luc do ------------
        # (doc RSS mot lan sau khi chay co the bo lo dinh that.)
        import threading
        try:
            import psutil
            _proc = psutil.Process()
        except ImportError:
            _proc = None
        _ram = {"dinh": 0.0, "chay": True}

        def _lay_mau():
            while _ram["chay"] and _proc is not None:
                _ram["dinh"] = max(_ram["dinh"],
                                   _proc.memory_info().rss / 1024 ** 2)
                time.sleep(0.01)
        _luong = threading.Thread(target=_lay_mau, daemon=True)
        _luong.start()

        torch.cuda.reset_peak_memory_stats()
        pre = []
        for r in range(a.lap):
            inp, spl, idx = inp0, spl0, idx0
            dong_bo()
            t = time.perf_counter()
            mot_prefill(inp, spl, idx)
            dong_bo()
            pre.append((time.perf_counter() - t) * 1000.0)

        # ---- do GENERATION (tru phan prefill) ----------------------------
        def mot_gen(inp, spl, idx):
            dat(spl, idx)
            with torch.no_grad():
                model.generate(**inp, max_new_tokens=a.sinh, num_beams=1,
                               do_sample=False, use_cache=True)

        for _ in range(max(1, a.warmup // 2)):
            inp, spl, idx = mau_in[0]
            mot_gen(inp, spl, idx)
        dong_bo()

        vram_prefill = torch.cuda.max_memory_allocated() / (1024 ** 2)

        # Dinh bo nho RIENG pha sinh token: reset NGAY TRUOC vong lap. Day
        # moi la cho viec nen KV cache co tac dung. Dinh chung (ca prefill)
        # bi pha prefill ap dao — moi phuong phap deu tinh du KV cho toan bo
        # ngu canh roi moi cat, nen dinh chung gan nhu bang nhau.
        torch.cuda.reset_peak_memory_stats()
        gen = []
        for r in range(a.lap):
            inp, spl, idx = inp0, spl0, idx0
            dong_bo()
            t = time.perf_counter()
            mot_gen(inp, spl, idx)
            dong_bo()
            gen.append((time.perf_counter() - t) * 1000.0)

        vram_gen = torch.cuda.max_memory_allocated() / (1024 ** 2)
        vram = max(vram_prefill, vram_gen)
        vram_model = torch.cuda.memory_allocated() / (1024 ** 2)

        # ---- KV cache LY THUYET va VRAM "hieu dung" (SO TINH) --------------
        # modeling_llama.py:650 ghi TOAN BO key/value vao cache; nen KV chi
        # che attention, khong giai phong bo nho. VRAM do duoc vi the KHONG
        # giam. benchmark_memory.py cu (va Fig. 5 cua bai) bao cao
        #     hieu_dung = do_duoc - percentile * KV_ly_thuyet
        # Giu lai cot nay de ve lai duoc hinh kieu bai, nhung DAY LA SO TINH:
        # caption phai ghi "computed", khong phai "measured".
        mc = model.config
        n_kv = getattr(mc, "num_key_value_heads", mc.num_attention_heads)
        hd = mc.hidden_size // mc.num_attention_heads
        n_tok = int(inp0.input_ids.shape[-1]) + a.sinh
        kv_mb = 2 * mc.num_hidden_layers * n_kv * hd * n_tok * 2 / 1024 ** 2
        cat = float(a.percentile) if dung_cum else 0.0
        vram_hieu_dung = vram_gen - cat * kv_mb
        _ram["chay"] = False
        _luong.join(timeout=1)
        ram = _ram["dinh"] or (
            _proc.memory_info().rss / 1024 ** 2 if _proc else None)

        # ---- LUOT TACH THANH PHAN (rieng, KHONG anh huong so do o tren) ---
        # Prefill : thoi gian torch.load file cum (centroid, nhan, nguong,
        #           phuong sai) — modeling_llama.py doc lai o MOI prompt.
        # Decode  : thoi gian LlamaAttention._pytorch_centroid_lookup (cham
        #           diem cum + chon cum + nhan phuong sai cua VA).
        # CA HAI do bang dong ho tuong + torch.cuda.synchronize() TRUOC va SAU
        # moi lan goi. Khong dong bo thi:
        #   - torch.load (chep tensor len GPU) phai CHO prefill truoc chay xong
        #     -> thoi gian doc file bi thoi phong (loi cua ban truoc);
        #   - CUDA event do khoang tren dong thoi gian GPU, gom ca luc GPU ngoi
        #     cho CPU phat lenh -> sai lech khac nhau giua 1K va 4K.
        # Dong bo lam luot nay cham hon, nhung no la luot RIENG: chi lay so ms
        # cua phan tra cum; latency tong van lay tu luot do chinh o tren.
        # Phan con lai = attention (che mask tren TOAN BO KV) + FFN + cac lop
        # khac. Ti le lay tu luot nay, nhan vao latency do o tren.
        tai_cum_ms = 0.0
        lookup_ms_buoc = 0.0
        pre_mu_tam = sum(pre) / len(pre)
        if dung_cum and a.lap_tach > 0:
            import torch as _t
            _goc_load = _t.load
            _dem = {"bat": False, "giay": 0.0}

            def _load_do(*ar, **kw):
                if not _dem["bat"]:
                    return _goc_load(*ar, **kw)
                _t.cuda.synchronize()
                t0_ = time.perf_counter()
                try:
                    return _goc_load(*ar, **kw)
                finally:
                    _t.cuda.synchronize()
                    _dem["giay"] += time.perf_counter() - t0_
            _lop_attn = type(model.model.layers[0].self_attn)
            _goc_lk = getattr(_lop_attn, "_pytorch_centroid_lookup", None)
            _lk = {"giay": 0.0}

            def _lk_do(self_, *ar, **kw):
                _t.cuda.synchronize()
                t0_ = time.perf_counter()
                ra_ = _goc_lk(self_, *ar, **kw)
                _t.cuda.synchronize()
                _lk["giay"] += time.perf_counter() - t0_
                return ra_
            try:
                _t.load = _load_do
                if _goc_lk is not None:
                    _lop_attn._pytorch_centroid_lookup = _lk_do
                # prefill: chi dem thoi gian doc file cum
                _dem.update(bat=True, giay=0.0)
                for _ in range(a.lap_tach):
                    dong_bo()
                    mot_prefill(inp0, spl0, idx0)
                    dong_bo()
                _dem["bat"] = False
                tai_cum_ms = _dem["giay"] * 1000.0 / a.lap_tach
                # decode: tong thoi gian lookup / so buoc giai ma
                _lk["giay"] = 0.0
                for _ in range(a.lap_tach):
                    dong_bo()
                    mot_gen(inp0, spl0, idx0)
                    dong_bo()
                lookup_ms_buoc = (_lk["giay"] * 1000.0 /
                                  (a.lap_tach * max(a.sinh - 1, 1)))
            finally:
                _t.load = _goc_load
                if _goc_lk is not None:
                    _lop_attn._pytorch_centroid_lookup = _goc_lk
            print(f"   [tach] prefill: doc file cum {tai_cum_ms:.1f} ms "
                  f"(phan con lai {pre_mu_tam - tai_cum_ms:.1f} ms) · "
                  f"decode: centroid lookup {lookup_ms_buoc:.2f} ms/buoc")

        def ms(xs):
            mu = sum(xs) / len(xs)
            sd = ((sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5
                  if len(xs) > 1 else None)
            return mu, sd

        def pct(xs, q):
            xs = sorted(xs)
            i = (len(xs) - 1) * q / 100.0
            lo = int(i)
            hi = min(lo + 1, len(xs) - 1)
            return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)

        pre_mu, pre_sd = ms(pre)
        gen_mu, gen_sd = ms(gen)
        # TPOT (time per output token): generate() sinh token DAU TIEN ngay
        # trong pha prefill, nen phan con lai (gen - prefill) chi gom
        # (sinh - 1) buoc giai ma. Chia cho sinh se ra so thap hon thuc te.
        n_giai_ma = max(a.sinh - 1, 1)
        tpot = [max(g - pre_mu, 0.0) / n_giai_ma for g in gen]
        tok_mu, tok_sd = ms(tpot)

        ra = {"khoa": khoa, "ten": ten,
              "prefill_ms": pre_mu, "prefill_std": pre_sd,
              "token_ms": tok_mu, "token_std": tok_sd,
              "gen_tong_ms": gen_mu,
              "vram_mb": vram, "vram_gen_mb": vram_gen,
              "vram_prefill_mb": vram_prefill, "vram_model_mb": vram_model,
              "kv_mb": kv_mb, "vram_hieu_dung_mb": vram_hieu_dung,
              # --- ten chi so chuan (TTFT / TPOT / E2E / TPS) ---------------
              "ttft_ms": pre_mu, "ttft_std": pre_sd,
              "ttft_p50": pct(pre, 50), "ttft_p95": pct(pre, 95),
              "tpot_ms": tok_mu, "tpot_std": tok_sd,
              "tpot_p50": pct(tpot, 50), "tpot_p95": pct(tpot, 95),
              "e2e_ms": gen_mu, "e2e_std": gen_sd,
              "e2e_p50": pct(gen, 50), "e2e_p95": pct(gen, 95),
              "tps": (1000.0 / tok_mu) if tok_mu else None,
              # --- bo nho: tinh (trong so) + dong (KV do that) + dinh ---------
              "vram_trong_so_mb": vram_trong_so,
              "kv_do_mb": kv_do,
              "vram_tam_mb": max(vram_gen - vram_trong_so - kv_do, 0.0),
              # --- tach thanh phan latency (luot profile rieng) -------------
              "prefill_tai_cum_ms": tai_cum_ms,
              "decode_lookup_ms": lookup_ms_buoc,
              "ram_mb": ram,
              "attn": attn,
              "gpu": (torch.cuda.get_device_name(0)
                      if hasattr(torch.cuda, "get_device_name") else None),
              "awq": bool(a.awq),
              "ngu_canh": int(inp0.input_ids.shape[-1]),
              "shared_prefix_length": int(spl0),
              "sinh": a.sinh, "lap": a.lap, "nap_giay": nap_giay}

        del model
        gc.collect()
        torch.cuda.empty_cache()
        return ra
    finally:
        os.chdir(cwd)


# --------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    for k, v in CAU_HINH.items():
        ap.add_argument(f"--{k}", default=v,
                        type=(int if isinstance(v, int) else str))
    ap.add_argument("--lim", default="",
                    help="tuong duong --hau_to=-lim<N>. Dung cai nay thay vi "
                         "--hau_to: gia tri '-lim5' bat dau bang dau gach nen "
                         "argparse tuong la mot tuy chon khac.")
    ap.add_argument("--cung_nhan", action="store_true", default=True,
                    help="(MAC DINH BAT) ep All-KV chay eager giong cac "
                         "phuong phap nen, de so sanh cung mot nhan.")
    ap.add_argument("--allkv_sdpa", action="store_true",
                    help="tat --cung_nhan: All-KV chay SDPA nhu pred.py:159 "
                         "(nhanh hon nhieu lan vi khac NHAN, khong phai vi "
                         "khac phuong phap).")
    ap.add_argument("--kiem_tra", action="store_true",
                    help="chi kiem duong dan, KHONG nap mo hinh")
    ap.add_argument("--lap_tach", type=int, default=5,
                    help="so lan chay luot tach thanh phan (0 = bo qua)")
    ap.add_argument("--chi", default="",
                    help="chi do mot so phuong phap, vd 'allkv sq'")
    ap.add_argument("--_worker", default="", help=argparse.SUPPRESS)
    ap.add_argument("--_ra", default="", help=argparse.SUPPRESS)
    a = ap.parse_args()

    if not a.repo:
        a.repo = os.environ.get("REPO", os.path.expanduser(
            "~/Desktop/DucDang/SqueezedAttention-main/SqueezedAttention-main"))
    if a.allkv_sdpa:
        a.cung_nhan = False
    if not a.awq:
        # Cung model voi cac lan chay pred (run_joint_kvcluster.sh dung AWQ).
        a.awq = os.environ.get("AWQ_MODEL_PATH", "")
        mac_dinh = "${HOME}/models/LLaMA-2-7B-32K-AWQ"
        if not a.awq and os.path.isdir(mac_dinh):
            a.awq = mac_dinh
    if not a.va_project:
        a.va_project = os.environ.get("VA_PROJECT") or os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))
    def giai_hau_to(hau_to, nc):
        if a.lim:
            return f"-lim{a.lim.lstrip('-lim')}"
        if hau_to == "auto":
            return f"-ctx{nc}" if int(nc) > 0 else ""
        # Cho phep viet --hau_to lim5 (khong dau gach) cho de go.
        if hau_to and not hau_to.startswith("-"):
            return "-" + hau_to
        return hau_to

    if a._worker:
        a.hau_to = giai_hau_to(a.hau_to, int(a.ngu_canh))
        kq = chay_mot(a, a._worker)
        with open(a._ra, "w", encoding="utf-8") as f:
            json.dump(kq, f)
        return

    # ---- quet MOT hoac NHIEU do dai ngu canh ----------------------------
    hau_to_goc = a.hau_to
    cac_nc = [int(x) for x in str(a.ngu_canh).replace(",", " ").split()]
    nc_ok = []
    for nc in cac_nc:
        a.ngu_canh = nc
        a.hau_to = giai_hau_to(hau_to_goc, nc)
        print(f"\n### ngu canh {nc}")
        if kiem_tra(a):
            nc_ok.append(nc)
        else:
            print(f"   -> BO QUA do dai {nc} (thieu cum). Gom cum: "
                  f"NGU_CANH={nc} bash gom_cum_1k.sh")
    if not nc_ok:
        sys.exit(1)
    if a.kiem_tra:
        return

    lam = [k for k, *_ in PP if not a.chi or k in a.chi.split()]
    tmp = os.path.abspath(a.out + "_tam")
    os.makedirs(tmp, exist_ok=True)
    ket = []
    for nc in nc_ok:
      a.ngu_canh = nc
      a.hau_to = giai_hau_to(hau_to_goc, nc)
      for k in lam:
        ten = next(t for kk, t, *_ in PP if kk == k)
        print(f"\n>>> {ten}  ·  ngu canh {nc}  (tien trinh rieng)")
        f_ra = os.path.join(tmp, f"{k}_{nc}.json")
        cmd = [sys.executable, os.path.abspath(__file__), "--_worker", k,
               "--_ra", f_ra]
        if not a.cung_nhan:
            cmd.append("--allkv_sdpa")
        cmd.append(f"--lap_tach={a.lap_tach}")
        for key in CAU_HINH:
            v = str(getattr(a, key))
            # "=" thay vi tach doi so: gia tri "-lim5" bat dau bang dau gach,
            # tach doi so thi tien trinh con lai vap dung loi vua roi.
            cmd += [f"--{key}={v}"]
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            cac_duong_dan_python(a) + [env.get("PYTHONPATH", "")]).rstrip(
                os.pathsep)
        r = subprocess.run(cmd, env=env)
        if r.returncode != 0 or not os.path.exists(f_ra):
            print(f"!! {ten} @ {nc} THAT BAI (ma {r.returncode}). Bo qua.")
            continue
        kq = json.load(open(f_ra, encoding="utf-8"))
        kq["ngu_canh_muc_tieu"] = nc
        ket.append(kq)
        print(f"    TTFT {kq['ttft_ms']:.1f} ms (p95 {kq['ttft_p95']:.1f})  ·  "
              f"TPOT {kq['tpot_ms']:.2f} ms (p95 {kq['tpot_p95']:.2f})  ·  "
              f"TPS {kq['tps'] or 0:.1f}  ·  E2E {kq['e2e_ms']:.0f} ms")
        print(f"    VRAM: trong so {kq['vram_trong_so_mb']:.0f} + KV do "
              f"{kq['kv_do_mb']:.0f} + tam {kq['vram_tam_mb']:.0f} = dinh "
              f"{kq['vram_gen_mb']:.0f} MB  ·  hieu dung (TINH) "
              f"{kq['vram_hieu_dung_mb']:.0f} MB  ·  RAM dinh "
              f"{(kq['ram_mb'] or 0):.0f} MB")

    if not ket:
        print("\nKhong phuong phap nao do duoc.")
        sys.exit(1)

    with open(a.out + ".json", "w", encoding="utf-8") as f:
        json.dump(ket, f, indent=2, ensure_ascii=False)
    print("\nDa ghi:", os.path.abspath(a.out + ".json"))

    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font, Alignment, PatternFill
    except ImportError:
        print("(thieu openpyxl — bo qua xlsx)")
        return
    wb = Workbook()
    ws = wb.active
    ws.title = "so"
    COT = [("Phương pháp", "ten"), ("Ngữ cảnh", "ngu_canh_muc_tieu"),
           ("TTFT ms", "ttft_ms"), ("TTFT std", "ttft_std"),
           ("TTFT p95", "ttft_p95"),
           ("TPOT ms", "tpot_ms"), ("TPOT std", "tpot_std"),
           ("TPOT p95", "tpot_p95"),
           ("TPS tok/s", "tps"), ("E2E ms", "e2e_ms"), ("E2E std", "e2e_std"),
           ("VRAM trọng số", "vram_trong_so_mb"), ("KV đo MB", "kv_do_mb"),
           ("VRAM tạm", "vram_tam_mb"), ("VRAM đỉnh sinh", "vram_gen_mb"),
           ("VRAM hiệu dụng (TÍNH)", "vram_hieu_dung_mb"),
           ("RAM đỉnh MB", "ram_mb"), ("Token sinh", "sinh"),
           ("Lần lặp", "lap"), ("Nhân attn", "attn")]
    head = [h for h, _ in COT]
    ws.append(head)
    for c in ws[1]:
        c.font = Font(bold=True, color="14243D", size=10)
        c.alignment = Alignment(horizontal="center", wrap_text=True)
        c.fill = PatternFill("solid", fgColor="F1EEE7")
    for k in ket:
        ws.append([k.get(f) for _, f in COT])
    for i in range(1, len(head) + 1):
        ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = \
            (20 if i == 1 else 12)
    c = ws.cell(row=ws.max_row + 2, column=1,
                value=("Đo trong một lần chạy trên cùng một máy; mỗi phương "
                       "pháp một tiến trình riêng để số VRAM không nhiễm "
                       "phân bổ còn sót. TTFT = prefill; TPOT = (E2E - TTFT)/(token sinh - 1); "
                       "TPS = 1000/TPOT; trung bình qua "
                       f"{a.lap} lần lặp sau {a.warmup} lần khởi động. "
                       "Caption phải ghi rõ tên GPU và độ dài ngữ cảnh."))
    c.font = Font(italic=True, size=9, color="8A8578")
    wb.save(a.out + ".xlsx")
    print("Da ghi:", os.path.abspath(a.out + ".xlsx"))
    print("\nVe hinh (memory + latency + lap token):")
    print(f"  python3 ve_3_hinh.py --toc_do {os.path.basename(a.out)}.json"
          "            # VRAM do that")
    print(f"  python3 ve_3_hinh.py --toc_do {os.path.basename(a.out)}.json"
          " --vram hieu_dung   # kieu bai bao (so tinh)")


if __name__ == "__main__":
    main()
#@@END do_toc_do.py
#@@BEGIN dien_bang3.py
#!/usr/bin/env python3
"""dien_bang3.py — tinh so cho Table III va dien vao main.tex.

Doc:
  --toc_do   bang3_toc_do.json (do_toc_do.py, khoa allkv / sq / jkv)
  --cum_sq   clustering_time.jsonl cua Squeezed (beta=0)
  --cum_jkv  clustering_time.jsonl cua Joint KVCluster (beta=0.3, gamma=0)
Ghi:
  bang3.tex  cac dong cua bang (de xem / dan tay)
  --main     neu co: thay 6 dong so trong Table III (giu ban sao .bak_bang3)
"""
import argparse
import json
import re
import shutil


def tb_cum(path):
    """Thoi gian gom cum TB moi context, BO context 0 (khoi dong CUDA)."""
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    xs = [r["giay"] for r in rows if r["dataidx"] != 0] or [r["giay"] for r in rows]
    mu = sum(xs) / len(xs)
    sd = (sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5 if len(xs) > 1 else 0.0
    return mu, sd, len(xs)


def pct(a, b):
    p = (b - a) / a * 100.0
    return "$0.0\\%$" if abs(p) < 0.05 else f"${p:+.1f}\\%$"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--toc_do", required=True)
    ap.add_argument("--cum_sq", required=True)
    ap.add_argument("--cum_jkv", required=True)
    ap.add_argument("--main", default="")
    a = ap.parse_args()

    kq = {r["khoa"]: r for r in json.load(open(a.toc_do, encoding="utf-8"))}
    sq, jk = kq["sq"], kq["jkv"]
    c_sq, sd_sq, n = tb_cum(a.cum_sq)
    c_jk, sd_jk, _ = tb_cum(a.cum_jkv)

    # (nhan trong main.tex, gia tri sq, gia tri jkv, dinh dang)
    dong = [
        ("Offline clustering (s/context)", c_sq, c_jk, "{:.2f}"),
        ("Prefill (ms)", sq["ttft_ms"], jk["ttft_ms"], "{:.1f}"),
        ("Decoding (ms/token)", sq["tpot_ms"], jk["tpot_ms"], "{:.1f}"),
        ("End-to-end, $32$ tokens (ms)", sq["e2e_ms"], jk["e2e_ms"], "{:.0f}"),
        ("Peak VRAM (MiB)", sq["vram_mb"], jk["vram_mb"], "{:.0f}"),
        ("Peak host memory (MiB)", sq["ram_mb"], jk["ram_mb"], "{:.0f}"),
    ]
    tex = []
    for nhan, x, y, f in dong:
        tex.append(f"{nhan:<34}& ${f.format(x)}$ & ${f.format(y)}$ & {pct(x, y)} \\\\")
    open("bang3.tex", "w", encoding="utf-8").write("\n".join(tex) + "\n")
    print("\n".join(tex))

    print("\n--- so de sua doan van Section 'Runtime and Memory Overhead' ---")
    print(f"GPU {sq.get('gpu')}, ngu canh {sq['ngu_canh']} token, attn {sq['attn']}, "
          f"lap {sq['lap']}")
    print(f"Gom cum: Squeezed {c_sq:.2f}±{sd_sq:.2f} s, JKV {c_jk:.2f}±{sd_jk:.2f} s "
          f"(TB {n} context) -> x{c_jk / c_sq:.2f}")
    for k, ten in (("ttft", "prefill"), ("tpot", "decode"), ("e2e", "e2e")):
        print(f"{ten}: sq {sq[k + '_ms']:.2f}±{sq[k + '_std']:.2f} | "
              f"jkv {jk[k + '_ms']:.2f}±{jk[k + '_std']:.2f}")
    if "allkv" in kq:
        ak = kq["allkv"]["tpot_ms"]
        print(f"All-KV decode {ak:.1f} ms/token -> Squeezed cham x{sq['tpot_ms'] / ak:.2f}, "
              f"JKV cham x{jk['tpot_ms'] / ak:.2f}")

    if a.main:
        # newline="": giu nguyen CRLF cua file soan tren Windows
        s = open(a.main, encoding="utf-8", newline="").read()
        shutil.copy(a.main, a.main + ".bak_bang3")
        for (nhan, *_), moi in zip(dong, tex):
            mau = re.compile(r"^" + re.escape(nhan)
                             + r"[ \t]*&[^\r\n]*\\\\[ \t]*(?=\r?$)", re.M)
            s, so = mau.subn(lambda _m: moi, s, count=1)
            if so != 1:
                raise SystemExit(f"Khong tim thay dong '{nhan}' trong {a.main}")
        open(a.main, "w", encoding="utf-8", newline="").write(s)
        print(f"\nDa dien Table III vao {a.main} (ban cu: {a.main}.bak_bang3).")
        print("Con lai: bo '\\est{Blue: placeholder ...}' trong caption va sua doan van "
              "\\est{...} theo so o tren.")


if __name__ == "__main__":
    main()
#@@END dien_bang3.py
__CAC_SCRIPT_NHUNG__
