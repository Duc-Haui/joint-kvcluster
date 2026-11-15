#!/bin/bash

# ==============================================================================
# run_joint_kvcluster.sh — PIPELINE Joint KVCluster (gom cum chung Key-Value)
# Ten thu muc ket qua van giu tien to E1 (Clusters-E1-*, pred_E1_*) de khop du lieu cu.
# ==============================================================================
# Pipeline 4 bước:
#   1. Offline Clustering VALUE-AWARE → Clusters-E1-b<BETA>-s<SEED>/
#   2. Pred (Baseline full KV + E1)
#   3. Eval cả 2
#   4. Bảng kết quả: Baseline vs E1
# ==============================================================================

set -e          # Dừng ngay nếu có lỗi
set -o pipefail # để lỗi của python không bị 'tee' che mất

# ─── PATH SETUP ───────────────────────────────────────────────────────────────
SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # thu muc chua script nay
VA_PROJECT="${VA_PROJECT:-$(dirname "$SCRIPTS")}"
REPO="${REPO:-$(dirname "$VA_PROJECT")/SqueezedAttention-main}"
AWQ_MODEL_PATH="${AWQ_MODEL_PATH:-$HOME/models/LLaMA-2-7B-32K-AWQ}"
export REPO VA_PROJECT AWQ_MODEL_PATH

# Auto-fix: đảm bảo đúng phiên bản mỗi khi Colab reset runtime
#   huggingface-hub<1.0  -> transformers 4.40 (fork trong repo) mới import được is_offline_mode
#   datasets<4.0         -> datasets 4.x đã bỏ script-dataset, LongBench.py sẽ lỗi
if [ "${SQ_PIP:-0}" = "1" ]; then
    python -m pip install huggingface-hub==0.25.0 "datasets<4.0" -q
fi

# ---- CHOT AWQ_TEMPERATURE ----------------------------------------------
# T chia score o CA buoc hieu chuan nguong (clustering.py) LAN buoc suy luan
# (modeling_llama.py). Khac 1.0 la khong con dung cong thuc bai bao. Tren shell
# local, mot lan `export` song suot phien -> chot cung o day.
if [ -n "${AWQ_TEMPERATURE}" ] && [ "${AWQ_TEMPERATURE}" != "1.0" ]; then
    echo "[CANH BAO] AWQ_TEMPERATURE=${AWQ_TEMPERATURE} (khac 1.0) -> ep ve 1.0."
fi
export AWQ_TEMPERATURE=1.0

export PYTHONPATH="${VA_PROJECT}:${REPO}/transformers/src:${REPO}:$PYTHONPATH"
export HF_HUB_OFFLINE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TORCH_LIB=$(python -c "import torch, os; print(os.path.dirname(torch.__file__) + '/lib')")
NVIDIA_LIB=$(python -c "import os, site; print(os.path.join(site.getsitepackages()[0], 'nvidia', 'cuda_runtime', 'lib'))")
export LD_LIBRARY_PATH=$NVIDIA_LIB:$TORCH_LIB:${CONDA_PREFIX}/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
PYTHON="python"

# ─── DEBUG ───────────────────────────────────────────────────────────────────
# SQ_DEBUG=1 -> modeling_llama.py in ra dòng [SQDBG] mỗi lần chọn cụm,
# cho biết thực tế giữ lại bao nhiêu % token so với mục tiêu (1 - percentile).
# Tắt bằng:  SQ_DEBUG=0 bash run_joint_kvcluster.sh
export SQ_DEBUG=${SQ_DEBUG:-1}
export SQ_DEBUG_LAYER=${SQ_DEBUG_LAYER:-0}
LOG_DIR="${REPO}/LongBench/logs_E1"
mkdir -p "$LOG_DIR"

# ─── SEED ────────────────────────────────────────────────────────────────────
# Đổi SQ_SEED -> khởi tạo k-means khác -> cluster khác -> điểm khác.
# Chạy nhiều seed rồi lấy mean/std. Mỗi seed có thư mục riêng.
SEED=${SQ_SEED:-0}
export SQ_SEED=$SEED

# ─── CHẾ ĐỘ CHỌN CỤM ─────────────────────────────────────────────────────────
#   budget    (mặc định) — cắt cứng (1-percentile) token mỗi head. So sánh
#                          iso-budget, dùng để ĐO CẢI TIẾN.
#   threshold — đúng công thức S_i > T của bài báo. Số token giữ lại KHÔNG bị
#                          ràng buộc, dùng để TÁI LẬP bài báo.
# Hai chế độ ghi vào thư mục pred KHÁC NHAU (đuôi _thr) để không đè lên nhau.
export SQ_SELECT_MODE="${SQ_SELECT_MODE:-budget}"
case "$SQ_SELECT_MODE" in
  threshold) MODE_TAG="_thr" ;;
  adaptive)  MODE_TAG="_adp" ;;
  budget)    MODE_TAG="_bgt" ;;
  *) echo "!! SQ_SELECT_MODE khong hop le: '$SQ_SELECT_MODE'"
     echo "   Chi nhan: threshold | budget | adaptive"; exit 1 ;;
esac

# ─── CHẠY THỬ NHANH ──────────────────────────────────────────────────────────
# SQ_LIMIT=2  -> chỉ cluster + pred 2 mẫu đầu, để kiểm tra đường ống (seed,
# gamma, kept%) trong vài phút trước khi đốt hàng giờ GPU.
# Kết quả chạy thử ghi vào thư mục RIÊNG (đuôi -lim<N>). Bắt buộc phải tách,
# nếu không thư mục cluster thật sẽ "có file" sau lần chạy thử rồi bị [SKIP],
# và pred sẽ chết vì FileNotFoundError khi chạm mẫu thứ 3 chưa có cụm.
LIMIT=${SQ_LIMIT:-0}
if [ "$LIMIT" -gt 0 ]; then
    SUFFIX="-lim${LIMIT}"
    LIMIT_CLUSTER_ARG="--max_samples $LIMIT"
    LIMIT_PRED_ARG="--limit $LIMIT"
    echo "⚡ CHẠY THỬ: chỉ $LIMIT mẫu. Kết quả KHÔNG dùng để báo cáo."
else
    SUFFIX=""
    LIMIT_CLUSTER_ARG=""
    LIMIT_PRED_ARG=""
fi

# ─── CẤU HÌNH ────────────────────────────────────────────────────────────────
MODEL_NAME="LLaMA-2-7B-32K"
# Cho phép runner truyền danh sách dataset: SQ_DATASETS="qasper narrativeqa"
read -r -a DATASETS <<< "${SQ_DATASETS:-qasper}"
PERC_CLUSTERS=5
# Cho phép runner truyền danh sách mức nén: SQ_PERCENTILES="0.7" bash ...
read -r -a PERCENTILES <<< "${SQ_PERCENTILES:-0.7}"
OBS_WINDOW=100

# Value-Aware params
ALPHA=1.0       # Trọng số K trong joint K-V clustering
BETA=${BETA:-0.05}       # Trọng số V (E1: Có bật Joint K-V)
GAMMA=${GAMMA:-0.1}       # Boost variance (S_i *= 1 + gamma*sigma_chuan_hoa).
                # LƯU Ý VỀ TÊN GỌI: gamma đã được nối thật tới lúc suy luận qua
                # pred.py --gamma. Ablation gốc định nghĩa E1 = CHỈ joint K-V,
                # tức gamma = 0.0; đặt khác 0 là đang chạy E1+E2 gộp. Vẫn hợp lệ,
                # nhưng khi viết luận văn phải gọi đúng tên, nếu không sẽ không
                # tách được đóng góp của joint K-V và của variance boost.

if [ "${SQ_PLACEBO:-0}" = "1" ]; then SUFFIX="${SUFFIX}-placebo"; fi

# ─── ĐƯỜNG DẪN GẮN THAM SỐ ───────────────────────────────────────────────────
# QUAN TRỌNG: beta đổi nhãn k-means nên MỖI beta phải có thư mục cluster RIÊNG,
# nếu không phần [SKIP] sẽ âm thầm dùng lại cluster của beta cũ.
# gamma CO doi cluster-dir: --gamma di vao run_value_aware_global_threshold
# nen T (global_threshold_*.pt) phu thuoc gamma va nam TRONG thu muc cluster.
PATH_TO_CLUSTERS_VA="${REPO}/Clusters-E1-b${BETA}-g${GAMMA}-s${SEED}${SUFFIX}/"
VB_TAG=""
if [ -n "${SQ_VBLEND:-}" ]; then VB_TAG="_vb${SQ_VBLEND}"; fi
if [ "${SQ_VSHUF:-0}" = "1" ]; then VB_TAG="${VB_TAG}_shuf"; fi
if [ -n "${SQ_VMODE:-}" ]; then VB_TAG="${VB_TAG}_${SQ_VMODE}"; fi
PRED_DIR_E1="pred_E1_b${BETA}_g${GAMMA}_s${SEED}${SUFFIX}${MODE_TAG}${VB_TAG}"

echo "========================================================="
echo "  PIPELINE ABLATION: E1 (Joint K-V Clustering)"
echo "  Model:      $MODEL_NAME"
echo "  Datasets:   ${DATASETS[*]}"
echo "  Centroids:  ${PERC_CLUSTERS}%"
echo "  Pruning:    ${PERCENTILES[*]}"
echo "  VA params:  α=$ALPHA β=$BETA γ=$GAMMA  seed=$SEED"
echo "  Clusters:   $PATH_TO_CLUSTERS_VA"
echo "  Pred out:   LongBench/${PRED_DIR_E1}/"
echo "========================================================="

# ═════════════════════════════════════════════════════════════════════════════
# BƯỚC 1: OFFLINE CLUSTERING VALUE-AWARE
# ═════════════════════════════════════════════════════════════════════════════
echo ""
echo "========================================================="
echo ">>> BƯỚC 1: Clustering E1 cho: ${DATASETS[*]}"
echo "========================================================="

cd "$REPO"
for DATASET in "${DATASETS[@]}"; do
    CLUSTERS_DIR="${PATH_TO_CLUSTERS_VA}${DATASET}/"
    mkdir -p "$CLUSTERS_DIR"

    # ── [VA cum an toan] ─────────────────────────────────────────────────
    # Co hoan tat: chi tao khi clustering THUC SU chay xong. Thu muc khong
    # rong KHONG con duoc coi la "da xong" — vi log va file .pt do dang cung
    # lam no khong rong.
    CO_HOAN_TAT="${CLUSTERS_DIR}/.hoan_tat"
    CUM_LOG="${LOG_DIR}/clustering_${DATASET}_b${BETA}_g${GAMMA}_s${SEED}${SUFFIX}.log"
    mkdir -p "$LOG_DIR"

    if [ -f "$CO_HOAN_TAT" ]; then
        _N_PT=$(ls "${CLUSTERS_DIR}"/centroids_labels_dict_*.pt 2>/dev/null | wc -l)
        echo "✅ [SKIP] Cluster E1 ${DATASET} da hoan tat (${_N_PT} mau)."
    else
        if [ -n "$(ls -A "$CLUSTERS_DIR" 2>/dev/null)" ]; then
            _N_CU=$(ls "${CLUSTERS_DIR}"/centroids_labels_dict_*.pt 2>/dev/null | wc -l)
            echo "⚠️  [DON] ${CLUSTERS_DIR} co du lieu nhung THIEU co hoan tat"
            echo "    -> bo cum do dang (${_N_CU} mau). Xoa va cum lai tu dau."
            rm -f "${CLUSTERS_DIR}"/*.pt "${CLUSTERS_DIR}"/clustering.log
        fi
        echo "🔄 [RUN] Clustering E1 cho: ${DATASET}..."
        $PYTHON "${VA_PROJECT}/patches/offline_clustering_value_aware.py" "$MODEL_NAME" \
            --dataset "$DATASET" \
            --percent_clusters $PERC_CLUSTERS \
            --observation_window $OBS_WINDOW \
            --alpha $ALPHA \
            --beta $BETA \
            --gamma $GAMMA \
            --seed $SEED \
            $LIMIT_CLUSTER_ARG \
            --output_path "$CLUSTERS_DIR" \
            --device 0 \
            --load_awq \
            --awq_model_path "$AWQ_MODEL_PATH" 2>&1 | tee "$CUM_LOG"

        # PIPESTATUS[0] = ma loi cua PYTHON. $? o day la ma loi cua tee,
        # gan nhu luon bang 0 nen khong bao gio bat duoc that bai.
        _MA=${PIPESTATUS[0]}
        if [ "$_MA" -ne 0 ]; then
            echo "❌ [ERROR] Clustering E1 that bai cho ${DATASET} (ma $_MA)!"
            echo "    Log: $CUM_LOG"
            echo "    Bo cum do dang GIU NGUYEN de xem, nhung KHONG co co hoan"
            echo "    tat nen lan chay sau se tu don va cum lai."
            exit 1
        fi
        _N_PT=$(ls "${CLUSTERS_DIR}"/centroids_labels_dict_*.pt 2>/dev/null | wc -l)
        if [ "$_N_PT" -eq 0 ]; then
            echo "❌ [ERROR] Clustering bao thanh cong nhung KHONG co file .pt nao."
            echo "    Log: $CUM_LOG"
            exit 1
        fi
        touch "$CO_HOAN_TAT"
        echo "✅ [DONE] Clustering E1 xong cho ${DATASET} — ${_N_PT} mau."
    fi
    # ── [het VA cum an toan] ─────────────────────────────────────────────
done

# ═════════════════════════════════════════════════════════════════════════════
# BƯỚC 2 & 3: PRED + EVAL (Baseline full KV + E1)
# ═════════════════════════════════════════════════════════════════════════════
cd "${REPO}/LongBench"

for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "========================================================="
    echo ">>> BƯỚC 2 & 3: Xử lý dataset: ${DATASET}"
    echo "========================================================="

    # ─── (A) Baseline: Full KV (TẠM TẮT ĐỂ TIẾT KIỆM THỜI GIAN TUNING) ───
    # BASELINE_FILE="pred/${MODEL_NAME}_baseline/${DATASET}.jsonl"
    # if [ -f "$BASELINE_FILE" ]; then
    #     echo "✅ [SKIP] Baseline đã có: ${BASELINE_FILE}"
    # else
    #     echo "🚀 Chạy Baseline (Full KV)..."
    #     PRED_DIR=pred $PYTHON pred.py --model $MODEL_NAME --task $DATASET --load_awq --awq_model_path "$AWQ_MODEL_PATH"
    #     PRED_DIR=pred $PYTHON eval.py --model $MODEL_NAME
    # fi

    # ─── (B) E1 (lưu vào ${PRED_DIR_E1}/) ───
    for PERCENTILE in "${PERCENTILES[@]}"; do
        VA_RESULT="${PRED_DIR_E1}/${MODEL_NAME}_PC${PERC_CLUSTERS}_PERC${PERCENTILE}/${DATASET}.jsonl"

        if [ -f "$VA_RESULT" ]; then
            echo "✅ [SKIP] E1 (${PERCENTILE}) đã có: ${VA_RESULT}"
            continue
        fi

        echo "🚀 Chạy E1 (Pruning: ${PERCENTILE})..."
        PRED_LOG="${LOG_DIR}/${DATASET}_b${BETA}_g${GAMMA}_s${SEED}${SUFFIX}${MODE_TAG}_PERC${PERCENTILE}.log"
        SQ_TAG="E1 ${DATASET} s${SEED} p${PERCENTILE}" \
        PRED_DIR=$PRED_DIR_E1 $PYTHON pred.py \
            --model $MODEL_NAME \
            --use_centroids \
            --percentile $PERCENTILE \
            --percent_clusters $PERC_CLUSTERS \
            --path_to_clusters "$PATH_TO_CLUSTERS_VA" \
            --obs_window $OBS_WINDOW \
            --gamma $GAMMA \
            $LIMIT_PRED_ARG \
            --task $DATASET \
            --load_awq \
            --awq_model_path $AWQ_MODEL_PATH 2>&1 | tee "$PRED_LOG"

        # Tóm tắt nhanh kept% của riêng mức nén này.
        # Truyền tham số qua biến môi trường, KHÔNG nội suy vào mã Python,
        # để không dính lỗi lồng dấu nháy trong heredoc/-c.
        if [ "$SQ_DEBUG" = "1" ]; then
            SQ_LOG="$PRED_LOG" SQ_PCT="$PERCENTILE" $PYTHON <<'PYDBG'
import os, re
log, pct = os.environ["SQ_LOG"], float(os.environ["SQ_PCT"])
ks = []
for line in open(log, errors="ignore"):
    if "[SQDBG]" in line:
        m = re.search(r"kept=([0-9.]+)", line)
        if m:
            ks.append(float(m.group(1)))
if ks:
    ks.sort()
    n = len(ks)
    print(f"   kept: mean={sum(ks)/n:.2%}  min={ks[0]:.2%}  p50={ks[n//2]:.2%}  "
          f"max={ks[-1]:.2%}  (target={1-pct:.2%}, n={n})")
else:
    print("   [!] khong thay dong [SQDBG] - SQ_DEBUG chua toi duoc modeling_llama.py?")
PYDBG
        fi

        PRED_DIR=$PRED_DIR_E1 $PYTHON eval.py \
            --model $MODEL_NAME \
            --use_centroids \
            --percentile $PERCENTILE \
            --percent_clusters $PERC_CLUSTERS
    done
done

# ═════════════════════════════════════════════════════════════════════════════
# BƯỚC 4: BẢNG KẾT QUẢ SO SÁNH
# ═════════════════════════════════════════════════════════════════════════════
echo ""
echo "========================================================="
echo ">>> KIỂM TRA NHANH — CHỈ RIÊNG SEED $SEED  (KHÔNG DÙNG ĐỂ BÁO CÁO)"
echo "========================================================="
echo "    Đây là điểm của MỘT seed duy nhất. Một seed có thể cho kết luận"
echo "    ngược hẳn với trung bình 3 seed — đã gặp thật trong dự án này."
echo "    Số để báo cáo nằm ở BẢNG 1/2/3 do aggregate_runs.py in ra."
echo "========================================================="

# Truyền cấu hình sang khối Python qua biến môi trường, để bảng luôn bám theo
# PERCENTILES/PERC_CLUSTERS ở đầu file thay vì viết cứng hai nơi.
export PCT_LIST=$(IFS=,; echo "${PERCENTILES[*]}")
export PRED_DIR_E1 SEED
export PC_VAL=$PERC_CLUSTERS

$PYTHON <<PYEOF
import json, os
model = "$MODEL_NAME"
datasets = ["qasper"]
percentiles = [float(x) for x in os.environ["PCT_LIST"].split(",")]
pc = int(os.environ["PC_VAL"])

print(f"{'Config':<25} {'Budget':<8} " + " ".join([f"{d[:4].upper():<10}" for d in datasets]))
print("=" * 60)

def get_score(base_dir, d_name, conf_dir):
    """Đọc điểm từ result.json hoặc {dataset}.json"""
    r = f"{base_dir}/{conf_dir}/result.json"
    if os.path.exists(r):
        with open(r) as f:
            data = json.load(f)
            return data.get(d_name, data.get("score", "N/A"))
    p = f"{base_dir}/{conf_dir}/{d_name}.json"
    if os.path.exists(p):
        with open(p) as f: return json.load(f).get("score", "N/A")
    return "N/A"

def get_baseline(d_name):
    """Baseline full-KV: run_offline_clustering.sh ghi vào pred_baseline/,
    bố cục cũ để ở pred/. Thử lần lượt cả hai thay vì trả N/A ngay."""
    for bd in ("pred_baseline", "pred"):
        v = get_score(bd, d_name, f"{model}_baseline")
        if v != "N/A":
            return v
    return "N/A"

# ─── Baseline (Full KV) ───
base_scores = [get_baseline(d) for d in datasets]
print(f"{'Baseline (Full KV)':<25} {'1.000':<8} " + " ".join([f"{s:<10}" for s in base_scores]))
print("-" * 60)

# ─── E1 ───
for p in percentiles:
    conf = f"{model}_PC{pc}_PERC{p}"
    sq_scores = [get_score(os.environ["PRED_DIR_E1"], d, conf) for d in datasets]
    # Budget = % KV cache phải nạp. Centroid chỉ lưu KEY, không lưu value,
    # nên overhead metadata = pc/2 chứ không phải pc (bài báo, muc 6.1:
    # 'centroids 5%% -> metadata overhead 2.5%% ... only centroids for the
    # keys and not the values'). Nhờ vậy Sq-70%% ra 0.325 đúng như Table 2.
    budget = (pc / 200.0) + (1.0 - p)
    print(f"{'E1-'+str(int(p*100))+'%':<25} {budget:<8.3f} " + " ".join([f"{s:<10}" for s in sq_scores]))
print("=" * 60)

# ─── So sánh E1 vs Baseline ───
print()
print(f"So sánh E1 vs Baseline (chỉ seed {os.environ.get('SEED','?')}, tham khảo):")
for p in percentiles:
    conf = f"{model}_PC{pc}_PERC{p}"
    deltas = []
    for d in datasets:
        base = get_baseline(d)
        va   = get_score(os.environ["PRED_DIR_E1"], d, conf)
        if isinstance(base, (int, float)) and isinstance(va, (int, float)):
            deltas.append(f"{va - base:+.2f}")
        else:
            deltas.append("N/A")
    # Budget = % KV cache phải nạp. Centroid chỉ lưu KEY, không lưu value,
    # nên overhead metadata = pc/2 chứ không phải pc (bài báo, muc 6.1:
    # 'centroids 5%% -> metadata overhead 2.5%% ... only centroids for the
    # keys and not the values'). Nhờ vậy Sq-70%% ra 0.325 đúng như Table 2.
    budget = (pc / 200.0) + (1.0 - p)
    print(f"  {'Δ E1-'+str(int(p*100))+'%':<23} {budget:<8.3f} " + " ".join([f"{d:<10}" for d in deltas]))
PYEOF

# ═════════════════════════════════════════════════════════════════════════════
# BƯỚC 5: BẢNG KEPT% THỰC TẾ (từ log [SQDBG])
# ═════════════════════════════════════════════════════════════════════════════
if [ "$SQ_DEBUG" = "1" ]; then
echo ""
echo "========================================================="
echo ">>> KEPT% THỰC TẾ (layer $SQ_DEBUG_LAYER, trung bình trên mọi head)"
echo "========================================================="
export LOG_DIR PCT_LIST BETA GAMMA SEED SUFFIX
$PYTHON <<'PYEOF'
import os, re, glob

log_dir = os.environ["LOG_DIR"]
pcts = [x for x in os.environ["PCT_LIST"].split(",")]

print(f"{'Config':<22} {'Target':>8} {'Mean':>8} {'Min':>8} {'p50':>8} {'Max':>8} {'Clusters':>12} {'VA':>6} {'n':>5}")
print("=" * 92)
for p in pcts:
    tag = f'_b{os.environ["BETA"]}_g{os.environ["GAMMA"]}_s{os.environ["SEED"]}{os.environ.get("SUFFIX","")}_PERC{p}.log'
    for path in sorted(glob.glob(os.path.join(log_dir, "*" + tag))):
        kept, clus, va = [], [], set()
        for line in open(path, errors="ignore"):
            if "[SQDBG]" not in line:
                continue
            m = re.search(r"kept=([0-9.]+)", line)
            if m: kept.append(float(m.group(1)))
            m = re.search(r"clusters=([0-9.]+)/([0-9]+)", line)
            if m: clus.append((float(m.group(1)), int(m.group(2))))
            m = re.search(r"va=(\S+)", line)
            if m: va.add(m.group(1))
        name = os.path.basename(path).replace(".log", "")
        if not kept:
            print(f"{name:<22} {'—':>8} {'không có dòng [SQDBG]':>8}")
            continue
        kept.sort()
        n = len(kept)
        cl = f"{sum(c for c,_ in clus)/len(clus):.0f}/{clus[0][1]}" if clus else "—"
        print(f"{name:<22} {1-float(p):>8.2%} {sum(kept)/n:>8.2%} {kept[0]:>8.2%} "
              f"{kept[n//2]:>8.2%} {kept[-1]:>8.2%} {cl:>12} {','.join(sorted(va)) or '—':>6} {n:>5}")
print("=" * 92)
print()
print("Đọc bảng:")
print("  • Mean ≈ Target  -> budget mode hoạt động đúng, E1 và baseline so sánh iso-budget.")
print("  • Mean >> Target -> lưới an toàn n_min đang kéo lên (cụm sink quá to); xem max_cluster trong log.")
print("  • Min rất thấp   -> có document mà cụm top-1 đã nuốt hết ngân sách.")
print("  • VA=off         -> cluster KHÔNG mang normalized_variance: gamma không có tác dụng.")
print(f"  • Log đầy đủ: {log_dir}/")
PYEOF
fi

echo ""
echo "========================================================="
echo "  PIPELINE E1 HOÀN THÀNH"
echo "  Clusters E1:  $PATH_TO_CLUSTERS_VA"
echo "  Pred gốc:     ${REPO}/LongBench/pred/"
echo "  Pred E1:      ${REPO}/LongBench/${PRED_DIR_E1}/"
echo "  Log kept%:    ${LOG_DIR}/"
echo "========================================================="
