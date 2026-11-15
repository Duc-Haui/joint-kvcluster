#!/bin/bash
_GOC="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"   # goc repo, tinh tu vi tri script

# ==============================================================================
# PIPELINE SQUEEZED ATTENTION & BASELINE (QASPER + NARRATIVEQA)
# ==============================================================================

REPO="${REPO:-$_GOC/SqueezedAttention-main}"
VA_PROJECT="${VA_PROJECT:-$_GOC/value_aware_squeezed_project}"
AWQ_MODEL_PATH="${AWQ_MODEL_PATH:-$HOME/models/LLaMA-2-7B-32K-AWQ}"

# Auto-fix: đảm bảo đúng phiên bản mỗi khi Colab reset runtime
if [ "${SQ_PIP:-0}" = "1" ]; then
    python -m pip install huggingface-hub==0.25.0 "datasets<4.0" -q
fi

export PYTHONPATH="${VA_PROJECT}:${REPO}/transformers/src:${REPO}:$PYTHONPATH"
export HF_HUB_OFFLINE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NVIDIA_LIB=$(python -c "import os, site; print(os.path.join(site.getsitepackages()[0], 'nvidia', 'cuda_runtime', 'lib'))")
export LD_LIBRARY_PATH=$NVIDIA_LIB:$CONDA_PREFIX/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
PYTHON="python"

# ─── SEED / DEBUG ────────────────────────────────────────────────────────────
# SQ_SEED đổi khởi tạo k-means -> cluster khác -> điểm khác. Chạy nhiều seed
# rồi lấy mean/std. Mỗi seed có thư mục cluster + pred riêng để không đè nhau.
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

# SQ_LIMIT=2 -> chạy thử 2 mẫu, ghi vào thư mục riêng đuôi -lim<N>.
# Phải tách thư mục: [SKIP] của clustering chỉ kiểm tra "thư mục có file hay
# không", nên chạy thử vào thư mục thật sẽ làm lần chạy đầy đủ bỏ qua
# clustering rồi chết FileNotFoundError ở mẫu chưa có cụm.
LIMIT=${SQ_LIMIT:-0}
if [ "$LIMIT" -gt 0 ]; then
    SUFFIX="-lim${LIMIT}"
    LIMIT_CLUSTER_ARG="--limit $LIMIT"
    LIMIT_PRED_ARG="--limit $LIMIT"
    echo "⚡ CHẠY THỬ: chỉ $LIMIT mẫu. Kết quả KHÔNG dùng để báo cáo."
else
    SUFFIX=""
    LIMIT_CLUSTER_ARG=""
    LIMIT_PRED_ARG=""
fi
export SQ_DEBUG=${SQ_DEBUG:-1}
export SQ_DEBUG_LAYER=${SQ_DEBUG_LAYER:-0}
LOG_DIR="${REPO}/LongBench/logs_SQ"
mkdir -p "$LOG_DIR"

PATH_TO_CLUSTERS="${REPO}/Clusters-GPU-s${SEED}${SUFFIX}/"
# Baseline full-KV KHÔNG phụ thuộc seed (không có clustering) nên dùng chung
# một thư mục và chỉ chạy đúng một lần.
BASELINE_DIR="pred_baseline${SUFFIX}"
PRED_DIR_NAME="pred_s${SEED}${SUFFIX}${MODE_TAG}"
export PRED_DIR="$PRED_DIR_NAME"

PERC_CLUSTERS=5
# Cho phép runner truyền danh sách mức nén: SQ_PERCENTILES="0.7" bash ...
read -r -a PERCENTILES <<< "${SQ_PERCENTILES:-0.7}"
MODEL_NAME="LLaMA-2-7B-32K" 

# --- THÊM NARRATIVEQA VÀO ĐÂY ---
# Cho phép runner truyền danh sách dataset: SQ_DATASETS="qasper narrativeqa"
read -r -a DATASETS <<< "${SQ_DATASETS:-qasper}"
# DATASETS=("multifieldqa_en")

cd $REPO

# ─── BƯỚC 1: OFFLINE CLUSTERING ───────────────────────────────────────────────
echo "========================================================="
echo ">>> BƯỚC 1: Chạy Clustering cho các dataset: ${DATASETS[*]}"
echo "========================================================="

cd $REPO
for DATASET in "${DATASETS[@]}"; do
    CLUSTERS_DIR="${PATH_TO_CLUSTERS}${DATASET}/"
    mkdir -p "$CLUSTERS_DIR"
    
    # Check: Nếu thư mục không rỗng (đã có file .pt), thì bỏ qua bước Clustering
    if [ "$(ls -A $CLUSTERS_DIR)" ]; then
        NCL=$(ls "$CLUSTERS_DIR"/centroids_labels_dict_*.pt 2>/dev/null | wc -l)
        echo "✅ [SKIP] Đã có cụm cho ${DATASET}: ${NCL} mẫu."
        echo "   ⚠️  Nếu con số này nhỏ hơn số mẫu của dataset, pred.py sẽ chết với FileNotFoundError"
        echo "       khi chạm mẫu chưa có cụm. Xoá ${CLUSTERS_DIR} để cluster lại từ đầu."
    else
        echo "🔄 [RUN] Không tìm thấy cụm, bắt đầu Clustering cho: ${DATASET}..."
        $PYTHON offline_clustering.py $MODEL_NAME \
            --dataset $DATASET \
            --percent_clusters $PERC_CLUSTERS \
            --observation_window 100 \
            --load_awq \
            --awq_model_path "$AWQ_MODEL_PATH" \
            --seed $SEED \
            $LIMIT_CLUSTER_ARG \
            --output_path "$CLUSTERS_DIR"
        
        if [ $? -ne 0 ]; then
            echo "❌ [ERROR] Clustering thất bại!"
            exit 1
        fi
    fi
done

cd ${REPO}/LongBench

# ─── BƯỚC 2 & 3: CHẠY PRED & EVAL ─────────────────────────────────────────────
for DATASET in "${DATASETS[@]}"; do
    echo "========================================================="
    echo ">>> BƯỚC 2 & 3: Xử lý dataset: ${DATASET}"
    echo "========================================================="

    # 1. Chạy Baseline (full KV) — không phụ thuộc seed, chỉ chạy một lần
    if [ -f "${BASELINE_DIR}/${MODEL_NAME}_baseline/${DATASET}.jsonl" ]; then
        echo "✅ [SKIP] Baseline đã có: ${BASELINE_DIR}/${MODEL_NAME}_baseline/${DATASET}.jsonl"
    else
        echo "🚀 Chạy Baseline..."
        PRED_DIR=$BASELINE_DIR $PYTHON pred.py --model $MODEL_NAME --task $DATASET $LIMIT_PRED_ARG --load_awq --awq_model_path "$AWQ_MODEL_PATH"
        PRED_DIR=$BASELINE_DIR $PYTHON eval.py --model $MODEL_NAME
    fi

    # 2. Chạy Squeezed Attention
    for PERCENTILE in "${PERCENTILES[@]}"; do
        SQ_RESULT="${PRED_DIR_NAME}/${MODEL_NAME}_PC${PERC_CLUSTERS}_PERC${PERCENTILE}/${DATASET}.jsonl"
        if [ -f "$SQ_RESULT" ]; then
            echo "✅ [SKIP] Sq seed=$SEED p=$PERCENTILE đã có: $SQ_RESULT"
            continue
        fi
        echo "🚀 Chạy Squeezed Attention (seed: ${SEED}, Pruning: ${PERCENTILE})..."
        SQ_TAG="SQ ${DATASET} s${SEED} p${PERCENTILE}" \
        $PYTHON pred.py \
            --model $MODEL_NAME \
            --use_centroids \
            --percentile $PERCENTILE \
            --percent_clusters $PERC_CLUSTERS \
            --path_to_clusters $PATH_TO_CLUSTERS \
            --obs_window 100 \
            $LIMIT_PRED_ARG \
            --task $DATASET \
            --load_awq \
            --awq_model_path "$AWQ_MODEL_PATH" 2>&1 \
            | tee "${LOG_DIR}/${DATASET}_s${SEED}${SUFFIX}${MODE_TAG}_PERC${PERCENTILE}.log"

        # 4. Chạy Eval (BỎ --task $DATASET)
        $PYTHON eval.py \
            --model $MODEL_NAME \
            --use_centroids \
            --percentile $PERCENTILE \
            --percent_clusters $PERC_CLUSTERS
    done
done

# ─── BƯỚC 4: BẢNG KẾT QUẢ TỔNG HỢP ───────────────────────────────────────────
echo "========================================================="
echo ">>> BẢNG KẾT QUẢ CUỐI CÙNG (seed=$SEED)"
echo "========================================================="

# Truyền cấu hình sang khối Python qua biến môi trường, để bảng luôn bám theo
# PERCENTILES/PERC_CLUSTERS ở đầu file thay vì viết cứng hai nơi.
export PCT_LIST=$(IFS=,; echo "${PERCENTILES[*]}")
export PC_VAL=$PERC_CLUSTERS
export BASELINE_DIR

$PYTHON << 'PYEOF'
import json, os
model = "LLaMA-2-7B-32K"
datasets = ["qasper"]
percentiles = [float(x) for x in os.environ["PCT_LIST"].split(",")]
pred_dir = os.environ.get("PRED_DIR", "pred")

print(f"{'Config':<15} {'Budget':<8} " + " ".join([f"{d[:4].upper():<10}" for d in datasets]))
print("="*60)

# Hàm lấy điểm
def get_score(d_name, conf_dir, base=None):
    p = f"{base or pred_dir}/{conf_dir}/result.json"
    if os.path.exists(p):
        with open(p) as f: return json.load(f).get(d_name, "N/A")
    return "N/A"

# Baseline (thư mục dùng chung, không theo seed)
bdir = os.environ.get("BASELINE_DIR", "pred_baseline")
base_scores = [get_score(d, f"{model}_baseline", base=bdir) for d in datasets]
print(f"{'All KV':<15} {'1.000':<8} " + " ".join([f"{s:<10}" for s in base_scores]))

# Squeezed
perc_clusters = int(os.environ["PC_VAL"])
for p in percentiles:
    conf = f"{model}_PC{perc_clusters}_PERC{p}"
    sq_scores = [get_score(d, conf) for d in datasets]
    # Budget = % KV cache phải nạp. Centroid chỉ lưu KEY, không lưu value,
    # nên overhead metadata = pc/2 chứ không phải pc (bài báo, muc 6.1:
    # 'centroids 5%% -> metadata overhead 2.5%% ... only centroids for the
    # keys and not the values'). Nhờ vậy Sq-70%% ra 0.325 đúng như Table 2.
    budget = (perc_clusters / 200.0) + (1.0 - p)
    print(f"{'Sq-'+str(int(p*100))+'%':<15} {budget:<8.3f} " + " ".join([f"{s:<10}" for s in sq_scores]))
print("="*60)
PYEOF