#!/bin/bash
# ==============================================================================
# CHẠY LẶP NHIỀU SEED → LẤY MEAN ± STD
# ==============================================================================
# Thiết kế (theo lựa chọn: "3 seed nhưng chỉ ở 1 mức nén"):
#
#   • seed 0        : chạy ĐỦ 3 mức nén (0.7 0.8 0.9) cho cả Squeezed và E1
#                     -> dùng để vẽ đường cong điểm-theo-ngân-sách.
#   • seed 1, 2     : chỉ chạy mức nén REPEAT_PCT (mặc định 0.7)
#                     -> dùng để báo cáo mean ± std tại mức nén đó.
#   • baseline fullKV: không có clustering nên không phụ thuộc seed,
#                     chỉ chạy đúng một lần vào pred_baseline/.
#
# Nguồn ngẫu nhiên duy nhất là khởi tạo k-means (SQ_SEED). Sinh token là
# greedy (do_sample=False) nên với cùng một bộ cluster, pred luôn tái lập.
# => std đo đúng "độ nhạy của phương pháp với khởi tạo clustering".
#
# CÁCH DÙNG
#   bash run_repeat.sh                 # seeds 0 1 2, lặp tại 0.7
#   SEEDS="0 1 2 3 4" bash run_repeat.sh
#   REPEAT_PCT=0.9 bash run_repeat.sh
#   ONLY=sq  bash run_repeat.sh        # chỉ chạy baseline Squeezed
#   ONLY=e1  bash run_repeat.sh        # chỉ chạy E1
# ==============================================================================
set -e
set -o pipefail

SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # thu muc chua script nay
VA_PROJECT="${VA_PROJECT:-$(dirname "$SCRIPTS")}"
REPO="${REPO:-$(dirname "$VA_PROJECT")/SqueezedAttention-main}"
AWQ_MODEL_PATH="${AWQ_MODEL_PATH:-$HOME/models/LLaMA-2-7B-32K-AWQ}"
export REPO VA_PROJECT AWQ_MODEL_PATH

SEEDS="${SEEDS:-0 1 2}"
FULL_PCTS="${FULL_PCTS:-0.7 0.8 0.9}"            # mức nén chạy ở seed đầu tiên
REPEAT_PCT="${REPEAT_PCT:-0.7 0.8 0.9}"          # mức nén lặp lại ở các seed sau
ONLY="${ONLY:-e1}"                     # both | sq | e1
# Danh sách dataset, cách nhau bằng dấu cách. Hai script con đọc qua SQ_DATASETS.
DATASETS="${DATASETS:-qasper_val narrativeqa_val}"
export SQ_DATASETS="$DATASETS"
# SQ_LIMIT=2 -> chạy thử vài mẫu. Phải export để hai script con nhìn thấy.
export SQ_LIMIT="${SQ_LIMIT:-0}"
# budget (mặc định, iso-budget) | threshold (đúng công thức S_i > T của bài báo)
export SQ_SELECT_MODE="${SQ_SELECT_MODE:-threshold}"
export SQ_DEBUG="${SQ_DEBUG:-0}"

# Bang tong hop o cuoi gom THEM nhanh goc cua che do kia vao cung mot bang,
# de Sq va ClusterKV dung canh nhau. Mac dinh tu lay che do bu:
#   chay threshold -> them budget   |   chay budget -> them threshold
# Dat SQ_THEM_MODE="" de tat. Thu muc che do kia chua co thi bo qua lang le.
case "$SQ_SELECT_MODE" in
  threshold) _MODE_BU="budget"    ;;
  budget)    _MODE_BU="threshold" ;;
  *)         _MODE_BU=""          ;;
esac
SQ_THEM_MODE="${SQ_THEM_MODE-$_MODE_BU}"
# "all" = ghep cap o MOI muc nen trong BANG 3, thay vi chi mot muc.
B3_PCT="${B3_PCT:-all}"

# Lưới value-aware: duyệt MỌI cặp (beta, gamma).
# Không truyền -> giữ hành vi cũ (1 cấu hình, lấy mặc định của run_joint_kvcluster.sh).
BETAS="${BETAS:-${BETA:-0}}"
GAMMAS="${GAMMAS:-${GAMMA:-0.1}}"
# 1 = xoá thư mục cluster ngay sau mỗi (beta, gamma) — CHỈ xoá khi đã thấy .jsonl.
# Cần thiết khi quét nhiều beta: mỗi beta chiếm ~21.3 KB / token được cluster.
SQ_RM_CLUSTERS="${SQ_RM_CLUSTERS:-1}"

# ── Thành phần ④ VALUE-ENRICHED PRUNING ───────────────────────────────────
#   Bổ đề:  o = (1-W)·o′ + W·μ_P   (chính xác)
#   Thay bằng ước lượng từ value centroid:
#           o″ = (1-Ŵ)·o′ + Ŵ·μ̂_P
#
#   SQ_VBLEND là CÔNG TẮC, không phải núm xoay:
#       ""  (để trống) = TẮT hoàn toàn. Chạy y như repo gốc, tên thư mục pred
#                        KHÔNG có hậu tố -> dùng lại được kết quả [SKIP] cũ.
#       1.0            = BẬT. λ=1 là ước lượng KHÔNG CHỆCH suy trực tiếp từ
#                        bổ đề, không phải giá trị dò được.
#       0              = mốc đối chứng: đi đúng code path nhưng c=0.
#
#   SQ_VMODE   mean    = một vector trung bình V cho mỗi (layer,head)  <- CHÍNH
#                        (cùng ngân sách với Squeezed gốc)
#              cluster = μ_v,i riêng từng cụm (tốn thêm K·D_v mỗi head)
#              zero    = đối chứng μ̂=0 (loại giả thuyết "chỉ co biên độ")
#   SQ_VSHUF=1         = giả dược: xáo trộn thứ tự value centroid.
#
#   CHỈ có tác dụng với ONLY=e1. Bộ cụm Squeezed gốc không có file
#   value_centroids_*.pt nên nhánh ④ tự tắt (value_centroids=None).
SQ_VBLEND="${SQ_VBLEND:-}"
SQ_VMODE="${SQ_VMODE:-cluster}"
SQ_VSHUF="${SQ_VSHUF:-}"
# Để trống = KHÔNG export -> run_joint_kvcluster.sh không thêm hậu tố vào tên thư mục pred.
if [ -n "$SQ_VBLEND" ]; then
    export SQ_VBLEND
    if [ -n "$SQ_VMODE" ]; then export SQ_VMODE; else unset SQ_VMODE; fi
    if [ "$SQ_VSHUF" = "1" ]; then export SQ_VSHUF; else unset SQ_VSHUF; fi
else
    unset SQ_VBLEND; unset SQ_VMODE; unset SQ_VSHUF
fi

SEED_ARR=($SEEDS)
FIRST_SEED="${SEED_ARR[0]}"

echo "========================================================="
echo "  CHẠY LẶP NHIỀU SEED"
echo "  seeds        : $SEEDS"
echo "  seed đầu     : $FIRST_SEED  -> mức nén: $FULL_PCTS"
echo "  seed còn lại :            -> mức nén: $REPEAT_PCT"
echo "  chạy         : $ONLY"
echo "  datasets     : $DATASETS"
echo "  chọn cụm     : $SQ_SELECT_MODE"
echo "  gộp thêm     : ${SQ_THEM_MODE:-không}   (bảng tổng hợp cuối)"
echo "  betas        : $BETAS"
echo "  gammas       : $GAMMAS"
echo "  xoa cluster  : $SQ_RM_CLUSTERS"
echo "  vblend (4)   : ${SQ_VBLEND:-TAT}   vmode: ${SQ_VMODE:--}   vshuf: ${SQ_VSHUF:-0}"
echo "========================================================="
if [ -n "${SQ_VBLEND:-}" ] && [ "$ONLY" = "sq" ]; then
    echo ""
    echo "!! SQ_VBLEND duoc dat nhung ONLY=sq -> KHONG co tac dung."
    echo "   Bo cum Squeezed goc khong co value_centroids_*.pt."
    echo "   Muon chay thanh phan (4) thi dat ONLY=e1."
fi
echo ""
echo "⚠️  Mỗi seed phải clustering lại từ đầu (seed đổi = nhãn k-means đổi)."
echo "    Ước lượng: thời-gian-1-lần-chạy × số-seed. Colab có thể ngắt giữa"
echo "    chừng — script có [SKIP] theo từng (seed, mức nén) nên chạy lại"
echo "    lệnh này sẽ tiếp tục chỗ dở, không làm lại từ đầu."
echo ""

for SEED in "${SEED_ARR[@]}"; do
    if [ "$SEED" = "$FIRST_SEED" ]; then
        PCTS="$FULL_PCTS"
    else
        PCTS="$REPEAT_PCT"
    fi

    echo ""
    echo "#########################################################"
    echo "#  SEED $SEED   |   mức nén: $PCTS"
    echo "#########################################################"

    if [ "$ONLY" = "both" ] || [ "$ONLY" = "sq" ]; then
        echo ">>> [seed $SEED] Squeezed Attention gốc"
        SQ_SEED=$SEED SQ_PERCENTILES="$PCTS" \
            bash "${REPO}/run_offline_clustering.sh"
    fi

    if [ "$ONLY" = "both" ] || [ "$ONLY" = "e1" ]; then
      for B in $BETAS; do
        for G in $GAMMAS; do
            echo ""
            echo ">>> [seed $SEED] E1 (value-aware)  beta=$B  gamma=$G"
            SQ_SEED=$SEED SQ_PERCENTILES="$PCTS" BETA="$B" GAMMA="$G" \
                bash "${SCRIPTS}/run_joint_kvcluster.sh"

            if [ "$SQ_RM_CLUSTERS" = "1" ]; then
                # nullglob + mang bash: KHONG dung `ls | wc` vi set -e +
                # pipefail se giet script khi glob khong khop.
                shopt -s nullglob
                _cdirs=( "${REPO}"/Clusters-E1-b${B}-g${G}-s${SEED}* )
                _jsonls=( "${REPO}"/LongBench/pred_E1_b${B}_g${G}_s${SEED}*/*/*.jsonl )
                shopt -u nullglob
                if [ ${#_jsonls[@]} -gt 0 ] && [ ${#_cdirs[@]} -gt 0 ]; then
                    echo "    [RM] co ${#_jsonls[@]} file .jsonl -> xoa ${#_cdirs[@]} thu muc cluster"
                    rm -rf "${_cdirs[@]}"
                else
                    echo "    [RM] BO QUA (jsonl=${#_jsonls[@]}, cluster_dir=${#_cdirs[@]})"
                fi
            fi
        done
      done
    fi
done

echo ""
echo "========================================================="
echo "  XONG TẤT CẢ CÁC SEED. Đang tổng hợp mean ± std..."
echo "========================================================="
if [ "$SQ_LIMIT" -gt 0 ]; then
    echo "⚡ Đây là lần CHẠY THỬ ($SQ_LIMIT mẫu) — bỏ qua bước tổng hợp."
    echo "   Xoá kết quả thử:  rm -rf ${REPO}/Clusters-*-lim${SQ_LIMIT} ${REPO}/LongBench/pred_*_lim${SQ_LIMIT}"
    exit 0
fi

cd "${REPO}/LongBench"
# aggregate_runs.py xử lý MỘT dataset mỗi lần -> lặp qua từng cái.
for DS in $DATASETS; do
    echo ""
    echo "#########################################################"
    echo "#  TỔNG HỢP: $DS"
    echo "#########################################################"
    python "${SCRIPTS}/aggregate_runs.py" \
        --pred_root "${REPO}/LongBench" \
        --dataset "$DS" \
        --seeds "$SEEDS" \
        --mode "$SQ_SELECT_MODE" \
        --them_mode "$SQ_THEM_MODE" \
        --repeat_pct "$B3_PCT"
done
