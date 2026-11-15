#!/bin/bash
# ==============================================================================
# chay_toan_bo_valid_test.sh — CHAY DU: 3 seed x 3 tap x 3 muc nen x 2 luat chon cum,
# CA tap VALID (chon beta, gamma) LAN tap TEST (bao cao), roi tong hop + bootstrap.
#
#   VALID : qasper_val, narrativeqa_val (tap validation chinh thuc, tao_tap_valid.py)
#           multifieldqa_en_val (50 cau tach tu ket qua test, tach_mfqa.py)
#   TEST  : qasper, narrativeqa (LongBench test), multifieldqa_en_test (100 cau con lai)
#   PHUONG PHAP: All-KV, Squeezed (threshold), ClusterKV (budget),
#                Squeezed+JKV, ClusterKV+JKV tren luoi BETAS x GAMMAS
#
# Chay lai bao nhieu lan cung duoc: cai gi DA DU thi bo qua, file DO DANG (bi ngat)
# thi xoa va chay lai. Cum (Clusters-*) cua JKV bi xoa ngay khi ca 2 luat da xong
# de khong day o dia.
#
# CACH DUNG (tren may chu Linux, thu muc value_aware_squeezed_project/scripts/):
#   KIEM_TRA=1 bash chay_toan_bo_valid_test.sh      # chi dem con thieu bao nhieu, KHONG chay
#   tmux new -s jkv                                  # nen chay trong tmux (dong SSH khong chet)
#   bash chay_toan_bo_valid_test.sh 2>&1 | tee chay_toan_bo.log
#   # chi 1 giai doan:  GIAI_DOAN="tong_hop" bash chay_toan_bo_valid_test.sh
#   # doi luoi:         BETAS="0.2 0.3 1.2" GAMMAS="0 0.1" bash chay_toan_bo_valid_test.sh
# ==============================================================================
set -u
set -o pipefail

SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # thu muc chua script nay
VA_PROJECT="${VA_PROJECT:-$(dirname "$SCRIPTS")}"
REPO="${REPO:-$(dirname "$VA_PROJECT")/SqueezedAttention-main}"
AWQ_MODEL_PATH="${AWQ_MODEL_PATH:-$HOME/models/LLaMA-2-7B-32K-AWQ}"
export REPO VA_PROJECT AWQ_MODEL_PATH
LB="${REPO}/LongBench"

SEEDS="${SEEDS:-0 1 2}"
PCTS="${PCTS:-0.7 0.8 0.9}"
MODES="${MODES:-threshold budget}"
BETAS="${BETAS:-0.05 0.1 0.2 0.3 0.5 1.0 1.2}"
GAMMAS="${GAMMAS:-0 0.1}"
TEST_DS="${TEST_DS-qasper narrativeqa multifieldqa_en}"   # TEST_DS="" -> chi chay VALID
VAL_DS="${VAL_DS:-qasper_val narrativeqa_val}"
ALL_DS="$VAL_DS $TEST_DS"
GIAI_DOAN="${GIAI_DOAN:-chuan_bi goc jkv tong_hop}"
KIEM_TRA="${KIEM_TRA:-0}"
OUT="${SCRIPTS}/ket_qua_cuoi"
PY="python3"
DU="$PY ${SCRIPTS}/kiem_tra_du.py --root ${LB}"

declare -A TAG=([threshold]=_thr [budget]=_bgt)
LOI=()
T0=$(date +%s)

co() { [[ " $GIAI_DOAN " == *" $1 "* ]]; }
gio() { local s=$(( $(date +%s) - T0 )); printf "%dh%02dm" $((s/3600)) $((s%3600/60)); }

tien_do() {
    read -r G J <<< "$($DU tong --tap "$ALL_DS" --muc "$PCTS" --seeds "$SEEDS" \
                         --betas "$BETAS" --gammas "$GAMMAS" --modes "$MODES")"
    local con=$((G + J)) xong=$((TONG0 - G - J))
    local msg="[tien do $(gio)] con thieu ${con} lan pred (goc ${G}, JKV ${J}); da xong ${xong}/${TONG0}"
    if [ "$xong" -gt 0 ]; then
        local tb=$(( ($(date +%s) - T0) / xong ))
        msg="$msg | TB $((tb/60)) phut/lan -> con khoang $(( con * tb / 3600 )) gio"
    fi
    echo "$msg"
}

echo "=================================================================="
echo " CHAY TOAN BO VALID + TEST   ($(date '+%F %T'))"
echo "   seeds=$SEEDS | muc nen=$PCTS | luat=$MODES"
echo "   beta=$BETAS | gamma=$GAMMAS"
echo "   valid=$VAL_DS (+ multifieldqa_en_val tach)  test=$TEST_DS"
echo "   giai doan: $GIAI_DOAN   KIEM_TRA=$KIEM_TRA"
echo "=================================================================="
cd "$SCRIPTS" || exit 1
for f in kiem_tra_du.py tao_tap_valid.py tach_mfqa.py phan_tich_heldout.py \
         xuat_ket_qua.py kiem_tra_cau_hinh_chung.py run_joint_kvcluster.sh; do
    [ -f "$f" ] || { echo "!! thieu $SCRIPTS/$f"; exit 1; }
done
df -h "$REPO" | tail -1 | awk '{print "   o dia con trong: "$4}'
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader

# ------------------------------------------------------------------ 0. chuan bi
if co chuan_bi; then
    echo -e "\n######## 0. CHUAN BI tap valid + va code ########"
    if [ "$KIEM_TRA" = "1" ]; then
        $PY tao_tap_valid.py --kiem_tra
    else
        $PY tao_tap_valid.py || { echo "!! tao_tap_valid.py loi"; exit 1; }
    fi
    # eval.py: file *_test (MFQA tach) cung phai cham bang metric tap goc
    $PY - "$LB/eval.py" "$KIEM_TRA" <<'PY'
import sys
p, kt = sys.argv[1], sys.argv[2] == "1"
s = open(p, encoding="utf-8").read()
cu = 'dataset2metric[dataset[:-4] if dataset.endswith("_val") else dataset]'
moi = 'dataset2metric[dataset.rsplit("_", 1)[0] if dataset.endswith(("_val", "_test")) else dataset]'
if moi in s:
    print("  = eval.py: da xu ly *_val/*_test")
elif cu in s:
    print("  + eval.py: them *_test" + (" (KIEM_TRA: chua ghi)" if kt else ""))
    if not kt:
        open(p, "w", encoding="utf-8").write(s.replace(cu, moi))
else:
    print("  !! eval.py chua duoc va *_val (chay tao_tap_valid.py truoc)")
PY
    if [ "$KIEM_TRA" != "1" ]; then
        for d in $VAL_DS; do
            f="$LB/data_val/$d.jsonl"
            n=$(grep -c . "$f" 2>/dev/null || echo 0)
            echo "   $d: $n cau"
            [ "$n" -ge 100 ] || { echo "!! $f qua it cau ($n) -> dung lai"; exit 1; }
        done
        $PY - "$LB/data_val" <<'PY'
import json, sys, os
for t in ("qasper_val", "narrativeqa_val"):
    f = os.path.join(sys.argv[1], t + ".jsonl")
    r = json.loads(open(f, encoding="utf-8").readline())
    print(f"   [{t}] cau 1: {r['input'][:80]!r} -> {r['answers'][:2]} | context {len(r['context'].split())} tu")
PY
    fi
fi

read -r G0 J0 <<< "$($DU tong --tap "$ALL_DS" --muc "$PCTS" --seeds "$SEEDS" \
                      --betas "$BETAS" --gammas "$GAMMAS" --modes "$MODES" 2>/dev/null || echo "0 0")"
TONG0=$((G0 + J0))
echo -e "\n>>> Con thieu: ${G0} lan pred phuong phap goc + ${J0} lan pred JKV = ${TONG0} lan"
echo "    (moi lan = 1 tap x 1 muc nen x 1 seed x 1 luat; tap 150-200 cau)"
if [ "$KIEM_TRA" = "1" ]; then
    for S in $SEEDS; do for M in $MODES; do
        x=$($DU thieu --thu_muc "pred_s${S}${TAG[$M]}" --tap "$ALL_DS" --muc "$PCTS")
        [ -n "$x" ] && echo "    goc  s$S $M: $x"
        for B in $BETAS; do for G in $GAMMAS; do
            x=$($DU thieu --thu_muc "pred_E1_b${B}_g${G}_s${S}${TAG[$M]}" --tap "$ALL_DS" --muc "$PCTS")
            [ -n "$x" ] && echo "    JKV  s$S $M b=$B g=$G: $x"
        done; done
    done; done
    echo -e "\n(KIEM_TRA=1: khong chay gi.)"
    exit 0
fi

# ------------------------------------------------------------------ 1. goc
if co goc; then
    echo -e "\n######## 1. PHUONG PHAP GOC: All-KV, Squeezed (thr), ClusterKV (bgt) ########"
    for S in $SEEDS; do
        for M in $MODES; do
            CAN=$($DU thieu --don --thu_muc "pred_s${S}${TAG[$M]}" --tap "$ALL_DS" --muc "$PCTS")
            BL=$($DU baseline --don --tap "$ALL_DS")
            for d in $BL; do [[ " $CAN " == *" $d "* ]] || CAN="$CAN $d"; done
            CAN=$(echo $CAN)
            [ -z "$CAN" ] && { echo "= goc s$S $M: du"; continue; }
            for d in $CAN; do
                $DU cum --don --dir "${REPO}/Clusters-GPU-s${S}/${d}" --tap "$d" >/dev/null
            done
            echo -e "\n>>> [$(gio)] goc seed=$S luat=$M tap: $CAN"
            ( cd "$REPO" && SQ_SEED=$S SQ_SELECT_MODE=$M SQ_PERCENTILES="$PCTS" \
                SQ_DATASETS="$CAN" SQ_DEBUG=1 bash "${REPO}/run_offline_clustering.sh" ) \
                || LOI+=("goc s$S $M ($CAN)")
        done
        # cum goc cua tap VAL: xoa khi ca 2 luat da du (cum TEST giu nguyen)
        for d in $VAL_DS; do
            ok=1
            for M in $MODES; do
                [ -z "$($DU thieu --thu_muc "pred_s${S}${TAG[$M]}" --tap "$d" --muc "$PCTS")" ] || ok=0
            done
            [ "$ok" = 1 ] && [ -d "${REPO}/Clusters-GPU-s${S}/${d}" ] && \
                { rm -rf "${REPO}/Clusters-GPU-s${S}/${d}"; echo "   [RM] Clusters-GPU-s${S}/${d}"; }
        done
        tien_do
    done
fi

# ------------------------------------------------------------------ 2. JKV
if co jkv; then
    echo -e "\n######## 2. JKV: Squeezed+JKV, ClusterKV+JKV tren luoi beta x gamma ########"
    for S in $SEEDS; do
      for B in $BETAS; do
        for G in $GAMMAS; do
          CDIR="${REPO}/Clusters-E1-b${B}-g${G}-s${S}"
          for d in $ALL_DS; do
            CAN_M=""
            for M in $MODES; do
                x=$($DU thieu --don --thu_muc "pred_E1_b${B}_g${G}_s${S}${TAG[$M]}" --tap "$d" --muc "$PCTS")
                [ -n "$x" ] && CAN_M="$CAN_M $M"
            done
            [ -z "$CAN_M" ] && continue
            # log [SQDBG] chi can cho tap test (bang sparsity); tap val tat de do day dia
            DBG=1; [[ "$d" == *_val ]] && DBG=0
            for M in $CAN_M; do
                echo -e "\n>>> [$(gio)] JKV seed=$S beta=$B gamma=$G tap=$d luat=$M"
                ( SQ_SEED=$S SQ_SELECT_MODE=$M BETA=$B GAMMA=$G SQ_PERCENTILES="$PCTS" \
                    SQ_DATASETS="$d" SQ_DEBUG=$DBG bash "${SCRIPTS}/run_joint_kvcluster.sh" ) \
                    || LOI+=("JKV s$S b$B g$G $d $M")
            done
            ok=1
            for M in $MODES; do
                [ -z "$($DU thieu --thu_muc "pred_E1_b${B}_g${G}_s${S}${TAG[$M]}" --tap "$d" --muc "$PCTS")" ] || ok=0
            done
            if [ "$ok" = 1 ] && [ -d "${CDIR}/${d}" ]; then
                rm -rf "${CDIR:?}/${d}"; echo "   [RM] $(basename "$CDIR")/${d}"
            fi
          done
          rmdir "$CDIR" 2>/dev/null || true
        done
        tien_do
      done
    done
fi

# ------------------------------------------------------------------ 3. tong hop
if co tong_hop; then
    echo -e "\n######## 3. TONG HOP: tach MFQA, chon tren VALID, bao cao TEST, bootstrap ########"
    mkdir -p "$OUT"
    $PY tach_mfqa.py --pred_root "$LB" || LOI+=("tach_mfqa")
    VAL3="qasper_val narrativeqa_val multifieldqa_en_val"
    TEST3="qasper narrativeqa multifieldqa_en_test"
    # (a) chon (beta,gamma) tren VALID -> bao cao TEST + paired bootstrap 1000 lan
    $PY phan_tich_heldout.py --pred_root "$LB" --che_do val --seeds "$SEEDS" \
        --muc "$PCTS" --n_boot 1000 --out "$OUT/heldout_val" || LOI+=("heldout val")
    # (b) doi chieu: chia 30/70 ngay tren tap test (20 cach chia)
    $PY phan_tich_heldout.py --pred_root "$LB" --che_do chia --seeds "$SEEDS" \
        --muc "$PCTS" --n_boot 1000 --n_splits 20 --out "$OUT/heldout_chia" || LOI+=("heldout chia")
    MUC_INT=$(for p in $PCTS; do printf "%d " "$($PY -c "print(round($p*100))")"; done)
    # (c) bang xep hang cau hinh tren VALID va tren TEST (xlsx)
    $PY kiem_tra_cau_hinh_chung.py --pred_root "$LB" --datasets "$VAL3" --muc_nen "$MUC_INT" \
        --seeds "$SEEDS" --out "$OUT/xep_hang_VALID.xlsx" || LOI+=("xep hang valid")
    $PY kiem_tra_cau_hinh_chung.py --pred_root "$LB" --datasets "$TEST3" --muc_nen "$MUC_INT" \
        --seeds "$SEEDS" --out "$OUT/xep_hang_TEST.xlsx" || LOI+=("xep hang test")
    $PY kiem_tra_cau_hinh_chung.py --pred_root "$LB" --datasets "$TEST_DS" --muc_nen "$MUC_INT" \
        --seeds "$SEEDS" --out "$OUT/xep_hang_TEST_day_du.xlsx" || LOI+=("xep hang test day du")
    # (d) bang so tung lan chay (gop seed) cho ca 6 tap
    $PY xuat_ket_qua.py --pred_root "$LB" --datasets "$VAL3 $TEST3 $TEST_DS" --seeds "$SEEDS" \
        --muc_nen "$MUC_INT" --gop_seed --out "$OUT/ket_qua_valid_test.xlsx" || LOI+=("xuat_ket_qua")
    # (e) sparsity thuc te (bang tab:sparsity) tu log [SQDBG]
    [ -f kept_pc.py ] && { $PY kept_pc.py --logs "$LB/logs_E1" --out "$OUT/kept_pc" || LOI+=("kept_pc"); }
    echo -e "\nKet qua o: $OUT"
    ls -la "$OUT"
fi

echo -e "\n=================================================================="
echo " XONG sau $(gio)  ($(date '+%F %T'))"
if [ ${#LOI[@]} -gt 0 ]; then
    echo " !! ${#LOI[@]} buoc LOI (chay lai lenh nay de thu lai cac buoc do):"
    printf '    - %s\n' "${LOI[@]}"
else
    echo " Khong co buoc nao loi."
fi
$DU tong --tap "$ALL_DS" --muc "$PCTS" --seeds "$SEEDS" --betas "$BETAS" --gammas "$GAMMAS" \
    --modes "$MODES" | awk '{print " Con thieu: "$1" lan goc, "$2" lan JKV (0 0 = du het)"}'
echo "=================================================================="
