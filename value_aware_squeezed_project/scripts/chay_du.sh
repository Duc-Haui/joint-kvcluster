#!/bin/bash
# Chay DU cac thu con thieu. Moi lenh co [SKIP] nen chay lai duoc, khong lam lai tu dau.
# Cach dung:   bash chay_du.sh test      # 2 cau/lenh, kiem tra pipeline, do thoi gian
#              bash chay_du.sh A         # (A) sparsity p=70,80,90, 40 cau, 1 seed
#              bash chay_du.sh B         # (B) luat budget tren validation, 3 seed, DAY DU cau
# Cac bien/duong dan lay tu run_repeat.sh cua ban (SQ_LIMIT, SQ_DEBUG, SEEDS, FULL_PCTS, ONLY, DATASETS, BETAS, GAMMAS).
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

# $1 dataset  $2 beta  $3 gamma  $4 ONLY(sq|e1)  $5 muc nen  $6 so cau (0 = day du)
thr() { SQ_SELECT_MODE=threshold SQ_LIMIT="$6" SQ_DEBUG=1 SEEDS="0" FULL_PCTS="$5" \
        ONLY="$4" DATASETS="$1" BETAS="$2" GAMMAS="$3" bash run_repeat.sh; }

A() {  # $1 = so cau (40 that, 2 thu)
  L="$1"
  # --- Squeezed goc (beta=0,gamma=0): ca 3 muc nen cho moi dataset ---
  for DS in qasper narrativeqa multifieldqa_en; do thr "$DS" 0 0 sq "0.7 0.8 0.9" "$L"; done
  # --- Squeezed+JKV: DUNG (beta,gamma) da chon cho tung o cua Table I (best-of-grid) ---
  thr qasper          1   0   e1 "0.7 0.9" "$L"
  thr qasper          0.3 0   e1 "0.8"     "$L"
  thr narrativeqa     1.2 0.1 e1 "0.7"     "$L"
  thr narrativeqa     1   0   e1 "0.8 0.9" "$L"
  thr multifieldqa_en 0.3 0   e1 "0.7 0.8" "$L"
  thr multifieldqa_en 0.2 0.1 e1 "0.9"     "$L"
}

B() {  # luat budget, validation, E1 (gamma=0) + baseline ClusterKV, 3 seed
  SQ_SELECT_MODE=budget ONLY=both SEEDS="0 1 2" DATASETS="qasper_val"      BETAS="0.3" GAMMAS="0" SQ_LIMIT=0 bash run_repeat.sh
  SQ_SELECT_MODE=budget ONLY=both SEEDS="0 1 2" DATASETS="narrativeqa_val" BETAS="1.2" GAMMAS="0" SQ_LIMIT=0 bash run_repeat.sh
}

case "$1" in
  test) A 2 ;;
  A)    A 40 ;;
  B)    B ;;
  *)    echo "dung: bash chay_du.sh test|A|B"; exit 1 ;;
esac
