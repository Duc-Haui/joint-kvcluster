#!/usr/bin/env bash
# chay_phuong_sai.sh — gom cum lai (chi prefill + k-means, KHONG sinh cau tra loi)
# de luu phuong sai Value trong cum, roi so sanh beta=0 / 0.3 / 1.2.
#
# Dat file nay vao: value_aware_squeezed_project/scripts/
# Chay:            bash chay_phuong_sai.sh
# Bien tuy chon:   LIMIT=20 DATASETS="qasper narrativeqa multifieldqa_en" BETAS="0 0.3 1.2"
set -euo pipefail

# ====== DUONG DAN MAY B (giong run_offline_clustering.sh) ======
SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # thu muc chua script nay
VA_PROJECT="${VA_PROJECT:-$(dirname "$SCRIPTS")}"
REPO="${REPO:-$(dirname "$VA_PROJECT")/SqueezedAttention-main}"
AWQ="${AWQ:-${AWQ_MODEL_PATH:-$HOME/models/LLaMA-2-7B-32K-AWQ}}"
# ================================================================

LIMIT="${LIMIT:-20}"
DATASETS="${DATASETS:-qasper narrativeqa multifieldqa_en}"
BETAS="${BETAS:-0 0.3 1.2}"
SEED="${SEED:-0}"
PY="${PY:-python}"
OUT="$REPO/Clusters-VAR"

[ -f "$REPO/offline_clustering.py" ] || { echo "!! Khong thay $REPO/offline_clustering.py"; exit 1; }
[ -d "$VA_PROJECT/value_aware" ] || { echo "!! Khong thay $VA_PROJECT/value_aware"; exit 1; }
[ -d "$AWQ" ] || { echo "!! Khong thay thu muc AWQ: $AWQ"; exit 1; }

# Moi truong GIONG HET run_offline_clustering.sh (custom transformers truoc)
export PYTHONPATH="${VA_PROJECT}:${REPO}/transformers/src:${REPO}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NVIDIA_LIB=$($PY -c "import os, site; print(os.path.join(site.getsitepackages()[0], 'nvidia', 'cuda_runtime', 'lib'))")
export LD_LIBRARY_PATH="$NVIDIA_LIB:${CONDA_PREFIX:-}/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export SQ_SEED="$SEED"

cd "$REPO"

# 1) Them 1 dong luu vvar vao offline_clustering.py (chi them 1 lan, co sao luu)
$PY - <<'EOF'
import re, shutil
p = "offline_clustering.py"
s = open(p, encoding="utf-8").read()
if "vvar_{dataidx}_" in s:
    print("[patch] da co dong luu vvar, bo qua")
else:
    m = re.search(r"\n([ \t]*)torch\.save\(nvar_dict, [^\n]*\n", s)
    if not m:
        raise SystemExit("!! Khong tim thay dong torch.save(nvar_dict, ...)")
    ind = m.group(1)
    them = (f"{ind}torch.save({{'vvar': vvar_dict, 'labels': lbl_dict}}, "
            "f'{args.output_path}/vvar_{dataidx}_{num_centroids}.pt')\n")
    shutil.copy(p, p + ".bak_truoc_vvar")
    s = s[:m.end()] + them + s[m.end():]
    open(p, "w", encoding="utf-8").write(s)
    print("[patch] da them dong luu vvar (sao luu: offline_clustering.py.bak_truoc_vvar)")
EOF

# 2) Gom cum vao thu muc MOI (khong dung den cum dang dung cho thi nghiem)
for ds in $DATASETS; do
  for b in $BETAS; do
    d="$OUT/${ds}_b${b}"
    if ls "$d"/vvar_*_*.pt >/dev/null 2>&1 && [ "$(ls "$d"/vvar_*_*.pt | wc -l)" -ge "$LIMIT" ]; then
      echo "[bo qua] $d da du $LIMIT file"; continue
    fi
    mkdir -p "$d"
    echo "=== $ds beta=$b -> $d"
    $PY offline_clustering.py LLaMA-2-7B-32K \
        --dataset "$ds" --percent_clusters 5 --observation_window 100 \
        --load_awq --awq_model_path "$AWQ" --seed "$SEED" --limit "$LIMIT" \
        --value_aware --alpha 1.0 --beta "$b" --gamma 0 \
        --output_path "$d" > "$d/log.txt" 2>&1 || {
          echo "!! LOI khi gom cum $ds beta=$b. 30 dong cuoi cua $d/log.txt:"
          tail -n 30 "$d/log.txt"; exit 1; }
    # Xoa ngay cac file cum khong dung cho phan tich (chi giu vvar_*.pt va log)
    find "$d" -maxdepth 1 -type f -name "*.pt" ! -name "vvar_*" -delete
    echo "   xong: $(ls "$d"/vvar_*_*.pt 2>/dev/null | wc -l) file vvar | $(du -sh "$d" | cut -f1)"
  done
done

# 3) So sanh (CPU)
for ds in $DATASETS; do
  ss=""
  for b in $BETAS; do [ "$b" = "0" ] || ss="$ss $OUT/${ds}_b${b}"; done
  $PY "$SCRIPTS/do_phuong_sai_cum.py" --goc "$OUT/${ds}_b0" --so_sanh $ss \
      --out "$SCRIPTS/phuong_sai_${ds}.csv"
done
echo "XONG. Ket qua: $SCRIPTS/phuong_sai_*.csv"

# XOA_HET=1 -> xoa toan bo Clusters-VAR sau khi da co CSV
if [ "${XOA_HET:-0}" = "1" ]; then
  ok=1
  for ds in $DATASETS; do [ -s "$SCRIPTS/phuong_sai_${ds}.csv" ] || ok=0; done
  if [ "$ok" = "1" ]; then
    echo "Xoa $OUT ($(du -sh "$OUT" | cut -f1))"; rm -rf "$OUT"
  else
    echo "!! Thieu CSV, KHONG xoa $OUT"
  fi
else
  echo "Clusters-VAR con lai: $(du -sh "$OUT" | cut -f1). Xoa het: rm -rf $OUT"
fi
