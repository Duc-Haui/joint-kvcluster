#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
do_phuong_sai_cum.py — kiem tra gia thuyet goc cua METHODOLOGY.md:
"token trong cung mot cum chi-Key co Value rat khac nhau".

CHI CHAY CPU. Doc cac file vvar_<idx>_<K>.pt do offline_clustering.py luu
(xem HUONG DAN ben duoi), khong chay mo hinh.

Moi file: {"vvar": {layer: (1,H,K)}, "labels": {layer: (1,H,N)}}
  vvar = trung binh ||v - mu_cum||^2 trong cum (Value THO, chua chuan hoa)

Chi so cho moi (layer, head):
  W = sum_k n_k * vvar_k / N   (phuong sai Value trong cum, co trong so)
So sanh W cua beta>0 voi beta=0 (Key-only) TREN CUNG TAI LIEU:
  ti_le = W(beta) / W(0)   < 1  -> cum dong nhat ve Value hon

Luu y khi viet bai: W giam gan nhu chac chan, vi K-means da toi uu mot
phan tren V. Ket qua chi xac nhan co che co chay, KHONG chung minh F1 tot hon.

CACH DUNG
  python3 do_phuong_sai_cum.py \
     --goc  Clusters-VAR/qasper_b0 \
     --so_sanh Clusters-VAR/qasper_b0.3 Clusters-VAR/qasper_b1.2 \
     --out  phuong_sai_qasper.csv
"""
import argparse, glob, os, re, csv
import torch


def doc(thu_muc):
    ra = {}
    for p in sorted(glob.glob(os.path.join(thu_muc, "vvar_*_*.pt"))):
        m = re.search(r"vvar_(\d+)_(\d+)\.pt$", p)
        if m:
            ra[int(m.group(1))] = torch.load(p, map_location="cpu")
    if not ra:
        raise SystemExit(f"!! Khong thay vvar_*.pt trong {thu_muc}")
    return ra


def W_theo_head(d):
    """-> {layer: tensor (H,)} phuong sai trong cum co trong so kich thuoc."""
    ra = {}
    for l, v in d["vvar"].items():
        v = v.float().reshape(v.shape[-2], v.shape[-1])            # (H,K)
        lb = d["labels"][l].long().reshape(v.shape[0], -1)          # (H,N)
        n = torch.zeros_like(v).scatter_add_(1, lb, torch.ones_like(lb, dtype=v.dtype))
        ra[l] = (n * v).sum(1) / n.sum(1).clamp(min=1)
    return ra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--goc", required=True, help="thu muc beta=0")
    ap.add_argument("--so_sanh", nargs="+", required=True)
    ap.add_argument("--out", default="phuong_sai_cum.csv")
    a = ap.parse_args()

    goc = doc(a.goc)
    hang = []
    for td in a.so_sanh:
        ss = doc(td)
        chung = sorted(set(goc) & set(ss))
        if not chung:
            print(f"!! {td}: khong co tai lieu chung voi {a.goc}")
            continue
        tl_all, giam = [], 0
        tong = 0
        for i in chung:
            wg, ws = W_theo_head(goc[i]), W_theo_head(ss[i])
            for l in wg:
                r = ws[l] / wg[l].clamp(min=1e-12)
                tl_all.append(r)
                giam += int((r < 1).sum()); tong += r.numel()
        tl = torch.cat(tl_all)
        hang.append({
            "thu_muc": td, "so_tai_lieu": len(chung),
            "ti_le_TB": round(tl.mean().item(), 4),
            "ti_le_trung_vi": round(tl.median().item(), 4),
            "phan_tram_head_giam": round(100 * giam / tong, 1),
        })
        print(f"{td}: {len(chung)} tai lieu | W(beta)/W(0) TB {tl.mean():.3f}"
              f" trung vi {tl.median():.3f} | {100*giam/tong:.1f}% (layer,head) giam")
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(hang[0]))
        w.writeheader(); w.writerows(hang)
    print("Da ghi:", os.path.abspath(a.out))


if __name__ == "__main__":
    main()
