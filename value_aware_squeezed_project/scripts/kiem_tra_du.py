#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kiem_tra_du.py — tro ly cho chay_toan_bo_valid_test.sh: biet file ket qua nao DA DU,
file nao DO DANG (bi ngat giua chung) va con thieu bao nhieu lan chay.

Vi sao can: run_joint_kvcluster.sh / run_offline_clustering.sh chi [SKIP] khi file .jsonl TON TAI,
khong kiem tra du cau. Mot lan chay bi ngat de lai file thieu cau -> lan sau bi bo qua
-> so lieu sai am tham. Script nay xoa file do dang TRUOC khi goi chung.

So cau mong doi:
  test : qasper 200, narrativeqa 200, multifieldqa_en 150 (LongBench)
  val  : dem dong cua <LongBench>/data_val/<tap>.jsonl

LENH CON
  thieu   --root LB --thu_muc pred_E1_b0.2_g0_s0_thr --tap "qasper qasper_val" --muc "0.7 0.8 0.9" [--don]
          -> in ra cac tap con thieu it nhat mot muc nen (cach nhau dau cach)
  baseline --root LB --tap "..."  [--don]   -> tap chua co All-KV day du
  cum     --dir <Clusters-GPU-s0/qasper_val> --root LB --tap qasper_val [--don]
          -> in "du" / "thieu"; --don: xoa bo cum do dang
  tong    --root LB --seeds .. --muc .. --betas .. --gammas .. --modes .. --tap ..
          -> dem so lan pred con thieu (de uoc luong thoi gian)
"""
import argparse
import glob
import os
import shutil
import sys

MODEL = "LLaMA-2-7B-32K"
N_TEST = {"qasper": 200, "narrativeqa": 200, "multifieldqa_en": 150}
TAG = {"threshold": "_thr", "budget": "_bgt"}


def so_dong(f):
    try:
        with open(f, encoding="utf-8") as fh:
            return sum(1 for x in fh if x.strip())
    except OSError:
        return -1


GIA_DINH_VAL = 200   # so cau mac dinh cua tao_tap_valid.py (--n 200)
_DON = False


def n_mong_doi(root, tap):
    if tap in N_TEST:
        return N_TEST[tap]
    f = os.path.join(root, "data_val", f"{tap}.jsonl")
    n = so_dong(f)
    if n <= 0 and not _DON:      # chi dem (KIEM_TRA): chua tao tap val -> gia dinh 200 cau
        return GIA_DINH_VAL
    if n <= 0:
        sys.exit(f"!! khong biet so cau cua '{tap}' (chua co {f}). Chay tao_tap_valid.py truoc.")
    return n


def file_pred(root, thu_muc, p, tap):
    return os.path.join(root, thu_muc, f"{MODEL}_PC5_PERC{p}", f"{tap}.jsonl")


def kiem_file(f, n, don):
    """-> True neu du. Xoa file do dang neu don."""
    k = so_dong(f)
    if k == n:
        return True
    if k >= 0 and don:
        os.remove(f)
        print(f"[DON] xoa file do dang ({k}/{n} cau): {f}", file=sys.stderr)
    return False


def thieu(root, thu_muc, taps, mucs, don):
    ra = []
    for t in taps:
        n = n_mong_doi(root, t)
        ok = [kiem_file(file_pred(root, thu_muc, p, t), n, don) for p in mucs]
        if not all(ok):
            ra.append(t)
    return ra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lenh", choices=["thieu", "baseline", "cum", "tong"])
    ap.add_argument("--root", required=True, help="thu muc LongBench/")
    ap.add_argument("--thu_muc", default="")
    ap.add_argument("--dir", default="")
    ap.add_argument("--tap", default="")
    ap.add_argument("--muc", default="0.7 0.8 0.9")
    ap.add_argument("--seeds", default="0 1 2")
    ap.add_argument("--betas", default="")
    ap.add_argument("--gammas", default="")
    ap.add_argument("--modes", default="threshold budget")
    ap.add_argument("--don", action="store_true")
    a = ap.parse_args()
    global _DON
    _DON = a.don
    root = os.path.abspath(a.root)
    taps, mucs = a.tap.split(), a.muc.split()

    if a.lenh == "thieu":
        print(" ".join(thieu(root, a.thu_muc, taps, mucs, a.don)))
    elif a.lenh == "baseline":
        ra = []
        for t in taps:
            f = os.path.join(root, "pred_baseline", f"{MODEL}_baseline", f"{t}.jsonl")
            if not kiem_file(f, n_mong_doi(root, t), a.don):
                ra.append(t)
        print(" ".join(ra))
    elif a.lenh == "cum":
        t = taps[0]
        n = n_mong_doi(root, t)
        k = len(glob.glob(os.path.join(a.dir, "centroids_labels_dict_*.pt")))
        if k >= n:
            print("du")
        else:
            if k > 0 and a.don and os.path.isdir(a.dir):
                shutil.rmtree(a.dir)
                print(f"[DON] xoa bo cum do dang ({k}/{n}): {a.dir}", file=sys.stderr)
            print("thieu")
    else:  # tong
        tong = goc = 0
        for s in a.seeds.split():
            for m in a.modes.split():
                tg = TAG[m]
                for t in taps:
                    n = n_mong_doi(root, t)
                    for p in mucs:
                        if so_dong(file_pred(root, f"pred_s{s}{tg}", p, t)) != n:
                            goc += 1
                        for b in a.betas.split():
                            for g in a.gammas.split():
                                f = file_pred(root, f"pred_E1_b{b}_g{g}_s{s}{tg}", p, t)
                                if so_dong(f) != n:
                                    tong += 1
        print(f"{goc} {tong}")


if __name__ == "__main__":
    main()
