#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
phan_tich_heldout.py -- chon (beta, gamma) tren tap HELD-OUT va kiem dinh
PAIRED BOOTSTRAP giua JKV va phuong phap Key-only. KHONG CAN GPU.

Doc thang cac file du doan da co:
    <pred_root>/pred_s<seed>_thr/LLaMA-2-7B-32K_PC5_PERC<p>/<tap>.jsonl          (Squeezed)
    <pred_root>/pred_s<seed>_bgt/...                                             (ClusterKV)
    <pred_root>/pred_E1_b<beta>_g<gamma>_s<seed>_thr/...                         (Squeezed+JKV)
    <pred_root>/pred_E1_b<beta>_g<gamma>_s<seed>_bgt/...                         (ClusterKV+JKV)

Cham QA F1 TUNG CAU dung dung qa_f1_score cua LongBench (max qua cac dap an),
lay trung binh qua cac seed cho tung cau.

QUY TRINH (cho moi rule x tap x muc nen)
  1. Chia cac cau hoi ngau nhien (co dinh --split_seed) thanh DEV (--dev_frac,
     mac dinh 30%) va TEST (70%).
  2. Chon (beta, gamma) co F1 trung binh cao nhat tren DEV, chi trong cac cau
     hinh co du --seeds seed va du so cau.  Mac dinh chi xet beta > 0 (JKV);
     them --cho_beta0 neu muon xet ca cau hinh chi-boost (beta = 0, gamma > 0).
  3. Bao cao F1 tren TEST cua cau hinh da chon va cua Key-only
     (mean +- std qua seed).
  4. Paired bootstrap (--n_boot lan) tren TEST: lay mau lai cac cau hoi, tinh
     hieu F1 trung binh JKV - Key-only -> khoang tin cay 95% va p-value
     (mot phia: ti le mau co hieu <= 0).
  5. Lap lai buoc 1-3 voi --n_splits cach chia khac nhau de xem ket qua co
     on dinh khong (bao nhieu lan JKV thang tren TEST).

CHE DO --che_do val (tap VALIDATION rieng, tao bang tao_tap_valid.py + tach_mfqa.py)
  Chon (beta, gamma) tren <tap>_val (qasper_val, narrativeqa_val, multifieldqa_en_val),
  bao cao + bootstrap tren tap TEST: qasper, narrativeqa (LongBench test day du) va
  multifieldqa_en_test (100 cau con lai). Khong chia lai, khong co --n_splits.
    python3 phan_tich_heldout.py --che_do val --out heldout_val

CACH DUNG (chay trong thu muc scripts)
  python3 phan_tich_heldout.py --pred_root ../../SqueezedAttention-main/LongBench
  python3 phan_tich_heldout.py --pred_root ... --rule thr --tap "narrativeqa" --muc "0.8"
Ket qua: <out>.csv (bang chinh), <out>_splits.csv (do on dinh), <out>.tex (dong LaTeX).
"""
import argparse
import csv
import glob
import json
import os
import random
import re
import statistics as st
import string
import sys
from collections import Counter

MODEL = "LLaMA-2-7B-32K"


# ------------------------------------------------- QA F1 (giong LongBench)
def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def f1_tokens(pred_toks, gt_toks):
    common = Counter(pred_toks) & Counter(gt_toks)
    same = sum(common.values())
    if same == 0:
        return 0.0
    p = same / len(pred_toks)
    r = same / len(gt_toks)
    return 2 * p * r / (p + r)


def qa_f1(pred, gt):
    return f1_tokens(normalize_answer(pred).split(), normalize_answer(gt).split())


def cham_file(f):
    """-> list F1 (0..100) tung cau, theo thu tu dong."""
    ra = []
    with open(f, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            pred = d["pred"]
            best = 0.0
            for gt in d["answers"]:
                best = max(best, qa_f1(pred, gt))
            ra.append(100.0 * best)
    return ra


# ------------------------------------------------- tim file
def file_pred(root, thu_muc, p, tap):
    return os.path.join(root, thu_muc, f"{MODEL}_PC5_PERC{p}", f"{tap}.jsonl")


def tim_cau_hinh(root, rule):
    """-> set (beta_str, gamma_str) co thu muc pred_E1_b*_g*_s*_<rule>."""
    ra = set()
    for d in glob.glob(os.path.join(root, f"pred_E1_b*_g*_s*_{rule}")):
        ten = os.path.basename(d)
        if "-lim" in ten or "placebo" in ten or "_vb" in ten:
            continue
        m = re.fullmatch(rf"pred_E1_b([0-9.]+)_g([0-9.]+)_s\d+_{rule}", ten)
        if m:
            ra.add((m.group(1), m.group(2)))
    return sorted(ra, key=lambda x: (float(x[0]), float(x[1])))


def nap(root, thu_muc_mau, seeds, p, tap, cache):
    """-> list[list F1] theo seed, hoac None neu thieu."""
    ra = []
    for s in seeds:
        f = file_pred(root, thu_muc_mau.format(s=s), p, tap)
        if f not in cache:
            cache[f] = cham_file(f) if os.path.exists(f) else None
        if cache[f] is None:
            return None
        ra.append(cache[f])
    n = {len(x) for x in ra}
    if len(n) != 1:
        return None
    return ra


def tb_seed(per_seed, idx):
    """F1 trung binh tren tap cau idx, cho tung seed -> (mean, std)."""
    vals = [st.mean([x[i] for i in idx]) for x in per_seed]
    sd = st.stdev(vals) if len(vals) > 1 else 0.0
    return st.mean(vals), sd


def tb_cau(per_seed):
    """F1 tung cau, lay trung binh qua seed."""
    return [st.mean(v) for v in zip(*per_seed)]


def bootstrap(a, b, idx, n_boot, rng):
    d = [a[i] - b[i] for i in idx]
    n = len(d)
    goc = st.mean(d)
    mau = []
    for _ in range(n_boot):
        mau.append(sum(d[rng.randrange(n)] for _ in range(n)) / n)
    mau.sort()
    lo = mau[int(0.025 * n_boot)]
    hi = mau[int(0.975 * n_boot) - 1]
    p_mot_phia = sum(1 for x in mau if x <= 0) / n_boot
    return goc, lo, hi, p_mot_phia


def chia(n, dev_frac, seed):
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    k = int(round(dev_frac * n))
    return sorted(idx[:k]), sorted(idx[k:])


TAP_TEST = {"multifieldqa_en": "multifieldqa_en_test"}


def mot_o_val(a, rule, tap, p, cfgs, seeds, cache, rng, TEN_RULE):
    """Chon tren <tap>_val, bao cao tren tap test. -> dict hang hoac None."""
    tv, tt = tap + "_val", TAP_TEST.get(tap, tap)
    goc_v = nap(a.pred_root, "pred_s{s}_" + rule, seeds, p, tv, cache)
    goc_t = nap(a.pred_root, "pred_s{s}_" + rule, seeds, p, tt, cache)
    if goc_v is None or goc_t is None:
        print(f"!! thieu Key-only ({'val' if goc_v is None else 'test'}): {rule} {tap} {p}")
        return None
    nv, nt = len(goc_v[0]), len(goc_t[0])
    ung = {}
    for b, g in cfgs:
        mau = f"pred_E1_b{b}_g{g}_s{{s}}_{rule}"
        xv = nap(a.pred_root, mau, seeds, p, tv, cache)
        xt = nap(a.pred_root, mau, seeds, p, tt, cache)
        if xv is not None and xt is not None and len(xv[0]) == nv and len(xt[0]) == nt:
            ung[(b, g)] = (xv, xt)
    if not ung:
        print(f"!! khong co cau hinh JKV du seed tren CA val va test: {rule} {tap} {p}")
        return None
    tv_idx, tt_idx = list(range(nv)), list(range(nt))
    best = max(ung, key=lambda c: tb_seed(ung[c][0], tv_idx)[0])
    vm, _ = tb_seed(ung[best][0], tv_idx)
    gvm, _ = tb_seed(goc_v, tv_idx)
    jm, js = tb_seed(ung[best][1], tt_idx)
    gm, gs = tb_seed(goc_t, tt_idx)
    d, lo, hi, pv = bootstrap(tb_cau(ung[best][1]), tb_cau(goc_t), tt_idx, a.n_boot, rng)
    best_all = max(ung, key=lambda c: tb_seed(ung[c][1], tt_idx)[0])
    bam, bas = tb_seed(ung[best_all][1], tt_idx)
    d2, lo2, hi2, pv2 = bootstrap(tb_cau(ung[best_all][1]), tb_cau(goc_t), tt_idx, a.n_boot, rng)
    print(f"{TEN_RULE[rule][1]:<14}{tap:<16}p={p}  chon(b={best[0]},g={best[1]}) "
          f"VAL {gvm:.2f}->{vm:.2f} | TEST {gm:.2f}->{jm:.2f} d={d:+.2f} "
          f"CI95[{lo:+.2f},{hi:+.2f}] p={pv:.3f} | best-of-grid test {bam:.2f} "
          f"(b={best_all[0]},g={best_all[1]})  [{len(ung)} cau hinh]")
    return {
        "rule": rule, "tap": tap, "p": p, "n_cau": nt,
        "n_dev": nv, "n_test": nt, "n_cau_hinh": len(ung),
        "chon_beta": best[0], "chon_gamma": best[1],
        "keyonly_val": round(gvm, 2), "jkv_val": round(vm, 2),
        "keyonly_test": round(gm, 2), "keyonly_test_sd": round(gs, 2),
        "jkv_test": round(jm, 2), "jkv_test_sd": round(js, 2),
        "delta_test": round(d, 2), "ci95_lo": round(lo, 2),
        "ci95_hi": round(hi, 2), "p_boot": round(pv, 4),
        "bestgrid_beta": best_all[0], "bestgrid_gamma": best_all[1],
        "bestgrid_full": round(bam, 2), "bestgrid_full_sd": round(bas, 2),
        "keyonly_full": round(gm, 2), "keyonly_full_sd": round(gs, 2),
        "bestgrid_delta": round(d2, 2), "bestgrid_ci_lo": round(lo2, 2),
        "bestgrid_ci_hi": round(hi2, 2), "bestgrid_p": round(pv2, 4),
        "splits_thang": "-", "splits_delta_tb": "-",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_root", default="../../SqueezedAttention-main/LongBench")
    ap.add_argument("--rule", default="thr bgt", help="'thr', 'bgt' hoac ca hai")
    ap.add_argument("--tap", default="narrativeqa qasper multifieldqa_en")
    ap.add_argument("--muc", default="0.7 0.8 0.9")
    ap.add_argument("--seeds", default="0 1 2")
    ap.add_argument("--dev_frac", type=float, default=0.3)
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--n_splits", type=int, default=20)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--cho_beta0", action="store_true",
                    help="xet ca cau hinh beta=0 (chi boost) khi chon")
    ap.add_argument("--out", default="heldout")
    ap.add_argument("--che_do", default="chia", choices=["chia", "val"],
                    help="chia = tach 30/70 tu tap test; val = chon tren *_val, bao cao tren test")
    a = ap.parse_args()

    seeds = a.seeds.split()
    cache = {}
    bang, bang_split, dong_tex = [], [], []
    rng = random.Random(12345)
    TEN_RULE = {"thr": ("Squeezed", "Squeezed+JKV"), "bgt": ("ClusterKV", "ClusterKV+JKV")}

    for rule in a.rule.split():
        cfgs = tim_cau_hinh(a.pred_root, rule)
        if not a.cho_beta0:
            cfgs = [c for c in cfgs if float(c[0]) > 0]
        print(f"\n=== rule={rule}: {len(cfgs)} cau hinh tim thay: "
              + ", ".join(f"({b},{g})" for b, g in cfgs))
        for tap in a.tap.split():
            for p in a.muc.split():
                if a.che_do == "val":
                    row = mot_o_val(a, rule, tap, p, cfgs, seeds, cache, rng, TEN_RULE)
                    if row is not None:
                        bang.append(row)
                        ten0 = TEN_RULE[rule][0]
                        dong_tex.append(
                            f"% {TEN_RULE[rule][1]} {tap} p={p}: chon ({row['chon_beta']},{row['chon_gamma']}) tren {tap}_val\n"
                            f"{ten0} & {int(round(float(p)*100))}\\% & {tap} & ${row['keyonly_test']:.2f}{{\\pm}}{row['keyonly_test_sd']:.2f}$ & "
                            f"${row['jkv_test']:.2f}{{\\pm}}{row['jkv_test_sd']:.2f}$ & ${row['delta_test']:+.2f}$ "
                            f"$[{row['ci95_lo']:+.2f},{row['ci95_hi']:+.2f}]$ & {row['p_boot']:.3f} \\\\")
                    continue
                goc = nap(a.pred_root, "pred_s{s}_" + rule, seeds, p, tap, cache)
                if goc is None:
                    print(f"!! thieu Key-only: {rule} {tap} {p}")
                    continue
                n = len(goc[0])
                ung = {}
                for b, g in cfgs:
                    x = nap(a.pred_root, f"pred_E1_b{b}_g{g}_s{{s}}_{rule}",
                            seeds, p, tap, cache)
                    if x is not None and len(x[0]) == n:
                        ung[(b, g)] = x
                if not ung:
                    print(f"!! khong co cau hinh JKV du seed: {rule} {tap} {p}")
                    continue

                # --- cach chia chinh
                dev, test = chia(n, a.dev_frac, a.split_seed)
                best = max(ung, key=lambda c: tb_seed(ung[c], dev)[0])
                jm, js = tb_seed(ung[best], test)
                gm, gs = tb_seed(goc, test)
                d, lo, hi, pv = bootstrap(tb_cau(ung[best]), tb_cau(goc), test,
                                          a.n_boot, rng)
                # --- best-of-grid tren TOAN BO (doi chieu voi Table II)
                tat_ca = list(range(n))
                best_all = max(ung, key=lambda c: tb_seed(ung[c], tat_ca)[0])
                bam, bas = tb_seed(ung[best_all], tat_ca)
                gam, gas = tb_seed(goc, tat_ca)
                d2, lo2, hi2, pv2 = bootstrap(tb_cau(ung[best_all]), tb_cau(goc),
                                              tat_ca, a.n_boot, rng)
                # --- nhieu cach chia
                thang, deltas = 0, []
                for k in range(a.n_splits):
                    dv, ts = chia(n, a.dev_frac, 1000 + k)
                    bk = max(ung, key=lambda c: tb_seed(ung[c], dv)[0])
                    dd = tb_seed(ung[bk], ts)[0] - tb_seed(goc, ts)[0]
                    deltas.append(dd)
                    thang += dd > 0
                ten0, ten1 = TEN_RULE[rule]
                row = {
                    "rule": rule, "tap": tap, "p": p, "n_cau": n,
                    "n_dev": len(dev), "n_test": len(test), "n_cau_hinh": len(ung),
                    "chon_beta": best[0], "chon_gamma": best[1],
                    "keyonly_test": round(gm, 2), "keyonly_test_sd": round(gs, 2),
                    "jkv_test": round(jm, 2), "jkv_test_sd": round(js, 2),
                    "delta_test": round(d, 2), "ci95_lo": round(lo, 2),
                    "ci95_hi": round(hi, 2), "p_boot": round(pv, 4),
                    "bestgrid_beta": best_all[0], "bestgrid_gamma": best_all[1],
                    "bestgrid_full": round(bam, 2), "bestgrid_full_sd": round(bas, 2),
                    "keyonly_full": round(gam, 2), "keyonly_full_sd": round(gas, 2),
                    "bestgrid_delta": round(d2, 2), "bestgrid_ci_lo": round(lo2, 2),
                    "bestgrid_ci_hi": round(hi2, 2), "bestgrid_p": round(pv2, 4),
                    "splits_thang": f"{thang}/{a.n_splits}",
                    "splits_delta_tb": round(st.mean(deltas), 2),
                }
                bang.append(row)
                for k, dd in enumerate(deltas):
                    bang_split.append({"rule": rule, "tap": tap, "p": p,
                                       "split": k, "delta_test": round(dd, 3)})
                print(f"{ten1:<14}{tap:<16}p={p}  chon(b={best[0]},g={best[1]})  "
                      f"TEST {gm:.2f}->{jm:.2f}  d={d:+.2f} CI95[{lo:+.2f},{hi:+.2f}] "
                      f"p={pv:.3f}  | splits thang {thang}/{a.n_splits}  "
                      f"| best-of-grid full {gam:.2f}->{bam:.2f} (b={best_all[0]},g={best_all[1]})")
                dong_tex.append(
                    f"% {ten1} {tap} p={p}: chon (beta,gamma)=({best[0]},{best[1]}) tren DEV\n"
                    f"{ten0} & {int(float(p)*100)}\\% & {tap} & ${gm:.2f}{{\\pm}}{gs:.2f}$ & "
                    f"${jm:.2f}{{\\pm}}{js:.2f}$ & ${d:+.2f}$ $[{lo:+.2f},{hi:+.2f}]$ & {pv:.3f} \\\\")

    if not bang:
        sys.exit("!! Khong co ket qua nao. Kiem tra --pred_root.")
    with open(a.out + ".csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(bang[0].keys()))
        w.writeheader()
        w.writerows(bang)
    with open(a.out + "_splits.csv", "w", newline="", encoding="utf-8") as f:
        if not bang_split:
            bang_split = [{"ghi_chu": "che_do val: khong chia lai"}]
        w = csv.DictWriter(f, fieldnames=list(bang_split[0].keys()))
        w.writeheader()
        w.writerows(bang_split)
    with open(a.out + ".tex", "w", encoding="utf-8") as f:
        f.write("\n".join(dong_tex) + "\n")

    # --- bang gon cho bai bao: hang = rule x p, cot = tap, o = delta [CI95]
    TAP = ["narrativeqa", "qasper", "multifieldqa_en"]
    idx = {(r["rule"], r["tap"], r["p"]): r for r in bang}
    gon = []
    for rule in a.rule.split():
        ten1 = TEN_RULE[rule][1]
        for p in a.muc.split():
            o = []
            for t in TAP:
                r = idx.get((rule, t, p))
                if r is None:
                    o.append("--")
                    continue
                cell = f"${r['delta_test']:+.2f}$ {{\\scriptsize$[{r['ci95_lo']:+.2f},{r['ci95_hi']:+.2f}]$}}"
                if r["ci95_lo"] > 0:
                    cell = f"$\\mathbf{{{r['delta_test']:+.2f}}}$ {{\\scriptsize$[{r['ci95_lo']:+.2f},{r['ci95_hi']:+.2f}]$}}"
                o.append(cell)
            gon.append(f"{ten1} & {int(round(float(p)*100))}\\% & " + " & ".join(o) + " \\\\")
        gon.append("\\midrule")
    if gon and gon[-1] == "\\midrule":
        gon.pop()
    with open(a.out + "_bang.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(gon) + "\n")
    print(f"\nDa ghi: {a.out}.csv, {a.out}_splits.csv, {a.out}.tex, {a.out}_bang.tex")


if __name__ == "__main__":
    main()
