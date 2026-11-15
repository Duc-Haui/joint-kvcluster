#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tach_mfqa.py — Phuong an B: MultiFieldQA-en khong co tap validation -> tach TEST thanh dev / test.

Moi cau hoi duoc nen va tra loi DOC LAP, nen khong can chay lai: tu moi file ket qua
  <pred_root>/pred_*/<cau hinh>/multifieldqa_en.jsonl
script ghi them hai file canh no:
  multifieldqa_en_val.jsonl   (cac cau DEV  — dung de CHON β, γ, cung qasper_val, narrativeqa_val)
  multifieldqa_en_test.jsonl  (cac cau TEST — dung de BAO CAO)
Cach chia co dinh (seed), luu tai <pred_root>/data_val/multifieldqa_en_split.json va dung lai
o moi lan chay -> moi cau hinh, moi seed, moi muc nen dung CUNG cach chia.

Chay lai bat cu luc nao (vd sau khi chay them cau hinh tren multifieldqa_en): file da tach
va con moi hon file goc thi bo qua; file goc chua du cau (dang chay do) thi bo qua.

CACH DUNG:
  python3 tach_mfqa.py                 # 50 dev / 100 test, seed 2024
  python3 tach_mfqa.py --kiem_tra      # chi in, khong ghi
  python3 tach_mfqa.py --xoa           # xoa moi file *_val / *_test da tach
"""
import argparse
import glob
import json
import os
import random

_ROOT_MAC_DINH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "../../SqueezedAttention-main/LongBench")
DS = "multifieldqa_en"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred_root", default=os.environ.get("ROOT", _ROOT_MAC_DINH))
    ap.add_argument("--n_dev", type=int, default=50)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--kiem_tra", action="store_true")
    ap.add_argument("--xoa", action="store_true")
    a = ap.parse_args()
    root = os.path.abspath(a.pred_root)
    print(f"[cau hinh] pred_root = {root}")
    files = sorted(glob.glob(os.path.join(root, "pred_*", "*", f"{DS}.jsonl")))
    if a.xoa:
        n = 0
        for f in files:
            for duoi in ("_val", "_test"):
                g = f.replace(f"{DS}.jsonl", f"{DS}{duoi}.jsonl")
                if os.path.exists(g):
                    os.remove(g)
                    n += 1
        print(f"da xoa {n} file")
        return
    if not files:
        raise SystemExit(f"!! khong thay pred_*/*/{DS}.jsonl trong {root}")

    def doc(f):
        with open(f, encoding="utf-8") as fh:
            return [x for x in fh if x.strip()]

    from collections import Counter
    dem = Counter(len(doc(f)) for f in files)
    n_cau = dem.most_common(1)[0][0]           # so cau pho bien nhat = file chay du
    if len(dem) > 1:
        print(f"[canh bao] so cau khac nhau giua cac file: {dict(dem)} -> lay {n_cau}")
    p_split = os.path.join(root, "data_val", f"{DS}_split.json")
    if os.path.exists(p_split):
        sp = json.load(open(p_split, encoding="utf-8"))
        if sp["n_cau"] != n_cau:
            raise SystemExit(f"!! {p_split} chia cho {sp['n_cau']} cau, nhung file co {n_cau} cau")
        print(f"[chia] dung lai {p_split}: {len(sp['dev'])} dev / {len(sp['test'])} test")
    else:
        idx = list(range(n_cau))
        random.Random(a.seed).shuffle(idx)
        sp = dict(n_cau=n_cau, seed=a.seed, dev=sorted(idx[:a.n_dev]), test=sorted(idx[a.n_dev:]))
        print(f"[chia] moi: {len(sp['dev'])} dev / {len(sp['test'])} test (seed {a.seed})")
        if not a.kiem_tra:
            os.makedirs(os.path.dirname(p_split), exist_ok=True)
            json.dump(sp, open(p_split, "w", encoding="utf-8"), indent=1)

    ghi = bo_thieu = bo_moi = 0
    for f in files:
        dong = doc(f)
        if len(dong) != n_cau:
            bo_thieu += 1
            continue
        for duoi, I in (("_val", sp["dev"]), ("_test", sp["test"])):
            g = f.replace(f"{DS}.jsonl", f"{DS}{duoi}.jsonl")
            if os.path.exists(g) and os.path.getmtime(g) >= os.path.getmtime(f):
                bo_moi += 1
                continue
            if not a.kiem_tra:
                with open(g, "w", encoding="utf-8") as fh:
                    fh.writelines(dong[i] if dong[i].endswith("\n") else dong[i] + "\n" for i in I)
            ghi += 1
    print(f"{'se ghi' if a.kiem_tra else 'da ghi'} {ghi} file | bo qua {bo_moi} file da co | "
          f"bo qua {bo_thieu} file goc chua du {n_cau} cau")
    print(f"\nChon tham so : --datasets \"qasper_val narrativeqa_val {DS}_val\"")
    print(f"Bao cao      : --datasets \"qasper narrativeqa {DS}_test\"")


if __name__ == "__main__":
    main()
