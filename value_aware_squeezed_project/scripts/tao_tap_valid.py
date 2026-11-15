#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tao_tap_valid.py — Tao tap VALIDATION (Qasper, NarrativeQA) theo dinh dang LongBench va
noi vao pipeline duoi ten dataset moi "<ten>_val", KHONG dung cham toi ket qua test.

  qasper_val       <- allenai/qasper,       split validation (moi bai bao lay 1 cau hoi)
  narrativeqa_val  <- deepmind/narrativeqa, split validation (moi truyen lay 1 cau hoi)
  multifieldqa_en_val / _test <- MultiFieldQA-en khong co validation (LongBench tu tao, chi co
                     test 150 cau) -> tach co dinh 50 cau dev / 100 cau test tu KET QUA DA CO
                     (goi tach_mfqa.py cung thu muc; khong can chay lai suy luan).

VIEC SCRIPT LAM
  1. Tai du lieu, chon N cau (seed co dinh), ghi <REPO>/LongBench/data_val/<ten>_val.jsonl
     cac truong giong LongBench: input, context, answers, length, dataset, language,
     all_classes, _id.
  2. Them "<ten>_val" va "<ten>_val_prompt" vao LongBench/config/dataset2prompt.json va
     dataset2maxlen.json (chep y nguyen cua tap goc).
  3. Va 4 file code de ten "*_val" doc tu data_val/ thay vi THUDM/LongBench test:
       LongBench/pred.py, LongBench/eval.py, offline_clustering.py,
       value_aware_squeezed_project/patches/offline_clustering_value_aware.py
     Moi file sao luu <file>.bak_val truoc khi sua; sua lai lan 2 thi bo qua (khong sua chong).
  Ket qua val ghi vao CUNG cac thu muc pred_* nhung ten file la qasper_val.jsonl /
  narrativeqa_val.jsonl, va thu muc cum .../<ten>_val/ -> khong de len ket qua test.

CACH DUNG (tren may chu, thu muc scripts/):
  python3 tao_tap_valid.py --kiem_tra          # chi xem se sua gi, khong ghi
  python3 tao_tap_valid.py                     # tao du lieu + va code
  python3 tao_tap_valid.py --hoan_tac          # tra 4 file code ve ban .bak_val
"""
import argparse
import json
import os
import random
import re
import shutil
import sys

REPO_MD = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')), 'SqueezedAttention-main')
VA_MD = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')), 'value_aware_squeezed_project')
TAP = ["qasper", "narrativeqa"]


# ------------------------------------------------------------------ du lieu
def _ds_list(x):
    """HF 'datasets' doi Sequence(struct) thanh dict-of-lists; tra ve list-of-dicts."""
    if isinstance(x, dict):
        ks = list(x)
        return [dict(zip(ks, v)) for v in zip(*(x[k] for k in ks))] if ks else []
    return list(x or [])


def _tai_parquet(repo, split):
    """Tai thang file parquet cua split (khong qua load_dataset(repo) — tranh loi
    UnicodeDecodeError 0x8b cua datasets<4). Thu nhanh main, roi refs/convert/parquet."""
    from huggingface_hub import list_repo_files, hf_hub_download
    from datasets import load_dataset
    loi = []
    for rev in (None, "refs/convert/parquet"):
        try:
            fs = list_repo_files(repo, repo_type="dataset", revision=rev)
        except Exception as e:  # noqa: BLE001
            loi.append(f"{rev}: {e}")
            continue
        chon = [f for f in fs if f.endswith(".parquet") and
                (f"/{split}/" in f"/{f}" or os.path.basename(f).startswith(f"{split}-")
                 or os.path.basename(f).startswith(f"{split}."))]
        if not chon:
            loi.append(f"{rev}: khong co file parquet cho split {split}")
            continue
        print(f"    {repo}@{rev or 'main'}: {len(chon)} file parquet ({split})")
        paths = [hf_hub_download(repo, f, repo_type="dataset", revision=rev) for f in chon]
        return load_dataset("parquet", data_files=paths, split="train")
    raise RuntimeError(f"Khong tai duoc parquet {repo}/{split}: " + " | ".join(loi))


def _qasper_s3():
    """Du phong: ban goc tu AI2 (qasper-train-dev-v0.3.tgz) -> list dict cung dang HF."""
    import io
    import tarfile
    import urllib.request
    url = "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz"
    print(f"    du phong: tai {url}")
    raw = urllib.request.urlopen(url, timeout=120).read()
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tf:
        m = next(x for x in tf.getmembers() if x.name.endswith("qasper-dev-v0.3.json"))
        d = json.load(tf.extractfile(m))
    ra = []
    for pid, b in d.items():
        ra.append(dict(id=pid, title=b["title"], abstract=b["abstract"],
                       full_text=dict(section_name=[x["section_name"] for x in b["full_text"]],
                                      paragraphs=[x["paragraphs"] for x in b["full_text"]]),
                       qas=dict(question=[q["question"] for q in b["qas"]],
                                question_id=[q["question_id"] for q in b["qas"]],
                                answers=[dict(answer=[a["answer"] for a in q["answers"]])
                                         for q in b["qas"]])))
    return ra


def _qasper_dap_an(a):
    if a.get("unanswerable"):
        return "Unanswerable"
    if a.get("yes_no") is not None:
        return "Yes" if a["yes_no"] else "No"
    if a.get("extractive_spans"):
        return ", ".join(a["extractive_spans"])
    return (a.get("free_form_answer") or "").strip()


def tao_qasper(n, seed):
    try:
        bai = list(_tai_parquet("allenai/qasper", "validation"))
    except Exception as e:  # noqa: BLE001
        print(f"    !! parquet loi ({e})")
        bai = _qasper_s3()
    rng = random.Random(seed)
    rng.shuffle(bai)
    ra = []
    for b in bai:
        doan = [b["title"], "Abstract", b["abstract"]]
        for ten, ps in zip(b["full_text"]["section_name"], b["full_text"]["paragraphs"]):
            doan.append(ten or "")
            doan += [p for p in ps if p]
        context = "\n\n".join(x for x in doan if x)
        cau = []
        qs = b["qas"]
        for i, q in enumerate(qs["question"]):
            da = []
            for an in _ds_list(_ds_list(qs["answers"])[i]["answer"]
                               if isinstance(qs["answers"], dict) else qs["answers"][i]["answer"]):
                t = _qasper_dap_an(an)
                if t and t not in da:
                    da.append(t)
            if da:
                cau.append((q, da, qs["question_id"][i]))
        if not cau:
            continue
        q, da, qid = rng.choice(cau)
        ra.append(dict(input=q, context=context, answers=da, length=len(context.split()),
                       dataset="qasper", language="en", all_classes=None, _id=f"val_{qid}"))
        if len(ra) >= n:
            break
    return ra


def tao_narrativeqa(n, seed):
    ds = _tai_parquet("deepmind/narrativeqa", "validation")
    theo_truyen = {}
    for r in ds:
        d = r["document"]
        theo_truyen.setdefault(d["id"], dict(text=d["text"], q=[]))["q"].append(
            (r["question"]["text"], [x["text"] for x in _ds_list(r["answers"])]))
    rng = random.Random(seed)
    ids = sorted(theo_truyen)
    rng.shuffle(ids)
    for i in ids:
        rng.shuffle(theo_truyen[i]["q"])
    # Validation chi co ~115 truyen -> vong 1 lay 1 cau/truyen, vong 2 lay cau thu hai, ...
    ra, vong = [], 0
    while len(ra) < n:
        them = 0
        for i in ids:
            t = theo_truyen[i]
            if vong < len(t["q"]) and len(ra) < n:
                q, da = t["q"][vong]
                ra.append(dict(input=q, context=t["text"], answers=da,
                               length=len(t["text"].split()), dataset="narrativeqa",
                               language="en", all_classes=None, _id=f"val_{i}_{vong}"))
                them += 1
        if not them:
            break
        vong += 1
    print(f"    {len(ids)} truyen, lay {len(ra)} cau ({vong} vong)")
    return ra


# ------------------------------------------------------------------ va code
def va_file(path, thay, kiem_tra):
    """thay: list (regex, chuoi_thay, mo_ta). -> so cho da sua."""
    if not os.path.exists(path):
        print(f"  !! khong thay {path}")
        return 0
    s = open(path, encoding="utf-8").read()
    if "SQ_VAL_PATCH" in s:
        print(f"  = {os.path.basename(path)}: da va tu truoc, bo qua")
        return 0
    tong = 0
    for rx, moi, mo_ta in thay:
        s2, k = re.subn(rx, moi, s)
        print(f"  {'+' if k else '!!'} {os.path.basename(path)}: {mo_ta} ({k} cho)")
        s, tong = s2, tong + k
    if tong and not kiem_tra:
        shutil.copy2(path, path + ".bak_val")
        open(path, "w", encoding="utf-8").write("# SQ_VAL_PATCH (tao_tap_valid.py)\n" + s
                                                if not s.startswith("#!") else
                                                s.replace("\n", "\n# SQ_VAL_PATCH (tao_tap_valid.py)\n", 1))
    return tong


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=os.environ.get("REPO", REPO_MD))
    ap.add_argument("--va_project", default=os.environ.get("VA_PROJECT", VA_MD))
    ap.add_argument("--n", type=int, default=200, help="so cau moi tap val")
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--tap", default="qasper narrativeqa")
    ap.add_argument("--kiem_tra", action="store_true", help="chi in, khong ghi gi")
    ap.add_argument("--hoan_tac", action="store_true")
    a = ap.parse_args()
    lb = os.path.join(a.repo, "LongBench")
    data_val = os.path.abspath(os.path.join(lb, "data_val"))
    files = [os.path.join(lb, "pred.py"), os.path.join(lb, "eval.py"),
             os.path.join(a.repo, "offline_clustering.py"),
             os.path.join(a.va_project, "patches", "offline_clustering_value_aware.py")]

    if a.hoan_tac:
        for f in files:
            if os.path.exists(f + ".bak_val"):
                shutil.copy2(f + ".bak_val", f)
                print("da tra lai", f)
        return

    # 1. du lieu
    print(f"== 1. Du lieu validation -> {data_val}")
    if not a.kiem_tra:
        os.makedirs(data_val, exist_ok=True)
    for t in a.tap.split():
        out = os.path.join(data_val, f"{t}_val.jsonl")
        if os.path.exists(out):
            print(f"  = da co {out}, bo qua (xoa file neu muon tao lai)")
            continue
        if a.kiem_tra:
            print(f"  se tao {out} ({a.n} cau)")
            continue
        print(f"  tai {t} ...")
        ra = (tao_qasper if t == "qasper" else tao_narrativeqa)(a.n + 40, a.seed)
        # bo cau trung tai lieu voi tap test LongBench (so 300 ky tu dau cua context)
        try:
            from datasets import load_dataset
            dau = lambda x: " ".join(x.split())[:300].lower()  # noqa: E731
            test = {dau(r["context"]) for r in load_dataset("THUDM/LongBench", t, split="test")}
            truoc = len(ra)
            ra = [r for r in ra if dau(r["context"]) not in test]
            print(f"  bo {truoc - len(ra)} cau trung tai lieu voi LongBench test")
        except Exception as e:  # noqa: BLE001
            print(f"  !! khong kiem tra duoc trung lap voi test ({e}); van tiep tuc")
        ra = ra[: a.n]
        with open(out, "w", encoding="utf-8") as f:
            for r in ra:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  + {out}: {len(ra)} cau, do dai TB {sum(r['length'] for r in ra) / len(ra):.0f} tu")

    # 2. config
    print("== 2. Config LongBench")
    for ten in ("dataset2prompt.json", "dataset2maxlen.json"):
        p = os.path.join(lb, "config", ten)
        c = json.load(open(p, encoding="utf-8"))
        moi = 0
        for t in a.tap.split():
            for k in (t, t + "_prompt"):
                if k in c and k.replace(t, t + "_val") not in c:
                    c[k.replace(t, t + "_val")] = c[k]
                    moi += 1
        print(f"  {'+' if moi else '='} {ten}: them {moi} khoa")
        if moi and not a.kiem_tra:
            shutil.copy2(p, p + ".bak_val")
            json.dump(c, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=4)

    # 3. code
    print("== 3. Va code")
    nap = (lambda q: f"(load_dataset('json', data_files=os.path.join({data_val!r}, dataset + "
                     f"'.jsonl'), split='train') if dataset.endswith('_val') else "
                     f"load_dataset({q}THUDM/LongBench{q}, dataset, split={q}test{q}))")
    rx_nap = r"load_dataset\((['\"])THUDM/LongBench\1,\s*dataset,\s*split=\1test\1\)"
    them_ten = (r'"narrativeqa", "qasper",',
                '"narrativeqa", "qasper", "qasper_val", "narrativeqa_val",',
                'them qasper_val/narrativeqa_val vao --dataset choices')
    for f in files:
        ten = os.path.basename(f)
        thay = []
        if ten == "eval.py":
            thay.append((r"dataset2metric\[dataset\]",
                         'dataset2metric[dataset.replace("_val", "").replace("_test", "")]',
                         "metric cua *_val / *_test = metric tap goc"))
        else:
            thay.append((rx_nap, lambda m: nap(m.group(1)), "*_val doc tu data_val/"))
            if ten != "pred.py":
                thay.append(them_ten)
        va_file(f, thay, a.kiem_tra)
    # 4. MultiFieldQA-en: tach dev/test tu ket qua da co
    print("== 4. MultiFieldQA-en: tach 50 dev / 100 test (tach_mfqa.py)")
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import tach_mfqa
        argv = sys.argv
        sys.argv = ["tach_mfqa.py", "--pred_root", lb] + (["--kiem_tra"] if a.kiem_tra else [])
        try:
            tach_mfqa.main()
        finally:
            sys.argv = argv
    except ImportError:
        print("  !! thieu tach_mfqa.py cung thu muc -> bo qua MFQA")
    except SystemExit as e:
        print(f"  !! {e}")

    # 5. tom tat
    print("== 5. Tap validation")
    for t in a.tap.split():
        f = os.path.join(data_val, f"{t}_val.jsonl")
        n = sum(1 for x in open(f, encoding="utf-8") if x.strip()) if os.path.exists(f) else 0
        print(f"  {t}_val: {n} cau  (chay suy luan bang run_repeat.sh DATASETS=\"{t}_val\")")
    sp = os.path.join(lb, "data_val", "multifieldqa_en_split.json")
    if os.path.exists(sp):
        d = json.load(open(sp, encoding="utf-8"))
        print(f"  multifieldqa_en_val: {len(d['dev'])} cau, multifieldqa_en_test: {len(d['test'])} cau"
              "  (tach tu ket qua test, khong chay lai)")
    print("\nXong." if not a.kiem_tra else "\n(--kiem_tra: chua ghi gi)")
    print("Kiem tra 'import os' co trong pred.py / offline_clustering*.py (cac file nay da dung os).")


if __name__ == "__main__":
    main()
