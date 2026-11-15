#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dem_lap.py — do hien tuong LAP TOKEN trong sinh van ban (Fig. 3a).

KHONG CAN GPU, KHONG CHAY LAI MO HINH. Chi doc cac file pred/*.jsonl da co.

HAI CACH DO
  --cach tu_lap   (MAC DINH)
      Token duoc tinh la LAP neu no nam trong mot n-gram DA XUAT HIEN
      TRUOC DO trong CHINH cau tra loi. Day la do "tu lap" (self-repetition).
      Khong can prompt -> tinh duoc ngay tu file san co.

  --cach chep_prompt
      Token duoc tinh la CHEP neu n-gram chua no co mat trong PROMPT.
      Dung dinh nghia bai bao, nhung pred.py (dong 127) KHONG luu prompt,
      nen phai dung lai prompt tu bo du lieu + config/dataset2prompt.json.
      Can mang de tai LongBench, khong can GPU.

VI SAO MAC DINH LA tu_lap
  Vong lap chep prompt bieu hien ra ngoai chinh la tu lap: mo hinh phat lai
  cau hoi + ngu canh nhieu lan, nen cac n-gram lap lai ngay trong ban than
  cau tra loi. Do tu lap bat duoc dung hien tuong do ma khong can them gi.
  NHUNG khi viet caption phai goi dung ten: "repeated tokens within the
  generation", KHONG duoc viet la "copied from the prompt" — hai dai luong
  khac nhau.

DAU RA
  <out>_so.xlsx  — CUNG LUOC DO 7 cot voi bang_tot_nhat_so.xlsx, nen ve
                   duoc bang chinh ve_so_sanh.py:
      python3 ve_so_sanh.py --xlsx dem_lap_so.xlsx --luoi \\
              --nhan_y "Avg. repeated tokens / sample"

CACH DUNG
    python3 dem_lap.py
    python3 dem_lap.py --n 4 --seeds "0 1 2"
    python3 dem_lap.py --cach chep_prompt
"""
import argparse
import glob
import json
import os
import re

CAU_HINH = {
    "pred_root": "",
    "model":     "LLaMA-2-7B-32K",
    "pc":        5,
    "datasets":  "qasper narrativeqa multifieldqa_en",
    "muc_nen":   "70 80 90",
    "seeds":     "0 1 2",
    "n":         4,        # do dai n-gram coi la "lap"
    "out":       "dem_lap",
}

NHANH = {"_thr": "Squeezed", "_bgt": "ClusterKV"}
TEN_NHANH = {"_thr": "Cải tiến Sq", "_bgt": "Cải tiến ClusterKV"}
TEN_DS = {"multifieldqa_en": "MultiFieldQA-en", "qasper": "Qasper",
          "narrativeqa": "NarrativeQA"}


def tach(s):
    """Tach token tho: chu + so, bo dau cau. Du cho viec dem lap."""
    return re.findall(r"\w+", s.lower())


def _chuoi_dai_nhat(co):
    """Doan LIEN TIEP dai nhat cac token bi danh dau la lap.

    Day moi la dau hieu cua VONG LAP sinh van ban. Tong so token lap co the
    cao chi vi cau tra loi dai; mot doan lap lien tiep 60 token thi khong
    the giai thich bang do dai duoc."""
    dai = cur = 0
    for b in co:
        cur = cur + 1 if b else 0
        dai = max(dai, cur)
    return dai


def dem_tu_lap(toks, n):
    """So token nam trong n-gram DA xuat hien truoc do trong cung chuoi.

    Moi vi tri i mo dau mot n-gram. Neu n-gram do da gap, danh dau ca n
    token thuoc no. Dem SO TOKEN bi danh dau (khong phai so n-gram) de con
    so co don vi "token/cau", so sanh duoc voi do dai cau tra loi.

    -> (so token lap, doan lap lien tiep dai nhat)
    """
    if len(toks) < n:
        return 0, 0
    da_gap, co = set(), [False] * len(toks)
    for i in range(len(toks) - n + 1):
        g = tuple(toks[i:i + n])
        if g in da_gap:
            for j in range(i, i + n):
                co[j] = True
        else:
            da_gap.add(g)
    return sum(co), _chuoi_dai_nhat(co)


def dem_chep(toks, tap_ngram_prompt, n):
    """So token nam trong n-gram CO MAT trong prompt."""
    if len(toks) < n:
        return 0, 0
    co = [False] * len(toks)
    for i in range(len(toks) - n + 1):
        if tuple(toks[i:i + n]) in tap_ngram_prompt:
            for j in range(i, i + n):
                co[j] = True
    return sum(co), _chuoi_dai_nhat(co)


def ngram_prompt(s, n):
    t = tach(s)
    return {tuple(t[i:i + n]) for i in range(max(0, len(t) - n + 1))}


# ---------------------------------------------------------------- prompt
_CACHE_PROMPT = {}


def nap_prompt(ds, n, root_cfg):
    """{index cau: tap n-gram cua prompt}. Chi dung cho --cach chep_prompt."""
    if ds in _CACHE_PROMPT:
        return _CACHE_PROMPT[ds]
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("!! --cach chep_prompt can thu vien 'datasets'.\n"
                         "   Cai:  pip install datasets")
    f_cfg = os.path.join(root_cfg, "dataset2prompt.json")
    if not os.path.exists(f_cfg):
        raise SystemExit(f"!! Khong thay {f_cfg}. Dung --cfg de chi duong dan"
                         " toi thu muc LongBench/config.")
    mau = json.load(open(f_cfg, encoding="utf-8"))[ds]
    print(f"   [chep_prompt] tai {ds} tu THUDM/LongBench ...")
    d = load_dataset("THUDM/LongBench", ds, split="test")
    ra = [ngram_prompt(mau.format(**o), n) for o in d]
    _CACHE_PROMPT[ds] = ra
    return ra


# ---------------------------------------------------------------- doc file
def do_file(path, n, cach, prompts):
    """-> dict 4 chi so trung binh tren cac cau, va so cau.

        lap        so token lap / cau          (chi so trong hinh cua ban)
        do_dai     so token sinh ra / cau      (de chuan hoa)
        lap_100    so token lap tren 100 token sinh ra
        chuoi      doan lap lien tiep dai nhat / cau

    PHAI co 'do_dai' thi 'lap' moi doc duoc: mot phuong phap sinh cau ngan
    hon se co 'lap' thap hon du khong he bot vong lap.
    """
    tong, tong_d, tong_c, soc = 0, 0, 0, 0
    ty_le = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            toks = tach(json.loads(line)["pred"])
            if cach == "tu_lap":
                sl, ch = dem_tu_lap(toks, n)
            else:
                g = prompts[i] if prompts and i < len(prompts) else set()
                sl, ch = dem_chep(toks, g, n)
            tong += sl
            tong_c += ch
            tong_d += len(toks)
            ty_le.append(100.0 * sl / len(toks) if toks else 0.0)
            soc += 1
    if not soc:
        return None, 0
    return {"lap": tong / soc, "do_dai": tong_d / soc,
            "lap_100": sum(ty_le) / soc, "chuoi": tong_c / soc}, soc


def quet(a, ds, cach, prompts):
    """-> {(nen, nhanh, cfg): {seed: gia tri}}"""
    ra = {}
    # All-KV: khong nen, nen chi co MOT gia tri. Ghi lai o moi muc nen de
    # script ve chi can loc theo muc nen la co du lieu.
    f = os.path.join(a.pred_root, "pred_baseline", f"{a.model}_baseline",
                     f"{ds}.jsonl")
    if os.path.exists(f):
        v, _ = do_file(f, a.n, cach, prompts)
        for nen in (int(x) for x in a.muc_nen.split()):
            ra[(nen, "—", "All KV")] = {"-": v}

    for seed in a.seeds.split():
        for tag, ten_moc in NHANH.items():
            # moc goc
            for d in sorted(glob.glob(os.path.join(
                    a.pred_root, f"pred_s{seed}{tag}",
                    f"{a.model}_PC{a.pc}_PERC*"))):
                f = os.path.join(d, f"{ds}.jsonl")
                if os.path.exists(f):
                    v, _ = do_file(f, a.n, cach, prompts)
                    nen = int(round(float(d.rsplit("PERC", 1)[-1]) * 100))
                    ra.setdefault((nen, "mốc", ten_moc), {})[seed] = v
            # cac cau hinh cai tien
            for d in sorted(glob.glob(os.path.join(
                    a.pred_root, f"pred_E1_b*_g*_s{seed}{tag}*",
                    f"{a.model}_PC{a.pc}_PERC*"))):
                cfg = os.path.basename(os.path.dirname(d))
                if "_vb" in cfg or "-lim" in cfg:
                    continue
                m = re.search(r"pred_E1_b([^_]+)_g([^_]+)_s\d+", cfg)
                if not m:
                    continue
                f = os.path.join(d, f"{ds}.jsonl")
                if os.path.exists(f):
                    v, _ = do_file(f, a.n, cach, prompts)
                    nen = int(round(float(d.rsplit("PERC", 1)[-1]) * 100))
                    ten = f"β={m.group(1)}  γ={m.group(2)}"
                    ra.setdefault((nen, TEN_NHANH[tag], ten), {})[seed] = v
    return ra


def tb_std(per, khoa=None):
    """per = {seed: dict 4 chi so}. khoa = ten chi so can tong hop."""
    vs = [(v[khoa] if khoa else v) for v in per.values() if v is not None]
    vs = [x for x in vs if x is not None]
    if not vs:
        return None, None, 0
    mu = sum(vs) / len(vs)
    if len(vs) < 2:
        return mu, None, 1
    sd = (sum((x - mu) ** 2 for x in vs) / (len(vs) - 1)) ** 0.5
    return mu, sd, len(vs)


# ---------------------------------------------------------------- main
def main():
    base = os.environ.get("BASE", os.path.expanduser(
        "~/Desktop/DucDang/SqueezedAttention-main"))
    mac_dinh = CAU_HINH["pred_root"] or os.environ.get(
        "ROOT", base + "/SqueezedAttention-main/LongBench")

    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_root", default=mac_dinh)
    ap.add_argument("--cfg", default="",
                    help="thu muc LongBench/config (chi can cho chep_prompt)")
    ap.add_argument("--model", default=CAU_HINH["model"])
    ap.add_argument("--pc", type=int, default=CAU_HINH["pc"])
    ap.add_argument("--datasets", default=CAU_HINH["datasets"])
    ap.add_argument("--muc_nen", default=CAU_HINH["muc_nen"])
    ap.add_argument("--seeds", default=CAU_HINH["seeds"])
    ap.add_argument("--n", type=int, default=CAU_HINH["n"],
                    help="do dai n-gram coi la lap (mac dinh 4)")
    ap.add_argument("--cach", default="tu_lap",
                    choices=["tu_lap", "chep_prompt"])
    ap.add_argument("--out", default=CAU_HINH["out"])
    a = ap.parse_args()

    if not a.cfg:
        a.cfg = os.path.join(a.pred_root, "config")
    loc = set(a.muc_nen.split())

    print(f"=== pred_root : {a.pred_root}")
    print(f"=== cach do   : {a.cach}   (n-gram = {a.n})")
    print(f"=== seeds     : {a.seeds}\n")

    bang, moc_allkv = [], {}
    for ds in a.datasets.split():
        prompts = (nap_prompt(ds, a.n, a.cfg)
                   if a.cach == "chep_prompt" else None)
        ra = quet(a, ds, a.cach, prompts)
        if not ra:
            print(f"!! {ds}: khong tim thay file pred nao.")
            continue
        print(f"── {TEN_DS.get(ds, ds)}")
        print(f"{'Nen':>5}  {'Nhanh':<20}{'Cau hinh':<16}"
              f"{'token lap':>11}{'std':>8}{'n':>4}"
              f"{'do dai':>9}{'lap/100':>9}{'chuoi':>8}")
        for (nen, nhanh, cfg) in sorted(ra, key=lambda k: (k[0], k[1], k[2])):
            if loc and str(nen) not in loc:
                continue
            per = ra[(nen, nhanh, cfg)]
            mu, sd, ns = tb_std(per, "lap")
            dd, _, _ = tb_std(per, "do_dai")
            r100, _, _ = tb_std(per, "lap_100")
            ch, _, _ = tb_std(per, "chuoi")
            g = lambda x, f="{:.2f}": "—" if x is None else f.format(x)
            print(f"{nen:>4}%  {nhanh:<20}{cfg:<16}"
                  f"{g(mu):>11}{g(sd):>8}{ns:>4}"
                  f"{g(dd, '{:.1f}'):>9}{g(r100):>9}{g(ch, '{:.1f}'):>8}")
            bang.append([TEN_DS.get(ds, ds), nen, nhanh, cfg, mu, sd, ns,
                         dd, r100, ch])
            if nhanh == "—":
                moc_allkv[(TEN_DS.get(ds, ds), nen)] = mu
        print()

    if not bang:
        print("Khong co du lieu. Kiem tra --pred_root.")
        return

    # ---- KIEM TRA BAT BUOC TRUOC KHI DUA VAO BAI -----------------------
    # All-KV la mo hinh KHONG nen: no la MUC SAN. Nen lam sinh ra vong lap,
    # nen moi phuong phap nen deu phai NAM TREN hoac BANG All-KV. Mot cau
    # hinh nen co so token lap THAP HON HAN All-KV khong chung minh duoc
    # dieu gi ve vong lap — kha nang cao la no sinh cau NGAN hon, hoac chi
    # so dang dem phan trung n-gram HOP LE voi ngu canh (dac biet n nho).
    canh_bao = []
    for r in bang:
        ds, nen, nhanh, cfg, mu = r[0], r[1], r[2], r[3], r[4]
        g0 = moc_allkv.get((ds, nen))
        if nhanh != "—" and mu is not None and g0 and mu < 0.5 * g0:
            canh_bao.append((ds, nen, nhanh, cfg, mu, g0))
    if canh_bao:
        print("=" * 72)
        print("!! CANH BAO — cac cau hinh NEN co token lap THAP HON NUA so")
        print("   voi All-KV (mo hinh khong nen). Day KHONG phai bang chung")
        print("   'cai tien chua vong lap'; hay kiem tra cot 'do dai' va")
        print(f"   chay lai voi n lon hon (dang dung n = {a.n}, nen thu 8).")
        print("=" * 72)
        for ds, nen, nhanh, cfg, mu, g0 in canh_bao[:12]:
            print(f"   {ds:<16}{nen:>3}%  {nhanh:<20}{cfg:<16}"
                  f"{mu:>8.2f}  vs All-KV {g0:.2f}")
        print()

    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
    except ImportError:
        print("!! Thieu openpyxl — khong ghi duoc xlsx.  pip install openpyxl")
        return

    wb = Workbook()
    ws = wb.active
    ws.title = "so"            # TEN SHEET phai la "so" de ve_so_sanh doc duoc
    head = ["Bộ dữ liệu", "Mức nén", "Nhánh", "Cấu hình", "mean", "std",
            "n_seed", "độ dài TB", "lặp/100 token", "chuỗi lặp dài nhất"]
    ws.append(head)
    for c in ws[1]:
        c.font = Font(bold=True, color="14243D", size=10)
        c.alignment = Alignment(horizontal="center", wrap_text=True)
        c.fill = PatternFill("solid", fgColor="F1EEE7")
        c.border = Border(bottom=Side(style="medium", color="14243D"))
    thin = Side(style="thin", color="D9D9D9")
    for r in bang:
        ws.append(r)
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.border = Border(bottom=thin)
            if isinstance(c.value, float):
                c.number_format = "0.0000"
    for i, h in enumerate(head, 1):
        ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = \
            {"Bộ dữ liệu": 17, "Nhánh": 20, "Cấu hình": 16,
             "chuỗi lặp dài nhất": 13, "lặp/100 token": 12}.get(h, 10)
    c = ws.cell(row=ws.max_row + 2, column=1,
                value=(f"Số token lặp trung bình mỗi câu, cách đo "
                       f"'{a.cach}', n-gram = {a.n}. "
                       + ("Token nằm trong n-gram đã xuất hiện trước đó "
                          "trong chính câu trả lời (tự lặp) — khi viết "
                          "caption phải gọi đúng là 'repeated tokens within "
                          "the generation', không phải 'copied from the "
                          "prompt'."
                          if a.cach == "tu_lap" else
                          "Token nằm trong n-gram có mặt trong prompt.")))
    c.font = Font(italic=True, size=9, color="8A8578")

    p = a.out + "_so.xlsx"
    wb.save(p)
    print("Da ghi:", os.path.abspath(p), f"  ({len(bang)} dong)")
    print("\nVe bieu do bang:")
    print(f'  python3 ve_so_sanh.py --xlsx {os.path.basename(p)} --luoi \\')
    print('          --nhan_y "Avg. repeated tokens / sample" --caption khong')


if __name__ == "__main__":
    main()
