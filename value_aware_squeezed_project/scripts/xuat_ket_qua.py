#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xuat_ket_qua.py — gom diem qa_f1 tu cac thu muc pred_* thanh MOT file .xlsx.

CACH DUNG: sua khoi CAU_HINH ngay ben duoi roi chay
    python3 xuat_ket_qua.py
(doi so dong lenh, neu truyen, se de len CAU_HINH — vd --seeds "0")

NGUON (giong aggregate_runs.py / bang_tong_hop.py)
    pred_baseline/<model>_baseline/<ds>.jsonl          -> All KV
    pred_s<seed>_thr/<model>_PC5_PERC<p>/              -> Squeezed   (threshold)
    pred_s<seed>_bgt/<model>_PC5_PERC<p>/              -> ClusterKV  (budget)
    pred_E1_b<B>_g<G>_s<seed>_thr|_bgt/...             -> E1 (value-aware)
    pred_hsq_s<seed>/...                               -> H-Sq (neu bat "hsq")
  Bo qua: *_vb* (thanh phan 4), *-lim<N>* (chay thu, tru khi GOM_LIM),
  pred_s<seed>/ tran (budget truoc khi va code; chi canh bao).

FILE XLSX — moi (moc nen[, seed]) la mot nhom, trong nhom XEP TU CAO XUONG THAP, co
cot hang va Δ so voi Squeezed (threshold) cung moc. Moi so tong hop la
CONG THUC tro ve sheet chi_tiet.
  bang_rong  hang = (moc nen, cau hinh), cot = tung dataset (mean, std, n,
             Δ Sq) + trung binh cac tap; xep theo trung binh cac tap
  tong_hop   1 dong / (dataset, moc nen, cau hinh); xep theo mean
  chi_tiet   1 dong / lan chay, kem Δ vs Squeezed CUNG SEED (ghep cap)
  cau_hinh   tham so da dung, canh bao, thu muc bi bo qua
"""
import argparse
import glob
import json
import os
import re
import string
import sys
from collections import Counter, defaultdict

# ═══════════════════════════════════════════════════════════════════════════
#  CAU HINH — sua o day roi chay `python3 xuat_ket_qua.py`
# ═══════════════════════════════════════════════════════════════════════════
CAU_HINH = {
    # thu muc LongBench/ chua cac pred_*; None = lay $ROOT, roi duong dan mac dinh
    "pred_root":   None,
    "datasets":    "qasper multifieldqa_en narrativeqa",
    "seeds":       "0 1 2",
    # False = MOI SEED RIENG: Squeezed/ClusterKV/E1 lay dung seed do, so voi
    #         Squeezed CUNG seed, khong lay trung binh (All KV dung chung).
    # True  = gop cac seed: mean ± std qua cac seed tren.
    "gop_seed":    False,
    "muc_nen":     "70 80 90",          # "" = moi muc nen tim thay
    "che_do":      "threshold budget",  # threshold = Squeezed, budget = ClusterKV
    # allkv  sq (Squeezed)  ck (ClusterKV)  e1 (value-aware)  hsq (phan cap)
    "phuong_phap": "allkv sq ck e1",
    "beta":        "",                  # loc E1, vd "0.05 0.2"; "" = tat ca
    "gamma":       "",                  # loc E1, vd "0 0.1";    "" = tat ca
    "gom_lim":     False,               # True = gom ca lan chay thu -lim<N>
    "model":       "LLaMA-2-7B-32K",
    "pc":          5,
    "out":         "ket_qua.xlsx",
}
_ROOT_MAC_DINH = "../../SqueezedAttention-main/LongBench"

QA = {"narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa",
      "musique", "triviaqa"}
CAT_DONG_DAU = {"trec", "triviaqa", "samsum", "lsht"}       # giong eval.py
TAG = {"_thr": "threshold", "_bgt": "budget"}
GOC_THEO_CHE_DO = {"threshold": "Squeezed", "budget": "ClusterKV"}
TEN_DS = {"qasper": "Qasper", "multifieldqa_en": "MFQA-en",
          "narrativeqa": "NarrativeQA"}


# ------------------------------------------------------------------ diem
def _norm(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _f1(p, g):
    common = Counter(p) & Counter(g)
    ns = sum(common.values())
    if ns == 0:
        return 0.0
    pr, rc = ns / len(p), ns / len(g)
    return 2 * pr * rc / (pr + rc)


def qa_f1(pred, gts):
    p = _norm(pred).split()
    best = 0.0
    for gt in gts:
        g = _norm(gt).split()
        if p and g:
            best = max(best, _f1(p, g))
    return best


_METRIC = {}


def _metric_longbench(root, ds):
    """Ham cham diem cua LongBench cho tap khong phai QA (neu nap duoc)."""
    if ds in _METRIC:
        return _METRIC[ds]
    fn = None
    try:
        sys.path.insert(0, root)
        import metrics as M  # noqa: E402  (LongBench/metrics.py)
        bang = {"gov_report": M.rouge_score, "qmsum": M.rouge_score,
                "multi_news": M.rouge_score, "samsum": M.rouge_score,
                "trec": M.classification_score, "lcc": M.code_sim_score,
                "repobench-p": M.code_sim_score,
                "passage_count": M.count_score,
                "passage_retrieval_en": M.retrieval_score}
        fn = bang.get(ds)
    except Exception as e:  # noqa: BLE001
        print(f"!! khong nap duoc LongBench/metrics.py ({e}); {ds} se bi bo qua")
    _METRIC[ds] = fn
    return fn


def diem_file(path, ds, root):
    """-> (diem trung binh x100, so cau) hoac (None, 0)."""
    xs = []
    for _d in ("_val", "_test"):                     # tap tach dev/test: cung metric tap goc
        if ds.endswith(_d):
            ds = ds[: -len(_d)]
    fn = None if ds in QA else _metric_longbench(root, ds)
    if ds not in QA and fn is None:
        return None, 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            pred = d["pred"]
            if ds in CAT_DONG_DAU:
                pred = pred.lstrip("\n").split("\n")[0]
            if ds in QA:
                xs.append(qa_f1(pred, d["answers"]))
            else:
                xs.append(max(fn(pred, gt, all_classes=d.get("all_classes"))
                              for gt in d["answers"]))
    return (100.0 * sum(xs) / len(xs), len(xs)) if xs else (None, 0)


# ------------------------------------------------------------------ quet


# ------------------------------------------------------------------ quet
def _pct(ten_thu_muc):
    return int(round(float(ten_thu_muc.rsplit("PERC", 1)[-1]) * 100))


def quet(a):
    rows, canh_bao, bo_qua = [], [], set()
    root, model, pc = a.pred_root, a.model, a.pc
    lim = r"(?:-lim\d+)?" if a.gom_lim else ""
    for ds in a.datasets:
        if "allkv" in a.phuong_phap:
            p = os.path.join(root, "pred_baseline", f"{model}_baseline", f"{ds}.jsonl")
            if os.path.exists(p):
                v, n = diem_file(p, ds, root)
                if v is not None:
                    rows.append(dict(dataset=ds, phuong_phap="All KV", che_do="—",
                                     beta="", gamma="", muc_nen=0, seed="all",
                                     so_cau=n, diem=v, duong_dan=p))
        for seed in a.seeds:
            for tag, che_do in TAG.items():
                if che_do not in a.che_do:
                    continue
                goc = GOC_THEO_CHE_DO[che_do]
                if {"Squeezed": "sq", "ClusterKV": "ck"}[goc] in a.phuong_phap:
                    for d in sorted(glob.glob(os.path.join(
                            root, f"pred_s{seed}*{tag}", f"{model}_PC{pc}_PERC*"))):
                        cfg = os.path.basename(os.path.dirname(d))
                        if re.fullmatch(rf"pred_s{seed}{lim}{tag}", cfg):
                            _them(rows, a, ds, d, goc, che_do, "", "", seed, "-lim" in cfg)
                if "e1" in a.phuong_phap:
                    for d in sorted(glob.glob(os.path.join(
                            root, f"pred_E1_b*_g*_s{seed}*", f"{model}_PC{pc}_PERC*"))):
                        cfg = os.path.basename(os.path.dirname(d))
                        m = re.fullmatch(
                            rf"pred_E1_b([^_]+)_g([^_]+)_s{seed}({lim}){tag}", cfg)
                        if not m:
                            if tag in cfg:
                                bo_qua.add(cfg)
                            continue
                        b, g = m.group(1), m.group(2)
                        if (a.beta and b not in a.beta) or (a.gamma and g not in a.gamma):
                            continue
                        _them(rows, a, ds, d, f"E1 b{b} g{g}", che_do, b, g, seed,
                              bool(m.group(3)))
            if "hsq" in a.phuong_phap:
                for d in sorted(glob.glob(os.path.join(
                        root, f"pred_hsq_s{seed}", f"{model}_PC1_*_PERC1_*_lookup"))):
                    mm = re.search(r"_PERC1_([0-9.]+)_PC2_", os.path.basename(d))
                    f = os.path.join(d, f"{ds}.jsonl")
                    if mm and os.path.exists(f):
                        p_ = int(round(float(mm.group(1)) * 100))
                        v, n = diem_file(f, ds, root)
                        if v is not None and (not a.muc_nen or p_ in a.muc_nen):
                            rows.append(dict(dataset=ds, phuong_phap="H-Sq",
                                             che_do="threshold", beta="", gamma="",
                                             muc_nen=p_, seed=seed, so_cau=n,
                                             diem=v, duong_dan=f))
            cu = glob.glob(os.path.join(root, f"pred_s{seed}",
                                        f"{model}_PC{pc}_PERC*", f"{ds}.jsonl"))
            if cu:
                canh_bao.append(f"pred_s{seed}/ (khong duoi che do) co {len(cu)} file "
                                f"{ds}.jsonl: ket qua budget TRUOC khi va code, "
                                f"KHONG dua vao bang.")
    return rows, canh_bao, sorted(bo_qua)


def _them(rows, a, ds, d, ten, che_do, b, g, seed, la_lim):
    p_ = _pct(d)
    if a.muc_nen and p_ not in a.muc_nen:
        return
    f = os.path.join(d, f"{ds}.jsonl")
    if not os.path.exists(f):
        return
    v, n = diem_file(f, ds, a.pred_root)
    if v is None:
        return
    rows.append(dict(dataset=ds, phuong_phap=ten + (" [lim]" if la_lim else ""),
                     che_do=che_do, beta=b, gamma=g, muc_nen=p_, seed=seed,
                     so_cau=n, diem=v, duong_dan=f))


# ------------------------------------------------------------------ xlsx
def ghi_xlsx(a, rows, canh_bao, bo_qua):
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.formatting.rule import CellIsRule
    from openpyxl.utils import get_column_letter as L

    F = "Arial"
    HDR = PatternFill("solid", fgColor="1F3864")
    MOC = PatternFill("solid", fgColor="FFF2CC")
    SOC = PatternFill("solid", fgColor="F2F2F2")
    thin = Side(style="thin", color="BFBFBF")
    BOX = Border(left=thin, right=thin, top=thin, bottom=thin)
    C = Alignment(horizontal="center", vertical="center", wrap_text=True)
    Lt = Alignment(horizontal="left", vertical="center")
    DELTA = '+0.00;-0.00;0.00'

    def hdr(ws, vals):
        for i, v in enumerate(vals, 1):
            c = ws.cell(row=1, column=i, value=v)
            c.font = Font(name=F, bold=True, color="FFFFFF")
            c.fill, c.alignment, c.border = HDR, C, BOX

    def to_mau(ws, rng):
        ws.conditional_formatting.add(rng, CellIsRule(
            operator="greaterThan", formula=["0"], font=Font(name=F, color="006100")))
        ws.conditional_formatting.add(rng, CellIsRule(
            operator="lessThan", formula=["0"], font=Font(name=F, color="C00000")))

    def bf(v):
        return float(v) if v not in ("", None) else None

    def la_moc(ten, cd):
        return ten == "Squeezed" and cd == "threshold"

    # ------------------------------------------------------------------
    # PS = tung seed rieng (khong lay trung binh). Moi nhom so sanh =
    # (dataset, muc nen, seed): cau hinh nao cung so voi Squeezed CUNG seed.
    # All KV khong phu thuoc seed -> seed = "all", dung chung cho moi seed.
    # ------------------------------------------------------------------
    PS = not a.gop_seed
    seed_idx = {s: i for i, s in enumerate(a.seeds)}

    def sk(r):
        if r["seed"] == "all":
            return "all"
        return str(r["seed"]) if PS else "TB"

    def sk_ord(s):
        return -1 if s == "all" else seed_idx.get(s, 99)

    nhom = defaultdict(list)
    for r in rows:
        nhom[(r["dataset"], r["muc_nen"], sk(r), r["phuong_phap"], r["che_do"])].append(r["diem"])
    # gia tri Python CHI de sap xep; so hien thi trong file la cong thuc
    val_py = {k: sum(v) / len(v) for k, v in nhom.items()}
    info = {(r["phuong_phap"], r["che_do"]): (r["beta"], r["gamma"]) for r in rows}
    ds_idx = {d: i for i, d in enumerate(a.datasets)}

    wb = Workbook()

    # ============================ chi_tiet (nguon cua moi cong thuc)
    ct = wb.active
    ct.title = "chi_tiet"
    hdr(ct, ["dataset", "mức nén %", "phương pháp", "chế độ", "β", "γ", "seed",
             "số câu", "điểm qa_f1", "Δ vs Squeezed cùng seed", "đường dẫn"])
    rows_ct = sorted(rows, key=lambda r: (
        ds_idx[r["dataset"]], r["muc_nen"], sk_ord(str(r["seed"])),
        -val_py[(r["dataset"], r["muc_nen"], sk(r), r["phuong_phap"], r["che_do"])],
        r["phuong_phap"], r["che_do"]))
    for i, r in enumerate(rows_ct, 2):
        vals = [r["dataset"], r["muc_nen"], r["phuong_phap"], r["che_do"],
                bf(r["beta"]), bf(r["gamma"]), str(r["seed"]), r["so_cau"], round(r["diem"], 4)]
        for j, v in enumerate(vals, 1):
            c = ct.cell(row=i, column=j, value=v)
            c.font = Font(name=F, color="0000FF" if j == 9 else "000000")
            c.alignment, c.border = (Lt if j == 3 else C), BOX
        ct[f"I{i}"].number_format = "0.00"
        ct[f"J{i}"] = (f'=IF(OR(G{i}="all",COUNTIFS($A:$A,A{i},$B:$B,B{i},$C:$C,"Squeezed",'
                       f'$D:$D,"threshold",$G:$G,G{i})=0),"",I{i}-SUMIFS($I:$I,$A:$A,A{i},'
                       f'$B:$B,B{i},$C:$C,"Squeezed",$D:$D,"threshold",$G:$G,G{i}))')
        ct[f"J{i}"].number_format = DELTA
        ct[f"J{i}"].font, ct[f"J{i}"].alignment, ct[f"J{i}"].border = Font(name=F), C, BOX
        ct[f"K{i}"] = os.path.relpath(r["duong_dan"], a.pred_root)
        ct[f"K{i}"].font = Font(name=F, color="808080", size=9)
        if la_moc(r["phuong_phap"], r["che_do"]):
            for col in "ABCDEFGHIJ":
                ct[f"{col}{i}"].fill = MOC
    n = len(rows_ct) + 1
    to_mau(ct, f"J2:J{n}")
    for k, w in zip("ABCDEFGHIJK", [16, 10, 22, 11, 7, 7, 7, 8, 11, 14, 70]):
        ct.column_dimensions[k].width = w
    ct.freeze_panes = "D2"
    ct.auto_filter.ref = f"A1:K{n}"

    X = "chi_tiet!"
    rA, rB, rC, rD, rG, rI = (f"{X}${c}$2:${c}${n}" for c in "ABCDGI")

    def tieu_chi(ds, muc, pp, cd, seed):
        """Dieu kien COUNTIFS/AVERAGEIFS tren chi_tiet. seed chi dua vao khi PS."""
        t = f"{rA},{ds},{rB},{muc},{rC},{pp},{rD},{cd}"
        return t + (f",{rG},{seed}" if PS else "")

    def tich(ds, muc, pp, cd):
        return f"({rA}={ds})*({rB}={muc})*({rC}={pp})*({rD}={cd})"

    # ============================ tong_hop
    th = wb.create_sheet("tong_hop")
    if PS:
        cot = [("dataset", 16), ("mức nén %", 10), ("seed", 7), ("hạng", 6),
               ("phương pháp", 22), ("chế độ", 11), ("β", 7), ("γ", 7),
               ("điểm", 9), ("Δ vs Squeezed cùng seed", 14), ("Δ vs gốc cùng chế độ", 13)]
    else:
        cot = [("dataset", 16), ("mức nén %", 10), ("seed", 7), ("hạng", 6),
               ("phương pháp", 22), ("chế độ", 11), ("β", 7), ("γ", 7),
               ("mean", 9), ("Δ vs Squeezed", 13), ("Δ vs gốc cùng chế độ", 13),
               ("n_seed", 8), ("std(seed)", 10), ("min", 8), ("max", 8),
               ("điểm từng seed", 42)]
    hdr(th, [c for c, _ in cot])
    keys = sorted(val_py, key=lambda k: (ds_idx[k[0]], k[1], sk_ord(k[2]), -val_py[k], k[3], k[4]))
    seed_txt = defaultdict(list)
    for r in rows_ct:
        seed_txt[(r["dataset"], r["muc_nen"], sk(r), r["phuong_phap"], r["che_do"])].append(
            f"{r['diem']:.2f}" if r["seed"] == "all" else f"s{r['seed']}={r['diem']:.2f}")
    soc, truoc = False, None
    for i, k in enumerate(keys, 2):
        ds, p_, s_, ten, cd = k
        b, g = info[(ten, cd)]
        for j, v in enumerate([ds, p_, s_ if PS or s_ == "all" else " ".join(a.seeds),
                               None, ten, cd, bf(b), bf(g)], 1):
            th.cell(row=i, column=j, value=v)
        seed_ref = f"$C{i}"
        dk = tieu_chi(f"$A{i}", f"$B{i}", f"$E{i}", f"$F{i}", seed_ref)
        moc = tieu_chi(f"$A{i}", f"$B{i}", '"Squeezed"', '"threshold"', seed_ref)
        goc = GOC_THEO_CHE_DO.get(cd)
        cg = tieu_chi(f"$A{i}", f"$B{i}", f'"{goc}"', f"$F{i}", seed_ref) if goc else None
        th[f"D{i}"] = f'=IF(I{i}="","",COUNTIFS($A:$A,A{i},$B:$B,B{i},$C:$C,C{i},$I:$I,">"&I{i})+1)'
        th[f"I{i}"] = f'=IFERROR(AVERAGEIFS({rI},{dk}),"")'
        th[f"J{i}"] = f'=IF(OR($B{i}=0,COUNTIFS({moc})=0),"",I{i}-AVERAGEIFS({rI},{moc}))'
        th[f"K{i}"] = (f'=IF(OR($B{i}=0,COUNTIFS({cg})=0),"",I{i}-AVERAGEIFS({rI},{cg}))'
                       if goc else '=""')
        if not PS:
            th[f"L{i}"] = f"=COUNTIFS({dk})"
            th[f"M{i}"] = (f'=IF(L{i}<2,"",SQRT(SUMPRODUCT({tich(f"$A{i}", f"$B{i}", f"$E{i}", f"$F{i}")}'
                           f'*({rI}-I{i})^2)/(L{i}-1)))')
            th[f"N{i}"] = f'=IF(L{i}=0,"",_xlfn.MINIFS({rI},{dk}))'
            th[f"O{i}"] = f'=IF(L{i}=0,"",_xlfn.MAXIFS({rI},{dk}))'
            th[f"P{i}"] = "  ".join(seed_txt[k])
        if (ds, p_, s_) != truoc:
            soc, truoc = not soc, (ds, p_, s_)
        for j in range(1, len(cot) + 1):
            col = L(j)
            c = th[f"{col}{i}"]
            nho = cot[j - 1][0] == "điểm từng seed"
            c.font = Font(name=F, size=9 if nho else 10, color="808080" if nho else "000000",
                          bold=la_moc(ten, cd) or col == "I")
            c.alignment, c.border = (Lt if col == "E" or nho else C), BOX
            c.fill = MOC if la_moc(ten, cd) else (SOC if soc else PatternFill())
        th[f"I{i}"].number_format = "0.00"
        for col in "JK":
            th[f"{col}{i}"].number_format = DELTA
        if not PS:
            for col in "MNO":
                th[f"{col}{i}"].number_format = "0.00"
    m = len(keys) + 1
    to_mau(th, f"J2:K{m}")
    for j, (_, w) in enumerate(cot, 1):
        th.column_dimensions[L(j)].width = w
    th.freeze_panes = "F2"
    th.auto_filter.ref = f"A1:{L(len(cot))}{m}"

    # ============================ bang_rong
    br = wb.create_sheet("bang_rong")
    dss = [d for d in a.datasets if any(r["dataset"] == d for r in rows)]
    con = ["điểm", "Δ Sq"] if PS else ["mean", "std", "n", "Δ Sq"]
    W = len(con)
    tieu_de = ["mức nén %", "seed", "hạng", "phương pháp", "chế độ", "β", "γ"]
    for d in dss:
        t = TEN_DS.get(d, d)
        tieu_de += [f"{t} {con[0]}"] + con[1:]
    tieu_de += ["TB các tập", "Δ Sq (TB)"]
    hdr(br, tieu_de)
    cfgs = defaultdict(list)
    for (ds, p_, s_, ten, cd), v in val_py.items():
        cfgs[(p_, s_, ten, cd)].append(v)
    hang = sorted(cfgs, key=lambda k: (k[0], sk_ord(k[1]), -sum(cfgs[k]) / len(cfgs[k]), k[2], k[3]))
    c0 = 8
    cTB, cDTB = L(c0 + W * len(dss)), L(c0 + W * len(dss) + 1)
    soc, truoc = False, None
    for i, (p_, s_, ten, cd) in enumerate(hang, 2):
        b, g = info[(ten, cd)]
        for j, v in enumerate([p_, s_ if PS or s_ == "all" else " ".join(a.seeds),
                               None, ten, cd, bf(b), bf(g)], 1):
            br.cell(row=i, column=j, value=v)
        means = []
        for j, d in enumerate(dss):
            cols = [L(c0 + W * j + o) for o in range(W)]
            cm, cdl = cols[0], cols[-1]
            dk = tieu_chi(f'"{d}"', f"$A{i}", f"$D{i}", f"$E{i}", f"$B{i}")
            moc = tieu_chi(f'"{d}"', f"$A{i}", '"Squeezed"', '"threshold"', f"$B{i}")
            br[f"{cm}{i}"] = f'=IFERROR(AVERAGEIFS({rI},{dk}),"")'
            br[f"{cdl}{i}"] = (f'=IF(OR($A{i}=0,{cm}{i}="",COUNTIFS({moc})=0),"",'
                               f'{cm}{i}-AVERAGEIFS({rI},{moc}))')
            if not PS:
                cs, cn = cols[1], cols[2]
                br[f"{cn}{i}"] = f"=COUNTIFS({dk})"
                br[f"{cs}{i}"] = (f'=IF({cn}{i}<2,"",SQRT(SUMPRODUCT('
                                  f'{tich(chr(34) + d + chr(34), f"$A{i}", f"$D{i}", f"$E{i}")}'
                                  f'*({rI}-{cm}{i})^2)/({cn}{i}-1)))')
                br[f"{cs}{i}"].number_format = "0.00"
            br[f"{cm}{i}"].number_format = "0.00"
            br[f"{cdl}{i}"].number_format = DELTA
            to_mau(br, f"{cdl}{i}")
            means.append(f"{cm}{i}")
        br[f"{cTB}{i}"] = f'=IFERROR(AVERAGE({",".join(means)}),"")'
        br[f"{cTB}{i}"].number_format = "0.00"
        br[f"{cDTB}{i}"] = (f'=IF(OR($A{i}=0,{cTB}{i}="",COUNTIFS($A:$A,$A{i},$B:$B,$B{i},'
                            f'$D:$D,"Squeezed",$E:$E,"threshold")=0),"",{cTB}{i}-SUMIFS('
                            f'${cTB}:${cTB},$A:$A,$A{i},$B:$B,$B{i},$D:$D,"Squeezed",$E:$E,"threshold"))')
        br[f"{cDTB}{i}"].number_format = DELTA
        br[f"C{i}"] = (f'=IF({cTB}{i}="","",COUNTIFS($A:$A,A{i},$B:$B,B{i},'
                       f'${cTB}:${cTB},">"&{cTB}{i})+1)')
        if (p_, s_) != truoc:
            soc, truoc = not soc, (p_, s_)
        dam = [L(c0 + W * j) for j in range(len(dss))] + [cTB]
        for c in range(1, c0 + W * len(dss) + 2):
            cell = br.cell(row=i, column=c)
            cell.font = Font(name=F, bold=la_moc(ten, cd) or L(c) in dam)
            cell.alignment, cell.border = (Lt if c == 4 else C), BOX
            cell.fill = MOC if la_moc(ten, cd) else (SOC if soc else PatternFill())
    to_mau(br, f"{cDTB}2:{cDTB}{len(hang) + 1}")
    for k_, w in zip("ABCDEFG", [10, 7, 6, 22, 11, 7, 7]):
        br.column_dimensions[k_].width = w
    rong = [12, 8] if PS else [12, 7, 5, 8]
    for j in range(len(dss)):
        for o, w in enumerate(rong):
            br.column_dimensions[L(c0 + W * j + o)].width = w
    br.column_dimensions[cTB].width = 11
    br.column_dimensions[cDTB].width = 10
    br.freeze_panes = "E2"
    br.auto_filter.ref = f"A1:{cDTB}{len(hang) + 1}"

    # ============================ cau_hinh
    ch = wb.create_sheet("cau_hinh")
    for k, v in [("pred_root", os.path.abspath(a.pred_root)), ("model", a.model),
                 ("percent_clusters", a.pc), ("datasets", " ".join(a.datasets)),
                 ("seeds", " ".join(a.seeds)),
                 ("gop_seed", "True (mean qua cac seed)" if a.gop_seed
                  else "False (moi seed rieng, so voi Squeezed cung seed)"),
                 ("muc_nen", " ".join(map(str, a.muc_nen)) or "tat ca"),
                 ("che_do", " ".join(a.che_do)), ("phuong_phap", " ".join(a.phuong_phap)),
                 ("beta", " ".join(a.beta) or "tat ca"),
                 ("gamma", " ".join(a.gamma) or "tat ca"),
                 ("gom_lim", a.gom_lim), ("so lan chay", len(rows)), ("", ""),
                 ("Cach doc", "Diem = qa_f1 x100 tinh lai tu .jsonl (logic LongBench). "
                  "Moi nhom = (moc nen, seed) khi gop_seed=False, hoac moc nen khi True; "
                  "trong nhom, cau hinh xep tu cao xuong thap; 'hang' = thu hang "
                  "trong nhom (bang_rong: theo trung binh cac tap). All KV khong "
                  "phu thuoc seed nen seed = all. "
                  "Δ vs Squeezed = so voi Squeezed (threshold) cung dataset va moc nen; "
                  "chi_tiet co them Δ ghep cap theo tung seed. std(seed) chia n-1. "
                  "Chu xanh trong chi_tiet = so lieu goc; moi so con lai la cong thuc."),
                 ("Nen vang", "Moc so sanh: Squeezed (threshold)")]:
        ch.append([k, str(v)])
    if canh_bao:
        ch.append([])
        ch.append(["CANH BAO"])
        for c in canh_bao:
            ch.append(["", c])
    if bo_qua:
        ch.append([])
        ch.append(["Thu muc bo qua (_vb, -lim, ten khac mau)"])
        for c in bo_qua:
            ch.append(["", c])
    for row in ch.iter_rows():
        for c in row:
            c.font = Font(name=F, bold=(c.column == 1))
            c.alignment = Alignment(wrap_text=True, vertical="top")
    ch.column_dimensions["A"].width = 26
    ch.column_dimensions["B"].width = 110

    wb.move_sheet("bang_rong", offset=-2)
    wb.move_sheet("tong_hop", offset=-1)
    wb.active = 0
    wb.save(a.out)
    print("Da ghi:", os.path.abspath(a.out))


# ------------------------------------------------------------------ main
def main():
    H = CAU_HINH
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred_root", default=H["pred_root"] or os.environ.get("ROOT", _ROOT_MAC_DINH))
    for k in ("datasets", "seeds", "muc_nen", "che_do", "phuong_phap", "beta",
              "gamma", "model", "out"):
        ap.add_argument(f"--{k}", default=H[k])
    ap.add_argument("--pc", type=int, default=H["pc"])
    ap.add_argument("--gom_lim", action="store_true", default=H["gom_lim"])
    ap.add_argument("--gop_seed", action="store_true", default=H["gop_seed"],
                    help="gop cac seed thanh mean ± std (mac dinh: moi seed rieng)")
    a = ap.parse_args()
    a.datasets, a.seeds = a.datasets.split(), a.seeds.split()
    a.muc_nen = [int(x) for x in a.muc_nen.split()]
    a.che_do, a.phuong_phap = a.che_do.split(), a.phuong_phap.split()
    a.beta, a.gamma = a.beta.split(), a.gamma.split()
    sai = [c for c in a.che_do if c not in TAG.values()]
    if sai:
        sys.exit(f"!! che_do khong hop le: {sai} (chi co threshold, budget)")

    if not os.path.isdir(a.pred_root):
        sys.exit(f"!! Khong thay pred_root: {os.path.abspath(a.pred_root)}\n"
                 f"   Sua CAU_HINH['pred_root'] hoac dat bien moi truong ROOT.")
    print(f"[cau hinh] datasets={a.datasets} seeds={a.seeds} muc_nen={a.muc_nen or 'tat ca'} "
          f"gop_seed={a.gop_seed} che_do={a.che_do} phuong_phap={a.phuong_phap} beta={a.beta or 'tat ca'} "
          f"gamma={a.gamma or 'tat ca'}")
    rows, canh_bao, bo_qua = quet(a)
    for c in canh_bao:
        print("!!", c)
    if not rows:
        dirs = sorted({os.path.basename(p) for p in glob.glob(os.path.join(a.pred_root, "pred_*"))})
        sys.exit("!! Khong tim thay ket qua nao. Thu muc pred_* hien co:\n   "
                 + "\n   ".join(dirs[:40]))
    tim = defaultdict(set)
    for r in rows:
        tim[(r["phuong_phap"], r["che_do"])].add(str(r["seed"]))
    for (pp, cd), ss in sorted(tim.items()):
        print(f"   {pp:<22} {cd:<10} seed: {' '.join(sorted(ss))}")
    print(f"Tim thay {len(rows)} lan chay "
          f"({len({(r['phuong_phap'], r['che_do']) for r in rows})} cau hinh, "
          f"{len({r['dataset'] for r in rows})} dataset).")
    ghi_xlsx(a, rows, canh_bao, bo_qua)


if __name__ == "__main__":
    main()
