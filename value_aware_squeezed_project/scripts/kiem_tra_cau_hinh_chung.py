#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kiem_tra_cau_hinh_chung.py — Tim MOT cap (β, γ) dung chung lam CA 3 tap tot hon phuong phap goc.

Hai che do chon cum, moi che do mot moc:
  threshold -> moc Squeezed  (pred_s<seed>_thr)
  budget    -> moc ClusterKV (pred_s<seed>_bgt)
Voi moi (β, γ), moi dataset, moi muc nen: MEAN va STD qua cac seed (qa_f1 × 100).
Mot o "thang" neu mean cua cau hinh > mean cua moc cung o. "DAT" = thang moi o.

HAI KHOI TAP (moi khoi 3 tap × 3 muc nen):
  VAL  (tap_val)  -> dung de CHON (β, γ): qasper_val, multifieldqa_en_val, narrativeqa_val
  TEST (tap_test) -> dung de BAO CAO:      qasper, multifieldqa_en_test, narrativeqa
  (multifieldqa_en_test = 100 cau con lai sau khi tach 50 cau dev; KHONG dung multifieldqa_en
   150 cau o khoi test vi no chua ca cau dev.)  Dat tap_val = "" -> chi con khoi test nhu cu.

THU TU: cao xuong thap theo VAL — so o val thang moc giam dan, roi muc vuot trung binh
        tren val (mean cau hinh − mean moc) giam dan; hoa thi xet tiep test.

XLSX <out>, moi che do mot bo sheet:
  <che_do>_beta_<β>   : β = "beta_co_dinh", moi γ mot dong
  <che_do>_xep_hang   : moi (β, γ)
  chi_tiet            : diem tung seed, mean, std, moc
  bo_qua, ghi_chu
Hang xam dau bang = moc. O mean: xanh = cao hon moc, do = khong cao hon. Cot std: chu xam.

Chi tinh cau hinh du moi seed trong "seeds" va file du cau (tru khi --cho_thieu_seed).
Can xuat_ket_qua.py CUNG thu muc.
CACH DUNG: sua CAU_HINH roi chay `python3 kiem_tra_cau_hinh_chung.py --muc_nen "70 80 90"`
"""
import argparse
import os
import statistics as st
import sys
from types import SimpleNamespace

# ═══════════════════════════════════════════════════════════════════════════
#  CAU HINH
# ═══════════════════════════════════════════════════════════════════════════
CAU_HINH = {
    # None = $ROOT, roi ../../SqueezedAttention-main/LongBench (tinh tu thu muc script)
    "pred_root":     None,
    "tap_val":       "qasper_val multifieldqa_en_val narrativeqa_val",
    "tap_test":      "qasper multifieldqa_en_test narrativeqa",
    "muc_nen":       "70 80 90",
    "seeds":         "0 1 2",
    "che_do":        "threshold budget",
    "beta_co_dinh":  "0.05",
    "cho_thieu_seed": False,     # True = van tinh cau hinh thieu seed
    "model":         "LLaMA-2-7B-32K",
    "pc":            5,
    "out":           "kiem_tra_cau_hinh_chung.xlsx",
}
_ROOT_MAC_DINH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "../../SqueezedAttention-main/LongBench")
TEN_DS = {"qasper": "Qasper", "multifieldqa_en": "MFQA-en", "narrativeqa": "NarrativeQA",
          "qasper_val": "Qasper val", "narrativeqa_val": "NarrQA val",
          "multifieldqa_en_val": "MFQA val", "multifieldqa_en_test": "MFQA test"}
MOC = {"threshold": "Squeezed", "budget": "ClusterKV"}


def o_cua(a, tap):
    """Danh sach o (ds, p) cua mot khoi: 'val' | 'test' | None (= ca hai)."""
    L = {"val": a.tap_val, "test": a.tap_test}
    khoi = [tap] if tap else [t for t in ("val", "test") if L[t]]
    return [(ds, p) for t in khoi for p in a.muc_nen for ds in L[t]]


def nhan_o(a, ds, p):
    goc = ds[:-4] if ds.endswith("_val") else ds[:-5] if ds.endswith("_test") else ds
    return f"{'VAL' if ds in a.tap_val else 'TEST'} {TEN_DS.get(goc, goc)} {p}%"


def _so(x):
    return f"{float(x):g}"


def _sd(xs):
    return st.stdev(xs) if len(xs) > 1 else 0.0


# ------------------------------------------------------------------ du lieu
def thu_thap(a):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        import xuat_ket_qua as X
    except ImportError:
        sys.exit("!! Can xuat_ket_qua.py nam cung thu muc")
    q = SimpleNamespace(pred_root=a.pred_root, model=a.model, pc=a.pc,
                        datasets=a.tap_val + a.tap_test,
                        seeds=a.seeds, muc_nen=a.muc_nen, che_do=a.che_do,
                        phuong_phap=["sq", "ck", "e1"], beta=[], gamma=[], gom_lim=False)
    rows, canh_bao, _ = X.quet(q)
    for c in canh_bao:
        print("!!", c)
    # D[cd][(ds, p)][khoa][seed] = (diem, so_cau); khoa = "moc" | (β, γ)
    D = {}
    for r in rows:
        cd = r["che_do"]
        if cd not in a.che_do or "[lim]" in r["phuong_phap"]:
            continue
        k = "moc" if r["beta"] == "" else (float(r["beta"]), float(r["gamma"]))
        D.setdefault(cd, {}).setdefault((r["dataset"], r["muc_nen"]), {}) \
            .setdefault(k, {})[str(r["seed"])] = (r["diem"], r["so_cau"])
    return D


def tinh(a, Dc, cd):
    """-> MOC_[(ds,p)] = dict(mean, sd, diem) ; T[(β,γ)][(ds,p)] = dict(...) ; bo_qua"""
    MOC_, T, bo_qua = {}, {}, []
    for (ds, p), G in Dc.items():
        if "moc" not in G:
            continue
        n_max = max(n for v in G.values() for _, n in v.values())
        mc = {s: d for s, (d, n) in G["moc"].items() if n == n_max and s in a.seeds}
        if len(mc) < len(a.seeds) and not a.cho_thieu_seed:
            bo_qua.append(dict(che_do=cd, cau_hinh=MOC[cd], dataset=ds, muc_nen=p,
                               ly_do="mốc thiếu seed / chưa đủ câu -> bỏ cả ô"))
            continue
        if not mc:
            continue
        MOC_[(ds, p)] = dict(mean=st.mean(mc.values()), sd=_sd(list(mc.values())), diem=mc)
        for k, v in G.items():
            if k == "moc":                       # β=0, γ=0 (E1 tai lap Squeezed) van giu lai
                continue
            ok = {s: d for s, (d, n) in v.items() if n == n_max and s in a.seeds}
            thieu = [s for s in a.seeds if s not in ok]
            if thieu and not a.cho_thieu_seed:
                bo_qua.append(dict(che_do=cd, cau_hinh=f"β={_so(k[0])}, γ={_so(k[1])}",
                                   dataset=ds, muc_nen=p,
                                   ly_do=f"thiếu / chưa đủ câu: seed {' '.join(thieu)}"))
                continue
            if not ok:
                continue
            xs = list(ok.values())
            T.setdefault(k, {})[(ds, p)] = dict(
                mean=st.mean(xs), sd=_sd(xs), diem=ok, n_seed=len(ok), thieu=thieu,
                moc=MOC_[(ds, p)]["mean"],
                thang_seed=sum(ok[s] > mc[s] for s in ok if s in mc))
    return MOC_, T, bo_qua


def xep_hang(a, T, keys=None):
    """Cao -> thap theo khoi chon (val neu co, khong thi test): so o thang moc, roi muc vuot
    trung binh; hoa thi xet khoi test."""
    khoi = [t for t in ("val", "test") if (a.tap_val if t == "val" else a.tap_test)]
    ra = []
    for k in (keys if keys is not None else T):
        r = dict(k=k, o=[T[k].get(c) for c in o_cua(a, None)])
        for t in khoi:
            o = [T[k].get(c) for c in o_cua(a, t)]
            co = [x for x in o if x]
            thang = sum(x["mean"] > x["moc"] for x in co)
            r[t] = dict(co=len(co), tong=len(o), thang=thang,
                        vuot=st.mean(x["mean"] - x["moc"] for x in co) if co else -1e9,
                        dat=len(co) == len(o) and thang == len(o))
        ra.append(r)
    ra.sort(key=lambda r: tuple(v for t in khoi for v in (-r[t]["thang"], -r[t]["vuot"])))
    return ra, khoi


# ------------------------------------------------------------------ man hinh
def _o(x, w, moc=None):
    if x is None:
        return "—".rjust(w)
    sao = "*" if moc is not None and x["mean"] > moc else " "
    return f"{x['mean']:.2f}±{x['sd']:.2f}{sao}".rjust(w)


def in_bang(a, cd, MOC_, R_beta, R_all, khoi):
    o_list = o_cua(a, None)
    nhan = [nhan_o(a, ds, p) for ds, p in o_list]
    w = max(15, max(len(x) for x in nhan) + 2)
    dau = f"{'':>15}" + "".join(n.rjust(w) for n in nhan) + "  " + "  ".join(
        f"thang moc ({t})" for t in khoi)
    moc = f"{MOC[cd]:>15}" + "".join(_o(MOC_.get(c), w) for c in o_list)

    def dong(r, ten):
        kq = []
        for t in khoi:
            x = r[t]
            s = f"{t} {x['thang']}/{x['tong']}"
            if x["co"] < x["tong"]:
                s += f" (thieu {x['tong'] - x['co']} o)"
            elif x["dat"]:
                s += " DAT"
            kq.append(s)
        return (f"{ten:>15}" + "".join(_o(x, w, x["moc"] if x else None) for x in r["o"])
                + "  " + " | ".join(kq))

    print(f"\n######## CHE DO {cd.upper()} — moc {MOC[cd]} (mean±std qua seed; * = cao hon moc) ########")
    print(f"\n--- β = {_so(a.beta_co_dinh)} ---")
    print(dau)
    print(moc)
    for r in R_beta:
        print(dong(r, f"γ={_so(r['k'][1])}"))
    print("\n--- moi (β, γ), cao -> thap ---")
    print(dau)
    print(moc)
    for r in R_all:
        print(dong(r, f"β={_so(r['k'][0])} γ={_so(r['k'][1])}"))
    for t in khoi:
        dat = [r for r in R_all if r[t]["dat"]]
        print(f"\n-> [{t}] {len(dat)} cap thang {MOC[cd]} o MOI o"
              + (": " + ", ".join(f"β={_so(r['k'][0])} γ={_so(r['k'][1])}" for r in dat)
                 if dat else "."))


# ------------------------------------------------------------------ xlsx
def ghi(a, KQ, bo_qua):
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.formatting.rule import FormulaRule
    from openpyxl.utils import get_column_letter as L
    F = "Arial"
    vien = Side(style="thin", color="BFBFBF")
    khung = Border(left=vien, right=vien, top=vien, bottom=vien)
    HDR = PatternFill("solid", fgColor="1F3864")
    XAM = PatternFill("solid", fgColor="F2F2F2")
    XANH = PatternFill("solid", fgColor="E2EFDA")
    o_list = o_cua(a, None)
    n = len(o_list)

    def o_tieu_de(c, text):
        if text is not None:
            c.value = text
        c.font = Font(name=F, bold=True, color="FFFFFF")
        c.fill = HDR
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        c.border = khung

    def bang(ws, cd, MOC_, R, cot_dau, khoi):
        """cot_dau: list ten cot nhan (vd ['β','γ']). Hang 1-2 tieu de, hang 3 = moc.
        Cot mean/std: khoi VAL roi khoi TEST; cuoi bang moi khoi 2 cot (so o thang, ket qua)."""
        c0 = len(cot_dau) + 1
        cot_mean = [c0 + 2 * i for i in range(n)]
        cot_khoi = {t: [cot_mean[o_list.index(c)] for c in o_cua(a, t)] for t in khoi}
        c_kq_t = {t: (c0 + 2 * n + 2 * j, c0 + 2 * n + 2 * j + 1) for j, t in enumerate(khoi)}
        c_kq = c0 + 2 * n + 2 * len(khoi) - 1      # cot cuoi
        gop = []                                   # tao kieu truoc, gop o sau
        for j, t in enumerate(cot_dau, 1):
            o_tieu_de(ws.cell(1, j), t)
            o_tieu_de(ws.cell(2, j), None)
            gop.append((1, j, 2, j))
        for i, (ds, p) in enumerate(o_list):
            c = cot_mean[i]
            o_tieu_de(ws.cell(1, c), nhan_o(a, ds, p))
            o_tieu_de(ws.cell(1, c + 1), None)
            o_tieu_de(ws.cell(2, c), "mean")
            o_tieu_de(ws.cell(2, c + 1), "std")
            gop.append((1, c, 1, c + 1))
        for t in khoi:
            for c, s in ((c_kq_t[t][0], f"Số ô >\n{MOC[cd]} ({t})"),
                         (c_kq_t[t][1], f"Kết quả\n({t})")):
                o_tieu_de(ws.cell(1, c), s)
                o_tieu_de(ws.cell(2, c), None)
                gop.append((1, c, 2, c))
        for r1, c1, r2, c2 in gop:
            ws.merge_cells(start_row=r1, start_column=c1, end_row=r2, end_column=c2)
        ws.row_dimensions[1].height = 30

        # hang 3: moc
        ws.cell(3, 1, MOC[cd])
        for i, cc in enumerate(o_list):
            m = MOC_.get(cc)
            if m:
                ws.cell(3, cot_mean[i], round(m["mean"], 4))
                ws.cell(3, cot_mean[i] + 1, round(m["sd"], 4))
        for j in range(1, c_kq + 1):
            c = ws.cell(3, j)
            c.font = Font(name=F, bold=True)
            c.fill = XAM
            c.border = khung
            c.alignment = Alignment(horizontal="center")

        for r in R:
            i = ws.max_row + 1
            nhan = [r["k"][0], r["k"][1]] if len(cot_dau) == 2 else [r["k"][1]]
            for j, v in enumerate(nhan, 1):
                ws.cell(i, j, v)
            for t, x in enumerate(r["o"]):
                if x:
                    ws.cell(i, cot_mean[t], round(x["mean"], 4))
                    ws.cell(i, cot_mean[t] + 1, round(x["sd"], 4))
            for t in khoi:
                cs, m = cot_khoi[t], len(cot_khoi[t])
                thang = "+".join(f"(N({L(c)}{i})>{L(c)}$3)*ISNUMBER({L(c)}{i})" for c in cs)
                co = "+".join(f"ISNUMBER({L(c)}{i})" for c in cs)
                ws.cell(i, c_kq_t[t][0], f"=({thang})&\"/{m}\"")
                ws.cell(i, c_kq_t[t][1], f"=IF(({co})<{m},\"thiếu dữ liệu\","
                                         f"IF(({thang})={m},\"ĐẠT\",\"không\"))")
            for j in range(1, c_kq + 1):
                c = ws.cell(i, j)
                c.border = khung
                c.alignment = Alignment(horizontal="center")
                c.font = Font(name=F)
        last = max(ws.max_row, 4)
        for c in cot_mean:
            cl = L(c)
            ws.conditional_formatting.add(
                f"{cl}4:{cl}{last}",
                FormulaRule(formula=[f"AND(ISNUMBER({cl}4),{cl}4>{cl}$3)"],
                            font=Font(name=F, color="2E7D32", bold=True), fill=XANH))
            ws.conditional_formatting.add(
                f"{cl}4:{cl}{last}",
                FormulaRule(formula=[f"AND(ISNUMBER({cl}4),{cl}4<={cl}$3)"],
                            font=Font(name=F, color="C62828")))
            for i in range(3, last + 1):
                ws.cell(i, c).number_format = "0.00"
                s = ws.cell(i, c + 1)
                s.number_format = "0.00"
                s.font = Font(name=F, color="808080", size=9, bold=(i == 3))
        for j in range(1, c0):
            ws.column_dimensions[L(j)].width = 8
        for c in cot_mean:
            ws.column_dimensions[L(c)].width = 9
            ws.column_dimensions[L(c + 1)].width = 6.5
        for t in khoi:
            ws.column_dimensions[L(c_kq_t[t][0])].width = 11
            ws.column_dimensions[L(c_kq_t[t][1])].width = 14
        ws.freeze_panes = ws.cell(4, c0)

    wb = Workbook()
    wb.remove(wb.active)
    for cd, (MOC_, T, R_beta, R_all, khoi) in KQ.items():
        ws = wb.create_sheet(f"{cd}_beta_{_so(a.beta_co_dinh)}")
        bang(ws, cd, MOC_, R_beta, [f"γ (β={_so(a.beta_co_dinh)})"], khoi)
        ws.column_dimensions["A"].width = 13
        bang(wb.create_sheet(f"{cd}_xep_hang"), cd, MOC_, R_all, ["β", "γ"], khoi)

    ws3 = wb.create_sheet("chi_tiet")
    cot = (["Chế độ", "β", "γ", "Tập", "Dataset", "Mức nén (%)"] + [f"Seed {s}" for s in a.seeds] +
           ["Mean", "Std", "Mốc (mean)", "Seed cao hơn mốc"])
    ws3.append(cot)
    for j in range(1, len(cot) + 1):
        o_tieu_de(ws3.cell(1, j), cot[j - 1])
    ns = len(a.seeds)
    for cd, (MOC_, T, _, _, _) in KQ.items():
        for k in sorted(T):
            for (ds, p), h in sorted(T[k].items(), key=lambda kv: (kv[0][0] not in a.tap_val,
                                                                   kv[0])):
                ws3.append([cd, k[0], k[1], "val" if ds in a.tap_val else "test",
                            TEN_DS.get(ds, ds), p] +
                           [round(h["diem"][s], 4) if s in h["diem"] else None for s in a.seeds])
                i = ws3.max_row
                rg = f"{L(7)}{i}:{L(6 + ns)}{i}"
                ws3.cell(i, 7 + ns, f"=AVERAGE({rg})")
                ws3.cell(i, 8 + ns, f"=IF(COUNT({rg})>1,STDEV({rg}),0)")
                ws3.cell(i, 9 + ns, round(h["moc"], 4))
                ws3.cell(i, 10 + ns, f"{h['thang_seed']}/{h['n_seed']}")
                for j in range(1, 11 + ns):
                    c = ws3.cell(i, j)
                    c.font = Font(name=F, color="0000FF") if 7 <= j < 7 + ns else Font(name=F)
                    c.border = khung
                    c.alignment = Alignment(horizontal="center")
                    if 7 <= j <= 9 + ns:
                        c.number_format = "0.00"
    for j, w in enumerate([10, 7, 7, 6, 13, 10] + [9] * ns + [9, 8, 10, 12], 1):
        ws3.column_dimensions[L(j)].width = w
    ws3.freeze_panes = "G2"

    ws4 = wb.create_sheet("bo_qua")
    cot = ["Chế độ", "Cấu hình", "Dataset", "Mức nén (%)", "Lý do"]
    ws4.append(cot)
    for j in range(1, 6):
        o_tieu_de(ws4.cell(1, j), cot[j - 1])
    for r in bo_qua:
        ws4.append([r["che_do"], r["cau_hinh"], TEN_DS.get(r["dataset"], r["dataset"]),
                    r["muc_nen"], r["ly_do"]])
        for c in ws4[ws4.max_row]:
            c.font = Font(name=F)
            c.border = khung
    for j, w in enumerate([10, 18, 13, 10, 42], 1):
        ws4.column_dimensions[L(j)].width = w

    ws5 = wb.create_sheet("ghi_chu")
    for dong in [
        ["Giá trị", "Mean = qa_f1 × 100 trung bình qua seed " + ", ".join(a.seeds) +
                    "; std = độ lệch chuẩn mẫu qua các seed (cùng mức nén, cùng tập)."],
        ["Mốc", "threshold -> Squeezed (pred_s<seed>_thr); budget -> ClusterKV (pred_s<seed>_bgt). "
                "Hàng xám đầu mỗi bảng."],
        ["Màu", "Ô mean xanh = cao hơn mốc cùng cột; đỏ = không cao hơn. Cột std chữ xám."],
        ["Tập", "VAL = " + ", ".join(a.tap_val or ["(không dùng)"]) + " — để CHỌN (β, γ). "
                "TEST = " + ", ".join(a.tap_test) + " — để BÁO CÁO. multifieldqa_en_test là 100 câu "
                "còn lại sau khi tách 50 câu dev (tach_mfqa.py)."],
        ["Thứ tự", "Cao xuống thấp theo VAL: số ô val thắng mốc giảm dần, rồi mức vượt mốc trung "
                   "bình trên val; hoà thì xét test. Cấu hình chưa có val xếp cuối."],
        ["ĐẠT", "Cao hơn mốc ở MỌI ô của khối đó (3 tập × 3 mức nén), tính riêng val và test."],
        ["chi_tiet", "Điểm từng seed (chữ xanh dương, đọc từ pred_*.jsonl); Mean/Std là công thức."],
        ["Lưu ý", "Chỉ cột test là kết quả báo cáo được; nên chọn (β, γ) theo val rồi đọc test của "
                  "đúng cấu hình đó, và nêu số cấu hình đã thử cùng std giữa các seed."],
        ["Nguồn", os.path.abspath(a.pred_root)],
    ]:
        ws5.append(dong)
        for c in ws5[ws5.max_row]:
            c.font = Font(name=F)
            c.alignment = Alignment(wrap_text=True, vertical="top")
    ws5.column_dimensions["A"].width = 12
    ws5.column_dimensions["B"].width = 100
    wb.save(a.out)
    print("\nDa ghi:", os.path.abspath(a.out))


# ------------------------------------------------------------------ main
def main():
    H = CAU_HINH
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred_root", default=H["pred_root"] or os.environ.get("ROOT", _ROOT_MAC_DINH))
    for k in ("tap_val", "tap_test", "muc_nen", "seeds", "che_do", "beta_co_dinh", "model", "out"):
        ap.add_argument(f"--{k}", default=H[k])
    ap.add_argument("--pc", type=int, default=H["pc"])
    ap.add_argument("--cho_thieu_seed", action="store_true", default=H["cho_thieu_seed"])
    a = ap.parse_args()
    a.tap_val, a.tap_test = a.tap_val.split(), a.tap_test.split()
    a.seeds, a.che_do = a.seeds.split(), a.che_do.split()
    a.muc_nen = [int(x) for x in a.muc_nen.split()]
    a.beta_co_dinh = float(a.beta_co_dinh)
    print(f"[cau hinh] pred_root = {os.path.abspath(a.pred_root)}")
    if not os.path.isdir(a.pred_root):
        sys.exit("!! Khong thay pred_root. Dat --pred_root hoac export ROOT=.../LongBench")
    print(f"[cau hinh] val={a.tap_val} test={a.tap_test} muc_nen={a.muc_nen} "
          f"seeds={a.seeds} che_do={a.che_do}")
    D = thu_thap(a)
    KQ, bo_qua = {}, []
    for cd in a.che_do:
        MOC_, T, bq = tinh(a, D.get(cd, {}), cd)
        bo_qua += bq
        if not T:
            print(f"\n!! {cd}: khong co cau hinh E1 nao du du lieu"
                  f"{'' if D.get(cd) else ' (khong tim thay pred_*' + ('_thr' if cd == 'threshold' else '_bgt') + ' nao)'}.")
            continue
        R_beta, khoi = xep_hang(a, T, [k for k in T if abs(k[0] - a.beta_co_dinh) < 1e-9])
        R_all, _ = xep_hang(a, T)
        in_bang(a, cd, MOC_, R_beta, R_all, khoi)
        KQ[cd] = (MOC_, T, R_beta, R_all, khoi)
    if not KQ:
        sys.exit("\n!! Khong co du lieu. Kiem tra pred_root (phai la thu muc .../LongBench).")
    ghi(a, KQ, bo_qua)


if __name__ == "__main__":
    main()
