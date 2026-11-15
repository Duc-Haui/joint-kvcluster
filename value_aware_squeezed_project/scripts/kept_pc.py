#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kept_pc.py — trich TY LE TOKEN THUC SU DUOC GIU (kept%) tu log SQDBG.

KHONG CAN GPU. Chi doc logs_E1/*.log da co.

VI SAO PHAI CO SO NAY
    Moi so lieu ve chat luong (F1, token lap) deu duoc gan nhan "nen 70%".
    Nhan do den tu 'target'. Nhung dong log cho thay:

        target=0.300  kept=0.8957

    nghia la dat muc tieu giu 30% nhung THUC TE giu 89.6%. Neu dung vay,
    cot "70% pruning" trong bai cua ban khong phai nen 70% — va moi so
    sanh giua cac phuong phap deu phai doc lai theo kept% THAT, chu khong
    theo target.

    Quan trong hon: neu cau hinh value-aware giu NHIEU hon moc goc o cung
    mot target, thi viec no it lap hon / diem cao hon co the chi vi no NEN
    IT HON. Script nay do dung dieu do.

DONG LOG DUOC DOC
    [SQDBG] E1 <ds> s<seed> p<nen> L<lop> mode=<thr|budget> target=<x>
            kept=<y> (<y>%) ratio=<r>x T=<t> N=<n> va=g=<gamma>
    beta/gamma lay tu TEN FILE: clustering_<ds>_b<beta>_g<gamma>_s<seed>.log
    File khong co _b.._g.. -> coi la moc goc (β=0, γ=0).

CACH DUNG
    python3 kept_pc.py --logs logs_E1
    python3 kept_pc.py --logs logs_E1 --moi_lop      # tach theo tung lop
    python3 kept_pc.py --logs logs_E1 --xem          # in thu 5 dong khop
"""
import argparse
import glob
import os
import re

DONG = re.compile(
    r"\[SQDBG\].*?\b(?P<ds>[a-z_0-9]+)\s+s(?P<seed>\d+)\s+"
    r"p(?P<nen>[0-9.]+)\s+L(?P<lop>\d+)\s+"
    r"mode=(?P<mode>\w+)\s+target=(?P<target>[0-9.eE+-]+)\s+"
    r"kept=(?P<kept>[0-9.eE+-]+)")
PHU = re.compile(r"ratio=(?P<ratio>[0-9.eE+-]+)x|N=(?P<N>\d+)|"
                 r"va=g=(?P<g>[0-9.eE+-]+)")
TEN = re.compile(r"_b(?P<b>[0-9.eE+-]+)_g(?P<g>[0-9.eE+-]+)_s(?P<s>\d+)")


def tb(xs):
    if not xs:
        return None, None
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    return m, (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def quet(a):
    """-> list ban ghi tung dong log."""
    files = sorted(glob.glob(os.path.join(a.logs, "*.log")))
    if not files:
        raise SystemExit(f"!! Khong thay file .log nao trong {a.logs}")
    ra, bo_qua, mau = [], 0, []
    for f in files:
        ten = os.path.basename(f)
        if not a.ca_lim and ("-lim" in ten or "_vb" in ten):
            bo_qua += 1
            continue
        m = TEN.search(ten)
        beta = float(m.group("b")) if m else 0.0
        gamma_ten = float(m.group("g")) if m else 0.0
        # run_joint_kvcluster.sh gan SUFFIX="-placebo" vao sau seed khi SQ_PLACEBO=1.
        # Hai nhanh phai tach rieng, neu khong chung bi gop thanh mot dong
        # va kept% trung binh tro nen vo nghia.
        pla = "-placebo" in ten
        with open(f, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                d = DONG.search(line)
                if not d:
                    continue
                if len(mau) < 5:
                    mau.append(line.rstrip())
                phu = {}
                for p in PHU.finditer(line):
                    phu.update({k: v for k, v in p.groupdict().items()
                                if v is not None})
                ra.append({
                    "file": ten, "ds": d.group("ds"),
                    "seed": int(d.group("seed")),
                    "nen": float(d.group("nen")), "lop": int(d.group("lop")),
                    "mode": d.group("mode"),
                    "target": float(d.group("target")),
                    "kept": float(d.group("kept")),
                    "ratio": float(phu["ratio"]) if "ratio" in phu else None,
                    "N": int(phu["N"]) if "N" in phu else None,
                    "beta": beta, "pla": pla,
                    "gamma": float(phu["g"]) if "g" in phu else gamma_ten,
                })
    return ra, bo_qua, mau, len(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="logs_E1")
    ap.add_argument("--out", default="kept_pc")
    ap.add_argument("--moi_lop", action="store_true",
                    help="in them bang kept% theo tung lop")
    ap.add_argument("--ca_lim", action="store_true",
                    help="doc ca cac file -lim / _vb (mac dinh bo qua)")
    ap.add_argument("--xem", action="store_true",
                    help="in 5 dong log dau tien da khop, de kiem regex")
    a = ap.parse_args()

    ban, bo_qua, mau, n_file = quet(a)
    print(f"=== {n_file} file .log, bo qua {bo_qua} file -lim/_vb")
    print(f"=== {len(ban):,} dong SQDBG doc duoc\n")
    if a.xem:
        for s in mau:
            print("   ", s[:150])
        print()
    if not ban:
        raise SystemExit("!! Khong dong nao khop. Chay lai voi --xem va gui "
                         "cho toi mot dong log mau.")

    # ---- gop theo (ds, mode, nen, beta, gamma) -------------------------
    gop = {}
    for r in ban:
        k = (r["ds"], r["mode"], r["nen"], r["beta"], r["gamma"],
             r["pla"])
        gop.setdefault(k, {"kept": [], "target": set(), "seed": set(),
                           "N": set()})
        gop[k]["kept"].append(r["kept"])
        gop[k]["target"].add(r["target"])
        gop[k]["seed"].add(r["seed"])
        if r["N"]:
            gop[k]["N"].add(r["N"])

    bang = []
    for (ds, mode, nen, b, g, pla), v in sorted(gop.items()):
        mu, sd = tb(v["kept"])
        tg = sorted(v["target"])[0] if v["target"] else None
        bang.append({"ds": ds, "mode": mode, "nen": nen, "beta": b,
                     "gamma": g, "pla": pla,
                     "target": tg, "kept": mu, "std": sd,
                     "n_dong": len(v["kept"]), "n_seed": len(v["seed"]),
                     "N": sorted(v["N"])[0] if v["N"] else None})

    for ds in sorted({r["ds"] for r in bang}):
        for mode in sorted({r["mode"] for r in bang if r["ds"] == ds}):
            print(f"── {ds} · mode = {mode}")
            print(f"{'p':>5}{'target':>9}{'β':>7}{'γ':>8}{'pla':>5}"
                  f"{'kept%':>9}{'±':>7}{'nen THAT':>11}{'dong':>8}{'seed':>6}")
            for r in sorted([x for x in bang
                             if x["ds"] == ds and x["mode"] == mode],
                            key=lambda x: (x["nen"], x["beta"], x["gamma"],
                                           x["pla"])):
                that = 100.0 * (1.0 - r["kept"])
                print(f"{r['nen']:>5.2f}{100 * r['target']:>8.1f}%"
                      f"{r['beta']:>7g}{r['gamma']:>8g}"
                      f"{('có' if r['pla'] else '—'):>5}"
                      f"{100 * r['kept']:>8.1f}%{100 * r['std']:>7.1f}"
                      f"{that:>10.1f}%{r['n_dong']:>8}{r['n_seed']:>6}")
            print()

    # ---- SO SANH value-aware voi moc goc o CUNG target -----------------
    print("=" * 78)
    print("SO SANH: cau hinh value-aware co GIU NHIEU HON moc goc khong?")
    print("Neu co, thi no NEN IT HON, va moi so sanh chat luong deu thien vi.")
    print("=" * 78)
    print(f"{'bo du lieu':<18}{'mode':<11}{'p':>5}{'β':>6}{'γ':>7}"
          f"{'kept% moc':>11}{'kept% VA':>10}{'chenh':>9}")
    co_thien_vi = False
    for ds in sorted({r["ds"] for r in bang}):
        for mode in sorted({r["mode"] for r in bang if r["ds"] == ds}):
            for nen in sorted({r["nen"] for r in bang
                               if r["ds"] == ds and r["mode"] == mode}):
                nhom = [r for r in bang if r["ds"] == ds
                        and r["mode"] == mode and r["nen"] == nen]
                moc = next((r for r in nhom if r["beta"] == 0
                            and r["gamma"] == 0 and not r["pla"]), None)
                if moc is None:
                    continue
                for r in sorted(nhom, key=lambda x: (x["beta"], x["gamma"])):
                    if r is moc:
                        continue
                    d = 100.0 * (r["kept"] - moc["kept"])
                    if d > 1.0:
                        co_thien_vi = True
                    print(f"{ds:<18}{mode:<11}{nen:>5.2f}"
                          f"{r['beta']:>6g}{r['gamma']:>7g}"
                          f"{100 * moc['kept']:>10.1f}%"
                          f"{100 * r['kept']:>9.1f}%{d:>+8.1f}")
    if not any(r["beta"] == 0 and r["gamma"] == 0 for r in bang):
        print("   (khong thay dong nao beta=0 gamma=0 — log cua moc goc co")
        print("    the nam o thu muc khac; khong so sanh duoc)")
    print()

    # ---- PLACEBO: kept% hai nhanh co khop nhau khong -------------------
    cap = {}
    for r in bang:
        k = (r["ds"], r["mode"], r["nen"], r["beta"], r["gamma"])
        cap.setdefault(k, {})[r["pla"]] = r["kept"]
    cap = {k: v for k, v in cap.items() if True in v and False in v}
    if cap:
        print("=" * 78)
        print("PLACEBO — kept% cua hai nhanh co khop khong?")
        print("Neu lech qua 2 diem thi phep so sanh dang lan them bien 'nen")
        print("khac nhau', khong con co lap duoc thong tin value.")
        print("=" * 78)
        print(f"{'bo du lieu':<18}{'mode':<11}{'p':>5}{'β':>6}{'γ':>7}"
              f"{'kept khong pla':>16}{'kept placebo':>14}{'lech':>8}")
        xau = 0
        for k in sorted(cap):
            a_, b_ = cap[k][False], cap[k][True]
            d = 100.0 * (b_ - a_)
            if abs(d) > 2.0:
                xau += 1
            print(f"{k[0]:<18}{k[1]:<11}{k[2]:>5.2f}{k[3]:>6g}{k[4]:>7g}"
                  f"{100 * a_:>15.1f}%{100 * b_:>13.1f}%{d:>+8.1f}")
        print()
        if xau:
            print(f"!! {xau} cap lech qua 2 diem — phai bao cao kept% kem theo")
            print("   ket qua F1, hoac chuyen sang che do ngan sach (budget)")
            print("   noi kept% bi ghim dung bang target.\n")
        else:
            print("   Hai nhanh giu cung mot luong cache -> so sanh F1 sach.\n")

    # ---- canh bao target vs kept ---------------------------------------
    lech = [r for r in bang if r["target"] and
            abs(r["kept"] - r["target"]) > 0.10]
    if lech:
        print("=" * 78)
        print("!! CANH BAO — kept% THAT lech khoi target qua 10 diem.")
        print("   Nhan 'nen 70/80/90%' trong bai dang lay tu target, khong")
        print("   phai tu so do duoc. Phai bao cao theo cot 'nen THAT',")
        print("   hoac hieu chinh lai nguong T cho dat target.")
        print("=" * 78)
        for r in sorted(lech, key=lambda x: -abs(x["kept"] - x["target"]))[:10]:
            print(f"   {r['ds']:<18}{r['mode']:<11}p={r['nen']:.2f} "
                  f"β={r['beta']:g} γ={r['gamma']:g}  "
                  f"target {100 * r['target']:.1f}%  →  giu thuc "
                  f"{100 * r['kept']:.1f}%")
        print()
    if co_thien_vi:
        print("!! Co cau hinh value-aware giu nhieu hon moc goc tren 1 diem.")
        print("   Phai ghi kept% that vao bang ket qua, neu khong phan bien")
        print("   se noi cai tien chi thang vi nen it hon.\n")

    if a.moi_lop:
        print("=" * 78)
        print("kept% theo tung LOP (trung binh moi cau hinh)")
        print("=" * 78)
        theo_lop = {}
        for r in ban:
            theo_lop.setdefault((r["ds"], r["mode"], r["nen"], r["lop"]),
                                []).append(r["kept"])
        for k in sorted(theo_lop):
            mu, _ = tb(theo_lop[k])
            print(f"   {k[0]:<18}{k[1]:<11}p={k[2]:.2f}  L{k[3]:<3}"
                  f"{100 * mu:>8.1f}%")
        print()

    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font, Alignment, PatternFill
    except ImportError:
        print("!! Thieu openpyxl — bo qua xlsx.  pip install openpyxl")
        return
    wb = Workbook()
    ws = wb.active
    ws.title = "kept"
    head = ["Bộ dữ liệu", "mode", "p", "target", "β", "γ", "placebo",
            "kept", "std", "nén thật", "n_dòng", "n_seed", "N"]
    ws.append(head)
    for c in ws[1]:
        c.font = Font(bold=True, color="14243D", size=10)
        c.alignment = Alignment(horizontal="center", wrap_text=True)
        c.fill = PatternFill("solid", fgColor="F1EEE7")
    for r in bang:
        ws.append([r["ds"], r["mode"], r["nen"], r["target"], r["beta"],
                   r["gamma"], "có" if r["pla"] else "",
                   r["kept"], r["std"], 1.0 - r["kept"],
                   r["n_dong"], r["n_seed"], r["N"]])
    for row in ws.iter_rows(min_row=2):
        for c in row:
            if isinstance(c.value, float):
                c.number_format = "0.0000"
    p = a.out + ".xlsx"
    wb.save(p)
    print("Da ghi:", os.path.abspath(p), f"  ({len(bang)} dong)")


if __name__ == "__main__":
    main()