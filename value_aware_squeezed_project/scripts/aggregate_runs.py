#!/usr/bin/env python3
"""
Tổng hợp kết quả nhiều seed thành bảng mean ± std, kèm bootstrap CI.

HAI LOẠI SAI SỐ ĐƯỢC BÁO CÁO RIÊNG — chúng trả lời hai câu hỏi khác nhau:

  1. std qua các seed      "nếu khởi tạo k-means khác đi thì điểm lệch bao nhiêu?"
                           -> đo độ nhạy của PHƯƠNG PHÁP. Cần chạy lại thật.

  2. bootstrap CI 95%      "nếu lấy 200 câu hỏi khác của cùng phân phối thì
                           điểm lệch bao nhiêu?" -> đo sai số do CỠ MẪU.
                           Tính lại từ điểm từng mẫu, không phải chạy lại.

Một cải tiến chỉ đáng tin khi khoảng cách giữa hai phương pháp lớn hơn CẢ HAI.
Phép so sánh mạnh nhất ở cuối: bootstrap GHÉP CẶP (paired) trên hiệu số từng
mẫu — vì cùng một câu hỏi được cả hai phương pháp trả lời, ghép cặp khử được
phần lớn phương sai do câu dễ/khó, nhạy hơn hẳn so sánh hai mean rời rạc.

Điểm số: qa_f1 theo đúng LongBench (chuẩn hoá kiểu SQuAD, token-F1, lấy max
trên các đáp án đúng). KHÔNG cắt câu trả lời — qasper không nằm trong danh
sách dataset bị cắt của LongBench, và cắt sẽ thổi điểm lên giả tạo.
"""
import argparse
import glob
import json
import os
import re
import string
from collections import Counter

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
#  CAU HINH — sua o day roi chay `python3 aggregate_runs.py`, khong can go gi.
#  Doi so dong lenh (neu truyen) van de len duoc; run_repeat.sh truyen tuong
#  minh nen no khong bi anh huong boi khoi nay.
# ═══════════════════════════════════════════════════════════════════════════
CAU_HINH = {
    "dataset":    "qasper",
    "seeds":      "0 1 2",
    "mode":       "threshold",   # che do CHINH, cung la MOC so sanh
    "them_mode":  "",      # gom them vao cung bang; "" = khong gom
    "repeat_pct": "0.7 0.8 0.9",         # mot muc nen, hoac "all" cho tat ca
    "pct":        "0.7 0.8 0.9",            # loc BANG 1/2; "" = in tat ca
    "pc":         5,
    "model":      "LLaMA-2-7B-32K",
    "n_boot":     2000,
    "csv":        "",
    # None = lay tu $ROOT, roi $BASE, roi duong dan mac dinh
    "pred_root":  None,
}


# ---------------------------------------------------------------------------
# qa_f1 — chép đúng logic LongBench/metrics.py
# ---------------------------------------------------------------------------
def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def f1_score(pred_tokens, gt_tokens):
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def qa_f1(prediction, ground_truths):
    p = normalize_answer(prediction).split()
    best = 0.0
    for gt in ground_truths:
        g = normalize_answer(gt).split()
        if not p or not g:
            continue
        best = max(best, f1_score(p, g))
    return best


def per_sample_scores(jsonl_path):
    """Trả về np.array điểm từng mẫu (thang 0-100)."""
    out = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out.append(qa_f1(d["pred"], d["answers"]) * 100)
    return np.asarray(out, dtype=float)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
def mean_per_question(per_seed):
    """
    Gộp nhiều seed của CÙNG một cấu hình thành MỘT vector dài n_câu_hỏi.

    Phải lấy trung bình theo từng câu hỏi, KHÔNG nối các seed lại với nhau.
    Nối 3 seed x 200 câu thành 600 dòng rồi bootstrap là coi cùng một câu hỏi
    được hỏi 3 lần như 3 quan sát độc lập -> CI hẹp đi khoảng sqrt(3) lần so
    với thực tế. Cùng 200 câu hỏi đó, chỉ chạy lại với seed khác.
    """
    vecs = [v for _, v in sorted(per_seed.items())]
    n = min(len(v) for v in vecs)
    return np.mean(np.stack([v[:n] for v in vecs]), axis=0)


def bootstrap_ci(x, n_boot=2000, alpha=0.05, rng=None):
    """CI phần trăm cho mean của x."""
    if len(x) == 0:
        return (float("nan"),) * 3
    rng = rng or np.random.default_rng(0)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return x.mean(), lo, hi


def paired_bootstrap(a, b, n_boot=2000, rng=None):
    """
    Bootstrap ghép cặp trên hiệu b - a (b = phương pháp mới).
    Trả về (mean_diff, lo, hi, p_hai_phia).
    Yêu cầu a, b cùng độ dài và cùng thứ tự mẫu.
    """
    n = min(len(a), len(b))
    d = b[:n] - a[:n]
    rng = rng or np.random.default_rng(0)
    idx = rng.integers(0, n, size=(n_boot, n))
    dm = d[idx].mean(axis=1)
    lo, hi = np.percentile(dm, [2.5, 97.5])
    # p hai phía: tỉ lệ lần bootstrap đổi dấu so với hiệu quan sát được
    obs = d.mean()
    p = 2 * min((dm <= 0).mean(), (dm >= 0).mean())
    return obs, lo, hi, min(p, 1.0)


def kiem_dinh_dau(d, eps=1e-9):
    """Sign test tren so cau DOI DIEM. Khong gia dinh phan bo.
    Tra (so_tang, so_giam, p_hai_phia)."""
    from math import comb
    up = int((d > eps).sum()); dn = int((d < -eps).sum()); m = up + dn
    if m == 0:
        return up, dn, 1.0
    k = min(up, dn)
    return up, dn, min(2 * sum(comb(m, i) for i in range(k + 1)) / (2 ** m), 1.0)


def do_tap_trung(d, k=5):
    """Ty le dong gop cua k cau co |hieu| lon nhat. >50% = hieu ung don cuc."""
    tong = float(np.abs(d).sum())
    if tong <= 0:
        return 0.0
    return 100.0 * float(np.sort(np.abs(d))[-k:].sum()) / tong


# ---------------------------------------------------------------------------
def fmt(v, nd=2):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def main():
    _base = os.environ.get("BASE", os.path.expanduser(
        "~/Desktop/DucDang/SqueezedAttention-main"))
    _goc = CAU_HINH["pred_root"] or os.environ.get(
        "ROOT", _base + "/SqueezedAttention-main/LongBench")

    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_root", default=_goc, help="Thư mục LongBench/")
    ap.add_argument("--model", default=CAU_HINH["model"])
    ap.add_argument("--dataset", default=CAU_HINH["dataset"])
    ap.add_argument("--pc", type=int, default=CAU_HINH["pc"], help="percent_clusters")
    ap.add_argument("--seeds", default=CAU_HINH["seeds"])
    ap.add_argument("--repeat_pct", default=CAU_HINH["repeat_pct"])
    ap.add_argument("--pct", default=CAU_HINH["pct"],
                    help="Chi in cac muc nen nay o BANG 1 va BANG 2, "
                         "cach nhau bang dau cach. Vi du: --pct \"0.9\". "
                         "De trong = in tat ca.")
    ap.add_argument("--n_boot", type=int, default=CAU_HINH["n_boot"])
    ap.add_argument("--mode", default=CAU_HINH["mode"],
                    choices=["budget", "threshold", "adaptive"],
                    help="budget = pred_* ; threshold = pred_*_thr ; adaptive = pred_*_adp")
    ap.add_argument("--them_mode", default=CAU_HINH["them_mode"],
                    help="Gom THEM nhanh goc cua che do khac vao cung bang, "
                         "vd --them_mode budget. Cach nhau bang dau cach. "
                         "Moc so sanh van la --mode.")
    ap.add_argument("--csv", default=CAU_HINH["csv"], help="Ghi bảng ra file CSV")
    args = ap.parse_args()

    root = args.pred_root
    seeds = args.seeds.split()
    _TAG = {"threshold": "_thr", "adaptive": "_adp", "budget": "_bgt"}
    _TEN = {"threshold": "Squeezed", "budget": "ClusterKV",
            "adaptive": "Sq-adaptive"}
    tag = _TAG[args.mode]
    TEN_GOC = _TEN[args.mode]

    # Che do CHINH truoc (no la moc so sanh), roi cac che do them vao.
    # Nhan khac nhau theo che do nen khong the dung ten -> khong lan moc.
    _modes = [args.mode] + [m for m in args.them_mode.split() if m != args.mode]
    for _m in _modes:
        if _m not in _TAG:
            print(f"Che do khong hop le: {_m}  (threshold|budget|adaptive)")
            return
    rng = np.random.default_rng(12345)
    print()
    print(f"[cấu hình] dataset={args.dataset}  seeds='{args.seeds}'  "
          f"mode={args.mode}" + (f"  +{args.them_mode}" if args.them_mode else "")
          + f"  pc={args.pc}  repeat_pct={args.repeat_pct}")
    print(f"[cấu hình] pred_root={os.path.abspath(args.pred_root)}")

    def jl(pred_dir, conf):
        return os.path.join(root, pred_dir, conf, f"{args.dataset}.jsonl")

    # ── thu thập: {(method, pct): {seed: scores}} ───────────────────────────
    runs = {}
    n_skip_vb = 0          # so thu muc _vb (thanh phan 4) bi bo qua

    # baseline full-KV (không phụ thuộc seed)
    base_path = jl("pred_baseline", f"{args.model}_baseline")
    if not os.path.exists(base_path):  # tương thích bố cục cũ
        base_path = jl("pred", f"{args.model}_baseline")
    baseline = per_sample_scores(base_path) if os.path.exists(base_path) else None

    for _m in _modes:
      _t, _ten = _TAG[_m], _TEN[_m]
      # Hau to che do PHAI ngan, neu khong nhan se tran qua be rong cot va
      # day lech moi cot phia sau.  " [threshold]" = 12 ky tu -> " ·thr" = 5.
      _VIET_TAT = {"threshold": "thr", "budget": "bgt", "adaptive": "adp"}
      _hau = "" if _m == args.mode else " ·" + _VIET_TAT[_m]
      for seed in seeds:
        # nhanh goc cua che do nay
        for d in sorted(glob.glob(os.path.join(root, f"pred_s{seed}{_t}", f"{args.model}_PC{args.pc}_PERC*"))):
            pct = d.rsplit("PERC", 1)[-1]
            p = os.path.join(d, f"{args.dataset}.jsonl")
            if os.path.exists(p):
                runs.setdefault((_ten, pct), {})[seed] = per_sample_scores(p)
        # E1 — tên thư mục mang cả beta/gamma nên gom theo cấu hình.
        # Thư mục có hậu tố _vb... là của thành phần ④ (value-enriched
        # pruning), hiện ĐANG TẮT nên được BỎ QUA. Số lượng bỏ qua được
        # đếm và in ra một dòng, để không lặp lại lỗi bỏ sót im lặng.
        for d in sorted(glob.glob(os.path.join(
                root, f"pred_E1_b*_g*_s{seed}{_t}*",
                f"{args.model}_PC{args.pc}_PERC*"))):
            cfg = os.path.basename(os.path.dirname(d))
            if "_vb" in cfg:
                n_skip_vb += 1
                continue
            m = re.search(r"pred_E1_b([^_]+)_g([^_]+)_s\d+", cfg)
            label = (f"E1 b{m.group(1)} g{m.group(2)}" if m else "E1") + _hau
            pct = d.rsplit("PERC", 1)[-1]
            p = os.path.join(d, f"{args.dataset}.jsonl")
            if os.path.exists(p):
                runs.setdefault((label, pct), {})[seed] = per_sample_scores(p)

    if not runs:
        print("Không tìm thấy kết quả nào.")
        print("  pred_root : %s" % os.path.abspath(root))
        print("  dataset   : %s.jsonl" % args.dataset)
        print("  mode      : %s  -> hậu tố thư mục: '%s'" % (args.mode, tag or "(không có)"))
        print("  pc        : %d  -> tên thư mục phải chứa 'PC%d'" % (args.pc, args.pc))
        _found = sorted(os.path.basename(p)
                        for p in glob.glob(os.path.join(root, "pred_*")))
        print("  Thư mục pred_* có trong pred_root: %s"
              % (", ".join(_found) if _found else "KHÔNG CÓ"))
        _ds = sorted({os.path.basename(p)
                      for p in glob.glob(os.path.join(root, "pred_*", "*", "*.jsonl"))})
        print("  Tên dataset thật tìm thấy: %s"
              % (", ".join(_ds) if _ds else "KHÔNG CÓ"))
        return

    if n_skip_vb:
        print("[bỏ qua] %d thư mục có hậu tố _vb (thành phần ④, đang tắt)."
              % n_skip_vb)

    # --pct: chi giu mot hoac vai muc nen trong BANG 1 va BANG 2.
    # (--repeat_pct chi dieu khien BANG 3, khong loc hai bang tren.)
    if args.pct:
        _avail = sorted({k[1] for k in runs}, key=float)
        _keep = [float(x) for x in args.pct.split()]
        _before = len(runs)
        runs = {k: v for k, v in runs.items()
                if any(abs(float(k[1]) - t) < 1e-9 for t in _keep)}
        if not runs:
            print("Không có cấu hình nào ở mức nén %s." % args.pct)
            print("Các mức nén có sẵn: %s" % ", ".join(_avail))
            return
        print("[--pct %s] giữ %d / %d cấu hình. "
              "Các mức có sẵn: %s" % (args.pct, len(runs), _before,
                                      ", ".join(_avail)))

    # ── BẢNG 1: mean ± std qua các seed ────────────────────────────────────
    print()
    print("=" * 126)
    print(f"BẢNG 1 — ĐIỂM qa_f1 ({args.dataset}, chọn cụm: {args.mode}), mean ± std qua các seed")
    print("=" * 126)
    hdr = (f"{'Config':<28} {'Budget':>7} {'n_seed':>7} {'mean':>8} "
           f"{'Δ ref':>8} {'Δ cùng seed':>12} {'std(seed)':>10} {'SEM(câu)':>9}"
           f"  {'điểm từng seed'}")

    rows = []
    print(hdr)
    print("-" * 126)
    if baseline is not None:
        _sem_b = (baseline.std(ddof=1) / np.sqrt(len(baseline))
                  if len(baseline) > 1 else float("nan"))
        print(f"{'All KV (full)':<28} {1.0:>7.3f} {'—':>7} {baseline.mean():>8.2f} "
              f"{'—':>8} {'—':>12} {'—':>10} {fmt(_sem_b):>9}  trần trên, không nén")
        rows.append(["All KV", 1.0, "", baseline.mean(), "", "", "", _sem_b, ""])

    # Gom theo MUC NEN roi moi sap xep trong tung nhom, thay vi sap xep theo
    # ten cau hinh — neu khong thi 80% va 90% xen ke nhau, rat kho doc.
    _by_pct = {}
    for (method, pct) in runs:
        _by_pct.setdefault(pct, []).append(method)

    def _stats(method, pct):
        per_seed = runs[(method, pct)]
        vals = np.array([v.mean() for _, v in sorted(per_seed.items())])
        pooled = mean_per_question(per_seed)
        return (
            vals.mean(),
            vals.std(ddof=1) if len(vals) > 1 else float("nan"),
            (pooled.std(ddof=1) / np.sqrt(len(pooled))
             if len(pooled) > 1 else float("nan")),
            len(vals),
            "  ".join(f"s{s}={per_seed[s].mean():.2f}" for s in sorted(per_seed)),
        )

    for pct in sorted(_by_pct, key=float):
        methods = _by_pct[pct]
        budget = args.pc / 200.0 + (1.0 - float(pct))
        # Moc so sanh cua nhom: Squeezed goc neu co, neu khong thi nhanh
        # beta=0 gamma=0 (suy bien ve dung phuong phap goc), neu khong nua
        # thi khong co cot Delta.
        ref_name, ref_mean = None, None
        for _cand in (TEN_GOC, "E1 b0 g0"):
            if _cand in methods:
                ref_name, ref_mean = _cand, _stats(_cand, pct)[0]
                break

        print()
        print(f"── mức nén {int(float(pct)*100)}%   ·   budget {budget:.3f}"
              f"   ·   mốc so sánh: {ref_name or 'KHÔNG CÓ'} "
              f"{'' if ref_mean is None else '(%.2f)' % ref_mean}")

        # Trong moi nhom: diem cao nhat len dau, de quet mat nhanh.
        _lech = 0
        for method in sorted(methods, key=lambda m: -_stats(m, pct)[0]):
            mean, std, sem_q, n_seed, detail = _stats(method, pct)
            d = "—" if ref_mean is None else f"{mean - ref_mean:+.2f}"
            # Δ CÙNG SEED: chỉ lấy các seed CẢ HAI đều có. Nếu một cấu hình
            # chạy 1 seed còn mốc chạy 3 seed thì "Δ ref" là số ảo — nó so
            # một seed với trung bình của ba, trong đó có hai seed xấu.
            d2, canh = "—", ""
            if ref_name is not None:
                _c = sorted(set(runs[(method, pct)]) & set(runs[(ref_name, pct)]))
                if _c:
                    _a = np.mean([runs[(ref_name, pct)][s].mean() for s in _c])
                    _b = np.mean([runs[(method, pct)][s].mean() for s in _c])
                    d2 = f"{_b - _a:+.2f}"
                    if len(_c) != len(runs[(ref_name, pct)]) and method != ref_name:
                        canh = " !"
                        _lech += 1
            mark = "  <= mốc" if method == ref_name else ""
            print(f"{method:<28} {budget:>7.3f} {n_seed:>7} {mean:>8.2f} "
                  f"{d:>8} {(d2 + canh):>12} {fmt(std):>10} {fmt(sem_q):>9}"
                  f"  {detail}{mark}")
            rows.append([f"{method}-{int(float(pct)*100)}%", budget, n_seed,
                         mean, d, d2, std, sem_q, detail])
        if _lech:
            print(f"   ! {_lech} cấu hình có ít seed hơn mốc — ĐỌC CỘT "
                  f"'Δ cùng seed', KHÔNG đọc 'Δ ref'.")

    print()
    print("-" * 126)
    print("Δ ref       = mean dòng − mean mốc, DÙ hai bên khác số seed.")
    print("Δ cùng seed = chỉ tính trên các seed CẢ HAI đều có. Dấu '!' báo hai")
    print("              bên khác số seed — khi đó CHỈ cột này so sánh được.")
    print("std(seed) = độ lệch do KHỞI TẠO K-MEANS. Cần >= 2 seed, n_seed=1 thì là '—'.")
    print("SEM(câu)  = sai số chuẩn của điểm trung bình trên tập câu hỏi hiện có.")
    print("            Luôn tính được, kể cả với 1 seed. Khi báo cáo phải ghi đúng tên:")
    print("            'mean ± SEM qua câu hỏi', KHÔNG được ghi là 'std qua seed'.")
    print("Lưu ý: SEM của từng cấu hình RIÊNG LẺ không dùng để so sánh hai cấu hình.")
    print("       Muốn so sánh thì dùng BẢNG 3 (bootstrap ghép cặp trên hiệu số).")

    # ── BẢNG 2: bootstrap CI (sai số do cỡ mẫu) ────────────────────────────
    print()
    print("=" * 126)
    print(f"BẢNG 2 — BOOTSTRAP CI 95% (lấy TRUNG BÌNH các seed theo từng câu hỏi)")
    print("=" * 126)
    print(f"{'Config':<34} {'n_câu':>7} {'mean':>8} {'CI 95%':>20} {'độ rộng':>9}")
    print("-" * 126)
    if baseline is not None:
        m, lo, hi = bootstrap_ci(baseline, args.n_boot, rng=rng)
        print(f"{'All KV (full)':<34} {len(baseline):>7} {m:>8.2f} {f'[{lo:.2f}, {hi:.2f}]':>20} {hi-lo:>9.2f}")
    widths = []
    for (method, pct) in sorted(runs, key=lambda k: (k[0], float(k[1]))):
        pooled = mean_per_question(runs[(method, pct)])
        m, lo, hi = bootstrap_ci(pooled, args.n_boot, rng=rng)
        widths.append(hi - lo)
        name = f"{method}-{int(float(pct)*100)}%"
        print(f"{name:<34} {len(pooled):>7} {m:>8.2f} {f'[{lo:.2f}, {hi:.2f}]':>20} {hi-lo:>9.2f}")
    print("-" * 126)
    print("CI ở đây = sai số do CỠ MẪU. n_câu là SỐ CÂU HỎI, không nhân với số seed:")
    print("các seed được lấy trung bình theo từng câu, vì cùng 200 câu đó chỉ chạy lại.")
    if widths:
        res = 0.7 * float(np.mean(widths))
        print(f"Giới hạn phân giải: hai cấu hình RỜI RẠC chênh nhau dưới ~{res:.1f} điểm thì")
        print("bộ dữ liệu này không phân biệt nổi. Muốn nhạy hơn -> dùng bảng 3 (ghép cặp).")

    # ── BẢNG 3: bootstrap ghép cặp MỌI cấu hình vs mốc ────────────────────
    # Khac ban cu: truoc chi ghep E1 vs moc, va chi o MOT muc nen. Gio ghep
    # moi cau hinh (ke ca nhanh goc cua che do them vao qua --them_mode) va
    # co the chay o moi muc nen (--repeat_pct all).
    _pcts_b3 = (sorted({k[1] for k in runs}, key=float)
                if args.repeat_pct == "all" else [args.repeat_pct])
    _ketqua = []
    for pct in _pcts_b3:
        sq_key = (TEN_GOC, pct)
        cmp_keys = sorted(k for k in runs if k[0] != TEN_GOC and k[1] == pct)
        if sq_key not in runs or not cmp_keys:
            continue
        print()
        print("=" * 126)
        print(f"BẢNG 3 — GHÉP CẶP tại mức nén {pct}:  cấu hình − {TEN_GOC} "
              f"(cùng câu hỏi, cùng seed)")
        print("=" * 126)
        print(f"{'So sánh':<40} {'seed':>7} {'Δ mean':>9} "
              f"{'CI 95% của Δ':>22} {'p':>7} {'tăng/giảm':>11} {'p_dấu':>7} "
              f"{'top5':>6}")
        print("-" * 126)
        for k in cmp_keys:
            common = sorted(set(runs[sq_key]) & set(runs[k]))
            if not common:
                print(f"{k[0] + ' vs ' + TEN_GOC:<40}   không có seed chung")
                continue
            _nhan = k[0] + ' vs ' + TEN_GOC
            # TUNG SEED: giu lai tu ban cu — cho thay hieu ung co lap lai o
            # moi lan boc tham khong, hay chi den tu mot seed may man.
            if len(common) > 1:
                for s_ in common:
                    _a, _b = runs[sq_key][s_], runs[k][s_]
                    if len(_a) != len(_b):
                        print(f"{_nhan:<40} {('s'+s_):>7}   số mẫu lệch "
                              f"({len(_a)} vs {len(_b)}) — bỏ qua")
                        continue
                    _d, _lo, _hi, _p = paired_bootstrap(_a, _b, args.n_boot, rng=rng)
                    _u, _n2, _pd = kiem_dinh_dau(_b - _a)
                    print(f"{_nhan:<40} {('s'+s_):>7} {_d:>+9.2f} "
                          f"{f'[{_lo:+.2f}, {_hi:+.2f}]':>22} {_p:>7.3f} "
                          f"{f'{_u}/{_n2}':>11} {_pd:>7.3f} "
                          f"{do_tap_trung(_b - _a):>5.0f}%")
            A = mean_per_question({s_: runs[sq_key][s_] for s_ in common})
            B = mean_per_question({s_: runs[k][s_] for s_ in common})
            m_ = min(len(A), len(B))
            A, B = A[:m_], B[:m_]
            d, lo, hi, p = paired_bootstrap(A, B, args.n_boot, rng=rng)
            up, dn, p_dau = kiem_dinh_dau(B - A)
            tt = do_tap_trung(B - A)
            _nh = "gộp" if len(common) > 1 else ("s" + common[0])
            print(f"{_nhan:<40} {_nh:>7} {d:>+9.2f} "
                  f"{f'[{lo:+.2f}, {hi:+.2f}]':>22} {p:>7.3f} "
                  f"{f'{up}/{dn}':>11} {p_dau:>7.3f} {tt:>5.0f}%"
                  + ("   <= kết luận" if len(common) > 1 else ""))
            _ketqua.append((pct, k[0], d, lo, hi, p, p_dau, tt, up, dn))
        print("-" * 126)

    if _ketqua:
        m_cmp = len(_ketqua)
        nguong = 0.05 / m_cmp
        print()
        print(f"Số phép so sánh đã thực hiện: {m_cmp}")
        print(f"Ngưỡng Bonferroni = 0.05/{m_cmp} = {nguong:.5f}")
        _dat = [k for k in _ketqua if k[5] < nguong]
        if _dat:
            print("Vượt ngưỡng:")
            for (pct, ten, d, lo, hi, p, p_dau, tt, up, dn) in _dat:
                _nghi = []
                if p_dau > 0.05:
                    _nghi.append("sign test không ủng hộ")
                if tt > 50:
                    _nghi.append(f"top-5 câu chiếm {tt:.0f}%")
                print(f"   nén {pct}  {ten}  Δ={d:+.2f}  p={p:.4f}"
                      + ("   [NGHI NGỜ: " + "; ".join(_nghi) + "]" if _nghi else ""))
        else:
            print("KHÔNG cấu hình nào vượt ngưỡng Bonferroni.")
        print()
        print("CÁCH ĐỌC")
        print("  CI của Δ không chứa 0  -> vượt nhiễu cỡ mẫu. Nhưng còn phải vượt")
        print("  ngưỡng Bonferroni ở trên: chạy nhiều ô rồi lấy ô đẹp nhất là")
        print("  chọn-sau-khi-biết-kết-quả, không phải kiểm định.")
        print("  p_dấu (sign test) = số câu tăng/giảm có lệch khỏi 50/50 không.")
        print("  p nhỏ mà p_dấu lớn -> thắng ít câu nhưng thắng đậm, không hệ thống.")
        print("  top5 = phần trăm biến động do 5 câu lớn nhất. >50% là đơn cực.")
        print("  Nếu Δ nhỏ hơn std(seed) ở BẢNG 1 thì vẫn có thể chỉ là may mắn")
        print("  của một lần khởi tạo k-means — cần thêm seed.")

    # ── CSV ────────────────────────────────────────────────────────────────
    if args.csv:
        import csv as _csv
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            w.writerow(["config", "budget", "n_seed", "mean", "delta_ref",
                        "delta_cung_seed", "std_seed", "sem_cauhoi", "per_seed"])
            w.writerows(rows)
        print(f"\nĐã ghi CSV: {args.csv}")


if __name__ == "__main__":
    main()
