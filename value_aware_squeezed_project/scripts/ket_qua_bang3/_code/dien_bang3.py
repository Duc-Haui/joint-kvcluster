#!/usr/bin/env python3
"""dien_bang3.py — tinh so cho Table III va dien vao main.tex.

Doc:
  --toc_do   bang3_toc_do.json (do_toc_do.py, khoa allkv / sq / jkv)
  --cum_sq   clustering_time.jsonl cua Squeezed (beta=0)
  --cum_jkv  clustering_time.jsonl cua Joint KVCluster (beta=0.3, gamma=0)
Ghi:
  bang3.tex  cac dong cua bang (de xem / dan tay)
  --main     neu co: thay 6 dong so trong Table III (giu ban sao .bak_bang3)
"""
import argparse
import json
import re
import shutil


def tb_cum(path):
    """Thoi gian gom cum TB moi context, BO context 0 (khoi dong CUDA)."""
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    xs = [r["giay"] for r in rows if r["dataidx"] != 0] or [r["giay"] for r in rows]
    mu = sum(xs) / len(xs)
    sd = (sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5 if len(xs) > 1 else 0.0
    return mu, sd, len(xs)


def pct(a, b):
    p = (b - a) / a * 100.0
    return "$0.0\\%$" if abs(p) < 0.05 else f"${p:+.1f}\\%$"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--toc_do", required=True)
    ap.add_argument("--cum_sq", required=True)
    ap.add_argument("--cum_jkv", required=True)
    ap.add_argument("--main", default="")
    a = ap.parse_args()

    kq = {r["khoa"]: r for r in json.load(open(a.toc_do, encoding="utf-8"))}
    sq, jk = kq["sq"], kq["jkv"]
    c_sq, sd_sq, n = tb_cum(a.cum_sq)
    c_jk, sd_jk, _ = tb_cum(a.cum_jkv)

    # (nhan trong main.tex, gia tri sq, gia tri jkv, dinh dang)
    dong = [
        ("Offline clustering (s/context)", c_sq, c_jk, "{:.2f}"),
        ("Prefill (ms)", sq["ttft_ms"], jk["ttft_ms"], "{:.1f}"),
        ("Decoding (ms/token)", sq["tpot_ms"], jk["tpot_ms"], "{:.1f}"),
        ("End-to-end, $32$ tokens (ms)", sq["e2e_ms"], jk["e2e_ms"], "{:.0f}"),
        ("Peak VRAM (MiB)", sq["vram_mb"], jk["vram_mb"], "{:.0f}"),
        ("Peak host memory (MiB)", sq["ram_mb"], jk["ram_mb"], "{:.0f}"),
    ]
    tex = []
    for nhan, x, y, f in dong:
        tex.append(f"{nhan:<34}& ${f.format(x)}$ & ${f.format(y)}$ & {pct(x, y)} \\\\")
    open("bang3.tex", "w", encoding="utf-8").write("\n".join(tex) + "\n")
    print("\n".join(tex))

    print("\n--- so de sua doan van Section 'Runtime and Memory Overhead' ---")
    print(f"GPU {sq.get('gpu')}, ngu canh {sq['ngu_canh']} token, attn {sq['attn']}, "
          f"lap {sq['lap']}")
    print(f"Gom cum: Squeezed {c_sq:.2f}±{sd_sq:.2f} s, JKV {c_jk:.2f}±{sd_jk:.2f} s "
          f"(TB {n} context) -> x{c_jk / c_sq:.2f}")
    for k, ten in (("ttft", "prefill"), ("tpot", "decode"), ("e2e", "e2e")):
        print(f"{ten}: sq {sq[k + '_ms']:.2f}±{sq[k + '_std']:.2f} | "
              f"jkv {jk[k + '_ms']:.2f}±{jk[k + '_std']:.2f}")
    if "allkv" in kq:
        ak = kq["allkv"]["tpot_ms"]
        print(f"All-KV decode {ak:.1f} ms/token -> Squeezed cham x{sq['tpot_ms'] / ak:.2f}, "
              f"JKV cham x{jk['tpot_ms'] / ak:.2f}")

    if a.main:
        # newline="": giu nguyen CRLF cua file soan tren Windows
        s = open(a.main, encoding="utf-8", newline="").read()
        shutil.copy(a.main, a.main + ".bak_bang3")
        for (nhan, *_), moi in zip(dong, tex):
            mau = re.compile(r"^" + re.escape(nhan)
                             + r"[ \t]*&[^\r\n]*\\\\[ \t]*(?=\r?$)", re.M)
            s, so = mau.subn(lambda _m: moi, s, count=1)
            if so != 1:
                raise SystemExit(f"Khong tim thay dong '{nhan}' trong {a.main}")
        open(a.main, "w", encoding="utf-8", newline="").write(s)
        print(f"\nDa dien Table III vao {a.main} (ban cu: {a.main}.bak_bang3).")
        print("Con lai: bo '\\est{Blue: placeholder ...}' trong caption va sua doan van "
              "\\est{...} theo so o tren.")


if __name__ == "__main__":
    main()
