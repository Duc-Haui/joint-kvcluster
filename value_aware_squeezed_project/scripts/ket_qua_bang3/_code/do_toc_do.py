#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
do_toc_do.py — do LATENCY va MEMORY cho 3 phuong phap, trong MOT lan chay.

Sinh ra so lieu cho hai hinh kieu Fig. 4 va Fig. 5 cua bai bao:
    Latency (ms):          Prefill (1K prompt)  /  Generation (per token)
    Memory Allocation (MB): VRAM Usage          /  RAM Usage

BA PHUONG PHAP
    allkv   All-KV      khong nen (use_centroids tat)
    sq      Squeezed    cum b0-g0, use_value_aware=0, SQ_SELECT_MODE=threshold
    vasq    VA-Squeezed cum b{B}-g{G}, use_value_aware=1, threshold

  Squeezed dung CHINH thu muc cum beta=0 gamma=0 voi VA TAT. Theo
  code (clustering.py:141-150) beta=0 cho phan hoach TRUNG voi Squeezed goc,
  nen khong can thu muc cum rieng cho moc — dung lai cum ban da co.

VI SAO MOI PHUONG PHAP MOT TIEN TRINH RIENG
  torch.cuda.max_memory_allocated() do dinh cua CA tien trinh. Nap nam mo
  hinh trong mot tien trinh thi so VRAM cua phuong phap sau bi nhiem phan
  bo con sot cua phuong phap truoc, va con de OOM. Script tu goi lai chinh
  no bang subprocess cho tung phuong phap, moi tien trinh ghi mot file JSON
  nho, roi gop lai.

DIEU KIEN DE SO CO NGHIA
  - Ca nam phuong phap do trong CUNG mot lan chay, tren CUNG mot may.
  - Co torch.cuda.synchronize() truoc va sau moi phep do. Thieu no la do
    thoi gian GUI lenh, khong phai thoi gian CHAY.
  - KHONG can chay full du lieu va KHONG can nhieu seed: latency/memory
    khong phu thuoc noi dung dataset, cung khong phu thuoc seed gom cum.
    Cai can nhieu la SO LAN LAP DO (--lap), lay mean ± std qua cac lan lap.

CACH DUNG
    # kiem tra duong dan truoc, KHONG nap mo hinh:
    python3 do_toc_do.py --kiem_tra

    # 0) gom cum o 1K (mot lan): bash gom_cum_1k.sh
    # 1) do that (mac dinh: 1K, All-KV cung nhan eager, 20 lan lap):
    python3 do_toc_do.py --lap 20

    # 2) (nen lam) quet nhieu do dai de thay xu huong ro hon:
    NGU_CANH="1024 2048 4096" bash gom_cum_1k.sh
    python3 do_toc_do.py --ngu_canh "1024 2048 4096" --lap 20

  CHI SO (dat ten theo chuan benchmark LLM)
    TTFT  time to first token  = mot lan prefill tren ca prompt
    TPOT  time per output token = (E2E - TTFT) / (so token sinh - 1)
    E2E   end-to-end            = generate() tron ven `sinh` token
    TPS   tokens per second     = 1000 / TPOT
    Moi chi so co mean, std, p50, p95 qua --lap lan do (sau --warmup lan
    khoi dong, co torch.cuda.synchronize truoc/sau moi lan).
    Bo nho: VRAM trong so (sau khi nap) + KV cache DO THAT (tong byte
    past_key_values) + phan tam = VRAM dinh pha sinh; RAM dinh lay mau
    RSS moi 10 ms trong luc do.

  CHI CAN 1 TAP, 1 SEED: latency/memory phu thuoc do dai prompt, khong phu
  thuoc noi dung tap hay seed gom cum. Mean ± std lay qua --lap lan do.

  HAI LOAI SO VRAM TRONG FILE KET QUA
    vram_gen_mb        DO THAT o pha sinh token. Nen KV KHONG lam giam so
                       nay (cache khong bi xoa, xem modeling_llama.py:650).
    vram_hieu_dung_mb  SO TINH = vram_gen_mb - percentile * KV_ly_thuyet,
                       dung cong thuc cua benchmark_memory.py cu / Fig. 5.

XUAT RA
    <out>.json   so lieu tho (moi phuong phap mot muc)
    <out>.xlsx   bang de ve bieu do bang ve_do_do.py
"""
import argparse
import json
import os
import subprocess
import sys
import time

CAU_HINH = {
    "repo":       "",          # SqueezedAttention-main/ (chua LongBench/)
    "model_name": "LLaMA-2-7B-32K",
    "awq":        "",          # duong dan model AWQ
    "dataset":    "multifieldqa_en",
    # Do dai prompt khi do. Bai bao bao cao o 1K -> mac dinh 1024.
    # modeling_llama.py tinh
    #     K = int(percent_clusters/100 * (shared_prefix_length - obs_window))
    # va nap centroids_tensor_dict_{idx}_{K}.pt, nen CUM PHAI DUOC GOM O DUNG
    # DO DAI NAY: chay gom_cum_1k.sh truoc (ghi vao thu muc ...-ctx1024).
    # 0 = dung max_length cua pred.py (config/model2maxlen.json, 5000) va
    # dung lai cum cua cac lan chay pred.
    # Co the cho NHIEU do dai de quet, vd "1024 2048 4096" (moi do dai can
    # cum rieng: NGU_CANH="1024 2048 4096" bash gom_cum_1k.sh).
    "ngu_canh":   "1024",
    "sinh":       32,          # so token sinh ra de do toc do giai ma
    "warmup":     5,
    "lap":        20,
    "mau":        3,           # so prompt khac nhau
    "beta":       "0.05",
    "gamma":      "0.1",
    "seed":       "0",
    "pc":         5,
    "percentile": "0.7",
    "obs_window": 100,
    # Hau to cua thu muc cum. Gom cum voi SQ_LIMIT=N thi run_E1.sh dat
    # SUFFIX="-limN" vao TEN THU MUC -> phai khai o day, neu khong script
    # tim nham thu muc khong ton tai.
    # "auto" = "-ctx<ngu_canh>" khi ngu_canh > 0 (thu muc cua gom_cum_1k.sh),
    #          ""               khi ngu_canh = 0 (cum cua run_E1.sh).
    "hau_to":     "auto",
    # value_aware_squeezed_project/ — de trong thi suy tu vi tri file nay
    # (script nam trong .../value_aware_squeezed_project/scripts/).
    "va_project": "",
    "out":        "do_toc_do",
}

# (khoa, ten hien thi, dung cum?, che do chon, cum b{B}-g{G}?, bat value-aware?)
# jkv = Joint KVCluster cua bai bao (beta>0, gamma=0): CHI doi phan hoach
#   (thu muc cum b{B}-g0), con buoc giai ma giong het Squeezed -> VA tat.
#   Voi gamma=0 boost (1+gamma*sigma)=1 nen VA bat hay tat cho cung ket qua;
#   tat de khong nap them value_centroids/normalized_variance khong dung toi.
PP = [
    ("allkv", "All-KV",          False, None,        False, False),
    ("sq",    "Squeezed Attn",   True,  "threshold", False, False),
    ("vasq",  "VA-Squeezed",     True,  "threshold", True,  True),
    ("jkv",   "Joint KVCluster", True,  "threshold", True,  False),
]


def cac_duong_dan_python(a):
    """Ba duong dan ma run_E1.sh:37 dat vao PYTHONPATH truoc khi chay.

        export PYTHONPATH="${VA_PROJECT}:${REPO}/transformers/src:${REPO}:..."

    Thieu bat ky cai nao la import vo nghia:
      VA_PROJECT            -> goi 'value_aware'
      REPO/transformers/src -> ban transformers DA VA (modeling_llama)
      REPO                  -> goi 'squeezedattention'
    """
    return [a.va_project,
            os.path.join(a.repo, "transformers", "src"),
            a.repo]


def duong_dan_cum(a, dung_va):
    b, g = (a.beta, a.gamma) if dung_va else ("0", "0")
    return os.path.join(a.repo,
                        f"Clusters-E1-b{b}-g{g}-s{a.seed}{a.hau_to}")


def kiem_tra(a):
    """Kiem moi thu CAN TRUOC KHI nap mo hinh. Loi o day re hon loi sau 5
    phut nap model."""
    loi = []
    lb = os.path.join(a.repo, "LongBench")
    if not os.path.isdir(lb):
        loi.append(f"Khong thay LongBench/: {lb}   (dung --repo)")
    if not os.path.exists(os.path.join(lb, "pred.py")):
        loi.append(f"Khong thay pred.py trong {lb}")
    if a.awq and not os.path.isdir(a.awq):
        loi.append(f"Khong thay model AWQ: {a.awq}")
    for d, vi_sao in zip(cac_duong_dan_python(a),
                         ("goi 'value_aware' (dung --va_project)",
                          "ban transformers DA VA",
                          "goi 'squeezedattention'")):
        if not os.path.isdir(d):
            loi.append(f"Khong thay {d}   <- can cho {vi_sao}")
    for dung_va in (False, True):
        d = duong_dan_cum(a, dung_va)
        con = os.path.join(d, a.dataset)
        if not os.path.isdir(con):
            loi.append(f"Khong thay thu muc cum: {con}")
        else:
            n_cum = len([f for f in os.listdir(con)
                         if f.startswith("centroids_tensor_dict_")])
            if n_cum == 0:
                loi.append(f"Thu muc cum rong (khong co centroids_*.pt): "
                           f"{con}")
            elif n_cum < a.mau:
                loi.append(f"Chi co {n_cum} prefix da gom cum trong {con}, "
                           f"nhung --mau = {a.mau}. Giam --mau xuong "
                           f"{n_cum}, hoac gom cum them.")
    print("=" * 66)
    if loi:
        print("THIEU:")
        for x in loi:
            print("  !! " + x)
        print("=" * 66)
        return False
    print("OK — du duong dan de chay. Cac thu muc cum:")
    for dung_va in (False, True):
        print("   " + duong_dan_cum(a, dung_va))
    print("=" * 66)
    return True


# --------------------------------------------------------------- worker
def chay_mot(a, khoa):
    """Do MOT phuong phap. Chay trong tien trinh rieng."""
    import gc
    import torch

    ten, dung_cum, che_do, cum_va, bat_va = next(
        (t, c, m, cv, v) for k, t, c, m, cv, v in PP if k == khoa)
    a.ngu_canh = int(a.ngu_canh)

    lb = os.path.join(a.repo, "LongBench")
    # Thu tu giong PYTHONPATH cua run_E1.sh: VA_PROJECT, transformers da va,
    # REPO — roi moi den LongBench. Phai dat TRUOC 'import pred', vi pred.py
    # keo theo transformers.models.llama.
    for d in reversed(cac_duong_dan_python(a) + [lb]):
        if d not in sys.path:
            sys.path.insert(0, d)
    cwd = os.getcwd()
    os.chdir(lb)                      # pred.py doc config/ bang duong dan tuong doi
    try:
        import pred as P
        from datasets import load_dataset

        if che_do:
            os.environ["SQ_SELECT_MODE"] = che_do
        os.environ.setdefault("SQ_DEBUG", "0")

        model2path = json.load(open("config/model2path.json"))
        d2p = json.load(open("config/dataset2prompt.json"))
        max_len = (a.ngu_canh or
                   json.load(open("config/model2maxlen.json"))[a.model_name])
        print(f"   max_length = {max_len}"
              + ("  (tu config/model2maxlen.json, giong pred.py)"
                 if not a.ngu_canh else "  (ep bang --ngu_canh)"))

        cfg = {
            "path_to_clusters_cosine": duong_dan_cum(a, cum_va) + "/"
                                       + a.dataset + "/",
            "use_centroids": dung_cum,
            "hierarchical_lookup": False,
            "percent_clusters": a.pc,
            "percent_clusters_l2": -1,
            "percentile": float(a.percentile),
            "percentile_lower": 0.7,
            "obs_window": a.obs_window,
            "value_aware_gamma": float(a.gamma) if bat_va else 0.0,
            "use_value_aware": bool(bat_va),
        }
        # ---- CUNG MOT NHAN ATTENTION cho moi phuong phap -----------------
        # pred.py:159:
        #     config._attn_implementation = "eager" if use_centroids else "sdpa"
        # Nghia la All-KV chay SDPA (nhan hop nhat, rat nhanh) con cac phuong
        # phap nen chay eager (attention viet tay, cham hon nhieu lan). So hai
        # cai do voi nhau la so HAI NHAN, khong phai so hai phuong phap.
        # --cung_nhan ep All-KV cung chay eager de phep so sanh cong bang.
        attn = "eager" if (dung_cum or a.cung_nhan) else "sdpa"
        if a.cung_nhan and not dung_cum:
            _goc_auto = P.AutoModelForCausalLM.from_pretrained
            _goc_llama = P.LlamaForCausalLM.from_pretrained

            def _ep_eager(goc):
                def f(*ar, **kw):
                    c = kw.get("config")
                    if c is not None:
                        c._attn_implementation = "eager"
                        c._flash_attn_2_enabled = False
                    return goc(*ar, **kw)
                return f

            class _ShimA:
                from_pretrained = staticmethod(_ep_eager(_goc_auto))

            class _ShimL:
                from_pretrained = staticmethod(_ep_eager(_goc_llama))

            P.AutoModelForCausalLM = _ShimA
            P.LlamaForCausalLM = _ShimL
            print("   [cung_nhan] ep All-KV chay eager de so cong bang")

        dev = torch.device("cuda:0")
        t0 = time.time()
        model, tok = P.load_model_and_tokenizer(
            model2path[a.model_name], a.model_name, dev, cfg,
            load_awq=bool(a.awq), awq_model_path=(a.awq or None))
        nap_giay = time.time() - t0
        torch.cuda.synchronize()
        # Bo nho TINH: trong so mo hinh (AWQ 4-bit) nam tren GPU sau khi nap.
        vram_trong_so = torch.cuda.memory_allocated() / (1024 ** 2)

        # ---- chuan bi vai prompt dai dung ~ngu_canh token ----------------
        data = load_dataset("THUDM/LongBench", a.dataset, split="test")
        mau_in = []
        for i, o in enumerate(data):
            if len(mau_in) >= a.mau:
                break
            o = dict(o)
            o["different_prefix_index"] = i
            tho = d2p[a.dataset].format(**o)
            khong_hoi = d2p[a.dataset + "_prompt"].format(**o)
            pr, spl = P.truncate_fn(tho, khong_hoi, tok, max_len,
                                     a.dataset, dev)
            # Kiem TRUOC khi chay: K suy ra tu spl phai co file cum tuong ung.
            # Khong co thi bao ngay kem K dang co, thay vi de torch.load nem
            # FileNotFoundError kho hieu o giua phep do.
            K = max(int(a.pc / 100.0 * (spl - a.obs_window)), 1)
            if dung_cum:
                f_cum = os.path.join(cfg["path_to_clusters_cosine"],
                                     f"centroids_tensor_dict_{i}_{K}.pt")
                if not os.path.exists(f_cum):
                    co = sorted({int(x.rsplit("_", 1)[-1][:-3])
                                 for x in os.listdir(
                                     cfg["path_to_clusters_cosine"])
                                 if x.startswith("centroids_tensor_dict_")})
                    raise SystemExit(
                        f"\n!! Khong co file cum cho prefix {i} voi K={K}.\n"
                        f"   Thu muc: {cfg['path_to_clusters_cosine']}\n"
                        f"   K dang co: {co}\n"
                        f"   K = pc/100 * (shared_prefix_length - obs_window),"
                        f" nen K khac nghia la PROMPT DAI KHAC luc gom cum.\n"
                        f"   Cach sua: BO --ngu_canh (dung max_length cua "
                        f"pred.py), HOAC gom cum lai o dung do dai muon do.")
            mau_in.append((tok(pr, truncation=False, return_tensors="pt")
                           .to(dev), spl, i))

        def dat(spl, idx):
            model.model.shared_prefix_length = spl
            model.model.different_prefix_index = idx

        def dong_bo():
            torch.cuda.synchronize()

        # ---- do PREFILL --------------------------------------------------
        def mot_prefill(inp, spl, idx):
            dat(spl, idx)
            with torch.no_grad():
                model(**inp, use_cache=True)

        for _ in range(a.warmup):
            inp, spl, idx = mau_in[0]
            mot_prefill(inp, spl, idx)
        dong_bo()

        # CHI do tren MOT prompt. Luan phien nhieu prompt thi std phan anh
        # chenh lech DO DAI giua cac prompt (latency ti le voi do dai), chu
        # khong phai nhieu cua phep do — ra thanh sai so to vo nghia.
        #
        # Chon prompt DAI NHAT, khong phai prompt dau tien. Ly do: muc dich
        # cua nen KV cache la cat chi phi attention, ma chi phi do ti le voi
        # do dai ngu canh. Do o prompt ngan la do dung cho ma phuong phap
        # KHONG duoc thiet ke cho — chi phi tra centroid co dinh se lan at,
        # va moi phuong phap deu ra gan bang nhau.
        mau_in.sort(key=lambda m: -m[1])
        inp0, spl0, idx0 = mau_in[0]
        print("   do dai (token) cua cac prompt ung vien: "
              + ", ".join(str(m[1]) for m in mau_in)
              + f"   -> do tren {spl0}")
        if spl0 < max_len * 0.8:
            print(f"   !! CANH BAO: prompt dai nhat ({spl0}) con xa tran "
                  f"max_length={max_len}.")
            print(f"      Tang --mau de quet them mau (can co cum cho tung "
                  f"prefix), hoac doi --dataset sang bo co tai lieu dai hon "
                  f"(narrativeqa dai 15K-25K).")

        # ---- KV cache DO THAT: tong byte cac tensor trong past_key_values
        # sau mot lan prefill. Day la bo nho cache that su ton tai, khong
        # phai cong thuc. (Nen KV o repo nay chi che attention, nen so nay
        # se BANG NHAU giua cac phuong phap — do la ket qua, khong phai loi.)
        def _byte_cache(c):
            if c is None:
                return 0
            if torch.is_tensor(c):
                return c.numel() * c.element_size()
            tong = 0
            for ten_ds in ("key_cache", "value_cache"):
                if hasattr(c, ten_ds):
                    tong += sum(_byte_cache(x) for x in getattr(c, ten_ds))
            if tong:
                return tong
            if isinstance(c, (list, tuple)):
                return sum(_byte_cache(x) for x in c)
            return 0

        dat(spl0, idx0)
        with torch.no_grad():
            _o = model(**inp0, use_cache=True)
        kv_do = _byte_cache(getattr(_o, "past_key_values", None)) / 1024 ** 2
        del _o
        torch.cuda.empty_cache()

        # ---- RAM DINH: lay mau RSS moi 10 ms trong suot luc do ------------
        # (doc RSS mot lan sau khi chay co the bo lo dinh that.)
        import threading
        try:
            import psutil
            _proc = psutil.Process()
        except ImportError:
            _proc = None
        _ram = {"dinh": 0.0, "chay": True}

        def _lay_mau():
            while _ram["chay"] and _proc is not None:
                _ram["dinh"] = max(_ram["dinh"],
                                   _proc.memory_info().rss / 1024 ** 2)
                time.sleep(0.01)
        _luong = threading.Thread(target=_lay_mau, daemon=True)
        _luong.start()

        torch.cuda.reset_peak_memory_stats()
        pre = []
        for r in range(a.lap):
            inp, spl, idx = inp0, spl0, idx0
            dong_bo()
            t = time.perf_counter()
            mot_prefill(inp, spl, idx)
            dong_bo()
            pre.append((time.perf_counter() - t) * 1000.0)

        # ---- do GENERATION (tru phan prefill) ----------------------------
        def mot_gen(inp, spl, idx):
            dat(spl, idx)
            with torch.no_grad():
                model.generate(**inp, max_new_tokens=a.sinh, num_beams=1,
                               do_sample=False, use_cache=True)

        for _ in range(max(1, a.warmup // 2)):
            inp, spl, idx = mau_in[0]
            mot_gen(inp, spl, idx)
        dong_bo()

        vram_prefill = torch.cuda.max_memory_allocated() / (1024 ** 2)

        # Dinh bo nho RIENG pha sinh token: reset NGAY TRUOC vong lap. Day
        # moi la cho viec nen KV cache co tac dung. Dinh chung (ca prefill)
        # bi pha prefill ap dao — moi phuong phap deu tinh du KV cho toan bo
        # ngu canh roi moi cat, nen dinh chung gan nhu bang nhau.
        torch.cuda.reset_peak_memory_stats()
        gen = []
        for r in range(a.lap):
            inp, spl, idx = inp0, spl0, idx0
            dong_bo()
            t = time.perf_counter()
            mot_gen(inp, spl, idx)
            dong_bo()
            gen.append((time.perf_counter() - t) * 1000.0)

        vram_gen = torch.cuda.max_memory_allocated() / (1024 ** 2)
        vram = max(vram_prefill, vram_gen)
        vram_model = torch.cuda.memory_allocated() / (1024 ** 2)

        # ---- KV cache LY THUYET va VRAM "hieu dung" (SO TINH) --------------
        # modeling_llama.py:650 ghi TOAN BO key/value vao cache; nen KV chi
        # che attention, khong giai phong bo nho. VRAM do duoc vi the KHONG
        # giam. benchmark_memory.py cu (va Fig. 5 cua bai) bao cao
        #     hieu_dung = do_duoc - percentile * KV_ly_thuyet
        # Giu lai cot nay de ve lai duoc hinh kieu bai, nhung DAY LA SO TINH:
        # caption phai ghi "computed", khong phai "measured".
        mc = model.config
        n_kv = getattr(mc, "num_key_value_heads", mc.num_attention_heads)
        hd = mc.hidden_size // mc.num_attention_heads
        n_tok = int(inp0.input_ids.shape[-1]) + a.sinh
        kv_mb = 2 * mc.num_hidden_layers * n_kv * hd * n_tok * 2 / 1024 ** 2
        cat = float(a.percentile) if dung_cum else 0.0
        vram_hieu_dung = vram_gen - cat * kv_mb
        _ram["chay"] = False
        _luong.join(timeout=1)
        ram = _ram["dinh"] or (
            _proc.memory_info().rss / 1024 ** 2 if _proc else None)

        # ---- LUOT TACH THANH PHAN (rieng, KHONG anh huong so do o tren) ---
        # Prefill : thoi gian torch.load file cum (centroid, nhan, nguong,
        #           phuong sai) — modeling_llama.py doc lai o MOI prompt.
        # Decode  : thoi gian LlamaAttention._pytorch_centroid_lookup (cham
        #           diem cum + chon cum + nhan phuong sai cua VA).
        # CA HAI do bang dong ho tuong + torch.cuda.synchronize() TRUOC va SAU
        # moi lan goi. Khong dong bo thi:
        #   - torch.load (chep tensor len GPU) phai CHO prefill truoc chay xong
        #     -> thoi gian doc file bi thoi phong (loi cua ban truoc);
        #   - CUDA event do khoang tren dong thoi gian GPU, gom ca luc GPU ngoi
        #     cho CPU phat lenh -> sai lech khac nhau giua 1K va 4K.
        # Dong bo lam luot nay cham hon, nhung no la luot RIENG: chi lay so ms
        # cua phan tra cum; latency tong van lay tu luot do chinh o tren.
        # Phan con lai = attention (che mask tren TOAN BO KV) + FFN + cac lop
        # khac. Ti le lay tu luot nay, nhan vao latency do o tren.
        tai_cum_ms = 0.0
        lookup_ms_buoc = 0.0
        pre_mu_tam = sum(pre) / len(pre)
        if dung_cum and a.lap_tach > 0:
            import torch as _t
            _goc_load = _t.load
            _dem = {"bat": False, "giay": 0.0}

            def _load_do(*ar, **kw):
                if not _dem["bat"]:
                    return _goc_load(*ar, **kw)
                _t.cuda.synchronize()
                t0_ = time.perf_counter()
                try:
                    return _goc_load(*ar, **kw)
                finally:
                    _t.cuda.synchronize()
                    _dem["giay"] += time.perf_counter() - t0_
            _lop_attn = type(model.model.layers[0].self_attn)
            _goc_lk = getattr(_lop_attn, "_pytorch_centroid_lookup", None)
            _lk = {"giay": 0.0}

            def _lk_do(self_, *ar, **kw):
                _t.cuda.synchronize()
                t0_ = time.perf_counter()
                ra_ = _goc_lk(self_, *ar, **kw)
                _t.cuda.synchronize()
                _lk["giay"] += time.perf_counter() - t0_
                return ra_
            try:
                _t.load = _load_do
                if _goc_lk is not None:
                    _lop_attn._pytorch_centroid_lookup = _lk_do
                # prefill: chi dem thoi gian doc file cum
                _dem.update(bat=True, giay=0.0)
                for _ in range(a.lap_tach):
                    dong_bo()
                    mot_prefill(inp0, spl0, idx0)
                    dong_bo()
                _dem["bat"] = False
                tai_cum_ms = _dem["giay"] * 1000.0 / a.lap_tach
                # decode: tong thoi gian lookup / so buoc giai ma
                _lk["giay"] = 0.0
                for _ in range(a.lap_tach):
                    dong_bo()
                    mot_gen(inp0, spl0, idx0)
                    dong_bo()
                lookup_ms_buoc = (_lk["giay"] * 1000.0 /
                                  (a.lap_tach * max(a.sinh - 1, 1)))
            finally:
                _t.load = _goc_load
                if _goc_lk is not None:
                    _lop_attn._pytorch_centroid_lookup = _goc_lk
            print(f"   [tach] prefill: doc file cum {tai_cum_ms:.1f} ms "
                  f"(phan con lai {pre_mu_tam - tai_cum_ms:.1f} ms) · "
                  f"decode: centroid lookup {lookup_ms_buoc:.2f} ms/buoc")

        def ms(xs):
            mu = sum(xs) / len(xs)
            sd = ((sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5
                  if len(xs) > 1 else None)
            return mu, sd

        def pct(xs, q):
            xs = sorted(xs)
            i = (len(xs) - 1) * q / 100.0
            lo = int(i)
            hi = min(lo + 1, len(xs) - 1)
            return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)

        pre_mu, pre_sd = ms(pre)
        gen_mu, gen_sd = ms(gen)
        # TPOT (time per output token): generate() sinh token DAU TIEN ngay
        # trong pha prefill, nen phan con lai (gen - prefill) chi gom
        # (sinh - 1) buoc giai ma. Chia cho sinh se ra so thap hon thuc te.
        n_giai_ma = max(a.sinh - 1, 1)
        tpot = [max(g - pre_mu, 0.0) / n_giai_ma for g in gen]
        tok_mu, tok_sd = ms(tpot)

        ra = {"khoa": khoa, "ten": ten,
              "prefill_ms": pre_mu, "prefill_std": pre_sd,
              "token_ms": tok_mu, "token_std": tok_sd,
              "gen_tong_ms": gen_mu,
              "vram_mb": vram, "vram_gen_mb": vram_gen,
              "vram_prefill_mb": vram_prefill, "vram_model_mb": vram_model,
              "kv_mb": kv_mb, "vram_hieu_dung_mb": vram_hieu_dung,
              # --- ten chi so chuan (TTFT / TPOT / E2E / TPS) ---------------
              "ttft_ms": pre_mu, "ttft_std": pre_sd,
              "ttft_p50": pct(pre, 50), "ttft_p95": pct(pre, 95),
              "tpot_ms": tok_mu, "tpot_std": tok_sd,
              "tpot_p50": pct(tpot, 50), "tpot_p95": pct(tpot, 95),
              "e2e_ms": gen_mu, "e2e_std": gen_sd,
              "e2e_p50": pct(gen, 50), "e2e_p95": pct(gen, 95),
              "tps": (1000.0 / tok_mu) if tok_mu else None,
              # --- bo nho: tinh (trong so) + dong (KV do that) + dinh ---------
              "vram_trong_so_mb": vram_trong_so,
              "kv_do_mb": kv_do,
              "vram_tam_mb": max(vram_gen - vram_trong_so - kv_do, 0.0),
              # --- tach thanh phan latency (luot profile rieng) -------------
              "prefill_tai_cum_ms": tai_cum_ms,
              "decode_lookup_ms": lookup_ms_buoc,
              "ram_mb": ram,
              "attn": attn,
              "gpu": (torch.cuda.get_device_name(0)
                      if hasattr(torch.cuda, "get_device_name") else None),
              "awq": bool(a.awq),
              "ngu_canh": int(inp0.input_ids.shape[-1]),
              "shared_prefix_length": int(spl0),
              "sinh": a.sinh, "lap": a.lap, "nap_giay": nap_giay}

        del model
        gc.collect()
        torch.cuda.empty_cache()
        return ra
    finally:
        os.chdir(cwd)


# --------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    for k, v in CAU_HINH.items():
        ap.add_argument(f"--{k}", default=v,
                        type=(int if isinstance(v, int) else str))
    ap.add_argument("--lim", default="",
                    help="tuong duong --hau_to=-lim<N>. Dung cai nay thay vi "
                         "--hau_to: gia tri '-lim5' bat dau bang dau gach nen "
                         "argparse tuong la mot tuy chon khac.")
    ap.add_argument("--cung_nhan", action="store_true", default=True,
                    help="(MAC DINH BAT) ep All-KV chay eager giong cac "
                         "phuong phap nen, de so sanh cung mot nhan.")
    ap.add_argument("--allkv_sdpa", action="store_true",
                    help="tat --cung_nhan: All-KV chay SDPA nhu pred.py:159 "
                         "(nhanh hon nhieu lan vi khac NHAN, khong phai vi "
                         "khac phuong phap).")
    ap.add_argument("--kiem_tra", action="store_true",
                    help="chi kiem duong dan, KHONG nap mo hinh")
    ap.add_argument("--lap_tach", type=int, default=5,
                    help="so lan chay luot tach thanh phan (0 = bo qua)")
    ap.add_argument("--chi", default="",
                    help="chi do mot so phuong phap, vd 'allkv sq'")
    ap.add_argument("--_worker", default="", help=argparse.SUPPRESS)
    ap.add_argument("--_ra", default="", help=argparse.SUPPRESS)
    a = ap.parse_args()

    if not a.repo:
        a.repo = os.environ.get("REPO", os.path.expanduser(
            "~/Desktop/DucDang/SqueezedAttention-main/SqueezedAttention-main"))
    if a.allkv_sdpa:
        a.cung_nhan = False
    if not a.awq:
        # Cung model voi cac lan chay pred (run_E1.sh dung AWQ).
        a.awq = os.environ.get("AWQ_MODEL_PATH", "")
        mac_dinh = "/home/mtahackathon/models/LLaMA-2-7B-32K-AWQ"
        if not a.awq and os.path.isdir(mac_dinh):
            a.awq = mac_dinh
    if not a.va_project:
        a.va_project = os.environ.get("VA_PROJECT") or os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))
    def giai_hau_to(hau_to, nc):
        if a.lim:
            return f"-lim{a.lim.lstrip('-lim')}"
        if hau_to == "auto":
            return f"-ctx{nc}" if int(nc) > 0 else ""
        # Cho phep viet --hau_to lim5 (khong dau gach) cho de go.
        if hau_to and not hau_to.startswith("-"):
            return "-" + hau_to
        return hau_to

    if a._worker:
        a.hau_to = giai_hau_to(a.hau_to, int(a.ngu_canh))
        kq = chay_mot(a, a._worker)
        with open(a._ra, "w", encoding="utf-8") as f:
            json.dump(kq, f)
        return

    # ---- quet MOT hoac NHIEU do dai ngu canh ----------------------------
    hau_to_goc = a.hau_to
    cac_nc = [int(x) for x in str(a.ngu_canh).replace(",", " ").split()]
    nc_ok = []
    for nc in cac_nc:
        a.ngu_canh = nc
        a.hau_to = giai_hau_to(hau_to_goc, nc)
        print(f"\n### ngu canh {nc}")
        if kiem_tra(a):
            nc_ok.append(nc)
        else:
            print(f"   -> BO QUA do dai {nc} (thieu cum). Gom cum: "
                  f"NGU_CANH={nc} bash gom_cum_1k.sh")
    if not nc_ok:
        sys.exit(1)
    if a.kiem_tra:
        return

    lam = [k for k, *_ in PP if not a.chi or k in a.chi.split()]
    tmp = os.path.abspath(a.out + "_tam")
    os.makedirs(tmp, exist_ok=True)
    ket = []
    for nc in nc_ok:
      a.ngu_canh = nc
      a.hau_to = giai_hau_to(hau_to_goc, nc)
      for k in lam:
        ten = next(t for kk, t, *_ in PP if kk == k)
        print(f"\n>>> {ten}  ·  ngu canh {nc}  (tien trinh rieng)")
        f_ra = os.path.join(tmp, f"{k}_{nc}.json")
        cmd = [sys.executable, os.path.abspath(__file__), "--_worker", k,
               "--_ra", f_ra]
        if not a.cung_nhan:
            cmd.append("--allkv_sdpa")
        cmd.append(f"--lap_tach={a.lap_tach}")
        for key in CAU_HINH:
            v = str(getattr(a, key))
            # "=" thay vi tach doi so: gia tri "-lim5" bat dau bang dau gach,
            # tach doi so thi tien trinh con lai vap dung loi vua roi.
            cmd += [f"--{key}={v}"]
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            cac_duong_dan_python(a) + [env.get("PYTHONPATH", "")]).rstrip(
                os.pathsep)
        r = subprocess.run(cmd, env=env)
        if r.returncode != 0 or not os.path.exists(f_ra):
            print(f"!! {ten} @ {nc} THAT BAI (ma {r.returncode}). Bo qua.")
            continue
        kq = json.load(open(f_ra, encoding="utf-8"))
        kq["ngu_canh_muc_tieu"] = nc
        ket.append(kq)
        print(f"    TTFT {kq['ttft_ms']:.1f} ms (p95 {kq['ttft_p95']:.1f})  ·  "
              f"TPOT {kq['tpot_ms']:.2f} ms (p95 {kq['tpot_p95']:.2f})  ·  "
              f"TPS {kq['tps'] or 0:.1f}  ·  E2E {kq['e2e_ms']:.0f} ms")
        print(f"    VRAM: trong so {kq['vram_trong_so_mb']:.0f} + KV do "
              f"{kq['kv_do_mb']:.0f} + tam {kq['vram_tam_mb']:.0f} = dinh "
              f"{kq['vram_gen_mb']:.0f} MB  ·  hieu dung (TINH) "
              f"{kq['vram_hieu_dung_mb']:.0f} MB  ·  RAM dinh "
              f"{(kq['ram_mb'] or 0):.0f} MB")

    if not ket:
        print("\nKhong phuong phap nao do duoc.")
        sys.exit(1)

    with open(a.out + ".json", "w", encoding="utf-8") as f:
        json.dump(ket, f, indent=2, ensure_ascii=False)
    print("\nDa ghi:", os.path.abspath(a.out + ".json"))

    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font, Alignment, PatternFill
    except ImportError:
        print("(thieu openpyxl — bo qua xlsx)")
        return
    wb = Workbook()
    ws = wb.active
    ws.title = "so"
    COT = [("Phương pháp", "ten"), ("Ngữ cảnh", "ngu_canh_muc_tieu"),
           ("TTFT ms", "ttft_ms"), ("TTFT std", "ttft_std"),
           ("TTFT p95", "ttft_p95"),
           ("TPOT ms", "tpot_ms"), ("TPOT std", "tpot_std"),
           ("TPOT p95", "tpot_p95"),
           ("TPS tok/s", "tps"), ("E2E ms", "e2e_ms"), ("E2E std", "e2e_std"),
           ("VRAM trọng số", "vram_trong_so_mb"), ("KV đo MB", "kv_do_mb"),
           ("VRAM tạm", "vram_tam_mb"), ("VRAM đỉnh sinh", "vram_gen_mb"),
           ("VRAM hiệu dụng (TÍNH)", "vram_hieu_dung_mb"),
           ("RAM đỉnh MB", "ram_mb"), ("Token sinh", "sinh"),
           ("Lần lặp", "lap"), ("Nhân attn", "attn")]
    head = [h for h, _ in COT]
    ws.append(head)
    for c in ws[1]:
        c.font = Font(bold=True, color="14243D", size=10)
        c.alignment = Alignment(horizontal="center", wrap_text=True)
        c.fill = PatternFill("solid", fgColor="F1EEE7")
    for k in ket:
        ws.append([k.get(f) for _, f in COT])
    for i in range(1, len(head) + 1):
        ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = \
            (20 if i == 1 else 12)
    c = ws.cell(row=ws.max_row + 2, column=1,
                value=("Đo trong một lần chạy trên cùng một máy; mỗi phương "
                       "pháp một tiến trình riêng để số VRAM không nhiễm "
                       "phân bổ còn sót. TTFT = prefill; TPOT = (E2E - TTFT)/(token sinh - 1); "
                       "TPS = 1000/TPOT; trung bình qua "
                       f"{a.lap} lần lặp sau {a.warmup} lần khởi động. "
                       "Caption phải ghi rõ tên GPU và độ dài ngữ cảnh."))
    c.font = Font(italic=True, size=9, color="8A8578")
    wb.save(a.out + ".xlsx")
    print("Da ghi:", os.path.abspath(a.out + ".xlsx"))
    print("\nVe hinh (memory + latency + lap token):")
    print(f"  python3 ve_3_hinh.py --toc_do {os.path.basename(a.out)}.json"
          "            # VRAM do that")
    print(f"  python3 ve_3_hinh.py --toc_do {os.path.basename(a.out)}.json"
          " --vram hieu_dung   # kieu bai bao (so tinh)")


if __name__ == "__main__":
    main()
