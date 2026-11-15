# SQ_VAL_PATCH (tao_tap_valid.py)
"""
Patched offline clustering script tương thích với repo gốc SqueezedAttention.

So với offline_clustering.py gốc:
1. Dùng `run_value_aware_clustering` từ value_aware package thay vì run_clustering gốc
2. Lưu thêm value_centroids và normalized_variance
3. Thêm CLI args: --alpha, --beta, --gamma

GIỮ NGUYÊN tất cả kỹ thuật offloading VRAM từ file gốc:
- Hook offload QKV sang CPU ngay lập tức
- Clustering chạy trên CPU
- del input_ids + torch.cuda.empty_cache() sau mỗi sample
- output_attentions=False
- sdpa attention backend
- Tên file output tương thích pred.py

Cách dùng:
    python offline_clustering_value_aware.py LLaMA-2-7B-32K \\
        --dataset 2wikimqa \\
        --output_path /tmp/clusters/2wikimqa/ \\
        --percent_clusters 5 \\
        --observation_window 100 \\
        --alpha 1.0 --beta 0.5 --gamma 0.3 \\
        --device 0
"""

import argparse
import json
import os
import sys
import time
import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "transformers", "src")))

class DummyRMSNorm: pass
class DummyModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"): raise AttributeError(name)
        return DummyRMSNorm
for mod_name in ['gemma2', 'phi3', 'qwen3', 'qwen2_moe', 'mamba', 'jamba']:
    full_name = f"transformers.models.{mod_name}.modeling_{mod_name}"
    mod = DummyModule(full_name)
    sys.modules[full_name] = mod
    sys.modules[f"transformers.models.{mod_name}"] = DummyModule(f"transformers.models.{mod_name}")

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

# Local imports - giả định chạy từ root của repo SqueezedAttention
from utils.modelutils import *  # noqa
from utils.datautils import *   # noqa
from utils.model_parse import parse_model, get_layers
from squeezedattention.utils import build_chat, truncate_fn

# Value-aware extension - đảm bảo `value_aware` package nằm trên PYTHONPATH
from value_aware.clustering import run_value_aware_clustering
from value_aware.threshold import run_value_aware_global_threshold


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("model", type=str, help="Model name (key trong model2path.json)")
    p.add_argument("--output_path", type=str, default="output/")
    p.add_argument(
        "--dataset", type=str, default="trec",
        choices=[
            "narrativeqa", "qasper", "qasper_val", "narrativeqa_val", "multifieldqa_en", "hotpotqa", "2wikimqa",
            "musique", "gov_report", "qmsum", "multi_news", "trec", "triviaqa",
            "samsum", "lcc", "repobench-p",
        ],
    )
    p.add_argument("--percent_clusters", type=int, default=5,
                   help="% centroids so với fixed context length")
    p.add_argument("--observation_window", type=int, default=100)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--load_4bit", action="store_true", help="Use 4-bit quantization")
    p.add_argument("--load_8bit", action="store_true", help="Use 8-bit quantization")
    p.add_argument("--load_awq", action="store_true", help="Load AWQ 4-bit model")
    p.add_argument("--awq_model_path", type=str, default="", help="Path to AWQ model")
    # Value-aware specific
    p.add_argument("--alpha", type=float, default=1.0,
                   help="Trọng số K trong joint K-V clustering")
    p.add_argument("--beta", type=float, default=0.5,
                   help="Trọng số V. 0 = tắt value-aware (về baseline gốc)")
    p.add_argument("--gamma", type=float, default=0.3,
                   help="Hệ số boost variance khi tính threshold/retrieve")
    p.add_argument("--kmeans_iters", type=int, default=100)
    p.add_argument("--max_length", type=int, default=0,
                   help="Do dai cat prompt khi gom cum. 0 = model2maxlen.json "
                        "(giong pred.py). Dat 1024 de gom cum cho phep do "
                        "latency/memory o 1K (do_toc_do.py --ngu_canh 1024).")
    p.add_argument("--max_samples", type=int, default=-1,
                   help="Giới hạn số sample. -1 = tất cả")
    p.add_argument("--seed", type=int,
                   default=int(os.environ.get("SQ_SEED", "0")),
                   help="Seed cho k-means init. Đổi seed -> cluster khác -> "
                        "dùng để chạy lặp lấy mean/std. Mặc định lấy từ $SQ_SEED.")
    return p.parse_args()


def main():
    args = parse_args()
    DEV = torch.device(f"cuda:{args.device}")

    # Load config
    model2path = json.load(open("LongBench/config/model2path.json", "r"))
    model2maxlen = json.load(open("LongBench/config/model2maxlen.json", "r"))
    model_path = model2path[args.model]
    max_length = args.max_length if args.max_length > 0 else model2maxlen[args.model]
    print(f"max_length = {max_length}")

    print(f"=== Value-Aware Squeezed Attention Offline Clustering ===")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Config: alpha={args.alpha}, beta={args.beta}, gamma={args.gamma}, seed={args.seed}")
    print(f"Centroids: {args.percent_clusters}% of context length")
    print(f"Output: {args.output_path}")
    print()

    # ─── Load model (GIỮ NGUYÊN config từ file gốc) ─────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    config = LlamaConfig.from_pretrained(model_path)
    config.return_qkv_states = True
    config._flash_attn_2_enabled = False    # ← giống file gốc (sdpa tiết kiệm VRAM hơn)
    config._attn_implementation = "sdpa"    # ← giống file gốc
    
    kwargs = {
        "config": config,
        "torch_dtype": torch.float16,
        "device_map": {"": DEV},
        "trust_remote_code": True,
        "attn_implementation": "sdpa",
    }
    
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=False,
        )
    elif args.load_8bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
        )

    if args.load_awq:
        # Lấy file config gốc đã chỉnh sửa từ thư mục AWQ
        custom_config = LlamaConfig.from_pretrained(args.awq_model_path)
        orig_config = LlamaConfig.from_pretrained(model_path)
        if hasattr(orig_config, "rope_scaling"): custom_config.rope_scaling = orig_config.rope_scaling
        if hasattr(orig_config, "max_position_embeddings"): custom_config.max_position_embeddings = orig_config.max_position_embeddings
        custom_config.return_qkv_states = True
        custom_config._flash_attn_2_enabled = False
        custom_config._attn_implementation = "sdpa"
        
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            args.awq_model_path,
            config=custom_config,
            torch_dtype=torch.float16,
            device_map={"": DEV},
            low_cpu_mem_usage=True,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            model_path,
            **kwargs
        )
    model.eval()

    model_type = parse_model(model)
    layers = get_layers(model, model_type)

    # Load LongBench dataset
    from datasets import load_dataset
    dataset = args.dataset
    dataset_name_prompt = dataset + "_prompt"
    data = (load_dataset('json', data_files=os.path.join(os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')), 'SqueezedAttention-main/LongBench/data_val'), dataset + '.jsonl'), split='train') if dataset.endswith('_val') else load_dataset("THUDM/LongBench", dataset, split="test"))

    dataset2prompt = json.load(open("LongBench/config/dataset2prompt.json", "r"))
    prompt_format = dataset2prompt[dataset]
    prompt_only_format = dataset2prompt[dataset_name_prompt]

    # Compute shared prefix lengths
    data_all = list(data)
    if args.max_samples > 0:
        data_all = data_all[: args.max_samples]
    shared_prefix_length = {}
    for i, sample in enumerate(data_all):
        prompt = prompt_format.format(**sample)
        prompt_only = prompt_only_format.format(**sample)
        prompt, sp_len = truncate_fn(
            prompt, prompt_only, tokenizer, max_length, dataset, DEV
        )
        shared_prefix_length[i] = sp_len
        assert sp_len > 0

    # ─── Hooks để collect K, V, Q (OFFLOAD SANG CPU NGAY) ────────────────────
    all_queries_layers = []
    all_keys_layers = []
    all_values_layers = []

    def hook(module, inp, out):
        _, qkv, _ = out
        queries, keys, values = qkv
        sp_len = shared_prefix_length[dataidx]

        queries = queries[:, :, :sp_len]
        keys = keys[:, :, :sp_len]
        values = values[:, :, :sp_len]

        all_queries_layers.append(queries)
        all_keys_layers.append(keys)
        all_values_layers.append(values)

    for layer in layers:
        layer.self_attn.register_forward_hook(hook)

    os.makedirs(args.output_path, exist_ok=True)

    # ─── Loop qua samples ────────────────────────────────────────────────────
    for dataidx, d in enumerate(tqdm(data_all)):
        # Seed k-means CO DINH theo (seed, dataidx) — GIONG HET repo goc
        # (offline_clustering.py:228). Khong co dong nay thi hai nhanh
        # khoi tao tu hai diem khac nhau va beta=0 KHONG suy bien ve goc.
        torch.manual_seed(args.seed * 1000003 + dataidx)
        all_queries_layers.clear()
        all_keys_layers.clear()
        all_values_layers.clear()

        prompt = prompt_format.format(**d)
        prompt_only = prompt_only_format.format(**d)
        prompt, _ = truncate_fn(
            prompt, prompt_only, tokenizer, max_length, dataset, DEV
        )
        input_ids = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids.to(DEV)

        print(f"dataidx: {dataidx} | length of input_ids: {len(input_ids[0])}")
        print(f"dataidx: {dataidx} | shared_prefix_length: {shared_prefix_length[dataidx]}")

        # Forward pass — GIỮ NGUYÊN setting từ file gốc
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids,
                do_sample=False,
                max_new_tokens=1,
                use_cache=False,            # ← giống file gốc (tiết kiệm VRAM)
                output_attentions=False,     # ← giống file gốc (False, không phải True)
            )

        # [QUAN TRỌNG] Free VRAM ngay sau hook capture — giống file gốc
        del generated_ids, input_ids
        torch.cuda.empty_cache()

        # Số centroid
        sp_len = shared_prefix_length[dataidx]
        percentage = ((args.percent_clusters * 1.0) / 100.0)
        num_clusters = int(percentage * (sp_len - args.observation_window))
        if num_clusters < 1:
            num_clusters = 1
        print(num_clusters)

        # ─── Báo cáo bộ nhớ (giống file gốc) ────────────────────────────────
        tokens_kept = args.observation_window + num_clusters
        kv_budget_percent = (tokens_kept / sp_len) * 100

        print("\n" + "=" * 50)
        print(f" BÁO CÁO NHANH - MẪU DỮ LIỆU {dataidx}")
        print(f"  - Tổng chiều dài ngữ cảnh gốc : {sp_len} tokens")
        print(f"  - Số tokens giữ lại (KV Cache) : {tokens_kept} tokens")
        print(f"  --> KV Budget sử dụng         : {kv_budget_percent:.2f}% (Nén được {100 - kv_budget_percent:.2f}%)")

        try:
            import psutil
            ram_used_gb = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)
            print(f" TIÊU THỤ BỘ NHỚ:")
            print(f"  - System RAM (CPU) đang dùng   : {ram_used_gb:.2f} GB")
        except ImportError:
            pass

        torch.cuda.empty_cache()
        vram_allocated_gb = torch.cuda.memory_allocated(DEV) / (1024 ** 3)
        vram_peak_gb = torch.cuda.max_memory_allocated(DEV) / (1024 ** 3)
        print(f"  - GPU VRAM đang dùng (Hiện tại): {vram_allocated_gb:.2f} GB")
        print(f"  - GPU VRAM Đỉnh (Peak)         : {vram_peak_gb:.2f} GB")
        print("=" * 50 + "\n")

        t0 = time.time()

        kc_dict, vc_dict, lbl_dict, vvar_dict, nvar_dict = run_value_aware_clustering(
            all_keys_layers,
            all_values_layers,
            num_clusters=num_clusters,
            observation_window=args.observation_window,
            alpha=args.alpha,
            beta=args.beta,
            num_iters=args.kmeans_iters,
            print_log=False,
            device=DEV,
            seed_base=args.seed,
        )

        global_threshold_dict = run_value_aware_global_threshold(
            keys_layers=all_keys_layers,
            queries_layers=all_queries_layers,
            key_centroids_dict=kc_dict,
            labels_dict=lbl_dict,
            normalized_variance_dict=nvar_dict,
            num_clusters=num_clusters,
            observation_window=args.observation_window,
            gamma=args.gamma,
            device=DEV,
        )
        torch.cuda.synchronize(DEV)  # cho GPU chay xong roi moi chot gio
        clustering_time = time.time() - t0
        # Ghi thoi gian (k-means + nguong) cua TUNG context, cho Table III.
        with open(os.path.join(args.output_path, "clustering_time.jsonl"),
                  "a", encoding="utf-8") as _f:
            _f.write(json.dumps({"dataidx": dataidx, "sp_len": sp_len,
                                 "num_clusters": num_clusters,
                                 "beta": args.beta, "gamma": args.gamma,
                                 "giay": clustering_time}) + "\n")

        # ─── Save kết quả (GPU) ──────────────────────────────────────────────
        # TÊN FILE PHẢI TƯƠNG THÍCH VỚI pred.py CỦA REPO GỐC
        os.makedirs(args.output_path, exist_ok=True)
        # Bỏ qua việc chuyển về CPU trước khi save để tăng tốc độ ghi file
        pass

        # File chính — tên giống hệt file gốc để pred.py đọc được
        torch.save(kc_dict, f'{args.output_path}/centroids_tensor_dict_{dataidx}_{num_clusters}.pt')
        torch.save(lbl_dict, f'{args.output_path}/centroids_labels_dict_{dataidx}_{num_clusters}.pt')
        torch.save(global_threshold_dict, f'{args.output_path}/global_threshold_{dataidx}_{num_clusters}.pt')

        # File bổ sung — value-aware specific (pred.py gốc bỏ qua, dùng cho online VA)
        torch.save(vc_dict, f'{args.output_path}/value_centroids_{dataidx}_{num_clusters}.pt')
        torch.save(vvar_dict, f'{args.output_path}/value_variance_{dataidx}_{num_clusters}.pt')
        torch.save(nvar_dict, f'{args.output_path}/normalized_variance_{dataidx}_{num_clusters}.pt')

        # ─── Cleanup ─────────────────────────────────────────────────────────
        n_layers = len(all_keys_layers)
        for _ in range(n_layers):
            del all_queries_layers[0]
            del all_keys_layers[0]
            del all_values_layers[0]
        torch.cuda.empty_cache()

        if dataidx == 0:
            print(f"  [first sample] clustering took {clustering_time:.2f}s, "
                  f"thresholds: {global_threshold_dict}")

    print(f"\nDone. Results saved to {args.output_path}")


if __name__ == "__main__":
    main()
