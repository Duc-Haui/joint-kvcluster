# SQ_VAL_PATCH (tao_tap_valid.py)
import os
import warnings

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# use this for now to filter out torch warnings
warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`",
    category=FutureWarning
)
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "transformers", "src")))

import types
class DummyRMSNorm: pass
class DummyModule(types.ModuleType):
    __file__ = "<dummy>"
    __loader__ = None
    __spec__ = None
    def __getattr__(self, name):
        if name.startswith("__"): raise AttributeError(name)
        return DummyRMSNorm
for mod_name in ['gemma2', 'phi3', 'qwen3', 'qwen2_moe', 'mamba', 'jamba']:
    full_name = f"transformers.models.{mod_name}.modeling_{mod_name}"
    mod = DummyModule(full_name)
    sys.modules[full_name] = mod
    sys.modules[f"transformers.models.{mod_name}"] = DummyModule(f"transformers.models.{mod_name}")

from datasets import load_dataset
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, LlamaConfig, BitsAndBytesConfig
from tqdm import tqdm
import numpy as np
import random
import argparse
import gc
import pickle
import textwrap
import sys
from squeezedattention.utils import build_chat, truncate_fn

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument("--path_to_clusters", type=str, default="/tmp")
    parser.add_argument("--use_centroids", action="store_true")
    parser.add_argument("--hierarchical_lookup", action="store_true")
    parser.add_argument("--percent_clusters", type=int, default=-1)
    parser.add_argument("--percent_clusters_l2", type=int, default=-1)
    parser.add_argument("--percentile", type=float, default=0.5)
    parser.add_argument("--percentile_lower", type=float, default=0.7)
    parser.add_argument("--obs_window", type=int, default=100)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--load_8bit", action="store_true", help="Load model in 8-bit to fit 16GB VRAM")
    parser.add_argument("--load_4bit", action="store_true", help="Load model in 4-bit (W4A16)")
    parser.add_argument("--load_awq", action="store_true", help="Load AWQ quantized model (only for Baseline)")
    parser.add_argument("--awq_model_path", type=str, default=None, help="Path to AWQ quantized model")
    parser.add_argument("--gamma", type=float, default=0.3,
                        help="Hệ số boost value-aware khi retrieve online (chỉ có tác dụng khi "
                             "thư mục cụm có file normalized_variance_*.pt). Phải đặt cùng giá trị "
                             "với --gamma của bước clustering.")
    parser.add_argument("--use_value_aware", type=int, default=1,
                    help="1 = bật VA nếu có file nvar; 0 = tắt hẳn (chạy như Squeezed gốc)")

    parser.add_argument("--limit", type=int, default=-1,
                        help="Chỉ chạy N mẫu đầu tiên (mặc định -1 = toàn bộ). Dùng để chạy thử.")
    parser.add_argument("--keep_old", action="store_true",
                        help="Giữ file .jsonl cũ và ghi nối tiếp. Mặc định là xoá file cũ trước khi chạy.")
    return parser.parse_args(args)

def get_pred(data, max_length, max_gen, prompt_format, prompt_only_format, dataset, device, model_name, model2path, out_path, config_params, load_8bit=False, load_4bit=False, load_awq=False, awq_model_path=None):
    # Khởi tạo mô hình một lần duy nhất
    model, tokenizer = load_model_and_tokenizer(model2path[model_name], model_name, device, config_params, load_8bit, load_4bit, load_awq, awq_model_path)

    # iterate over longbench dataset
    for json_obj in tqdm(data):
        different_prefix_index = json_obj.pop('different_prefix_index')
        prompt_raw = prompt_format.format(**json_obj)
        prompt_noquery_raw = prompt_only_format.format(**json_obj)

        # perform truncation
        prompt, truncated_shared_prefix_length = truncate_fn(prompt_raw, prompt_noquery_raw, tokenizer, max_length, dataset, device)
        model.model.shared_prefix_length = truncated_shared_prefix_length
        model.model.different_prefix_index = different_prefix_index

        # encode input
        input_data = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input_data.input_ids.shape[-1]

        # ─── VÁ LỖ HỔNG 1: DỌN RÁC TRƯỚC KHI GENERATE ───
        del prompt_raw, prompt_noquery_raw, prompt
        gc.collect()
        torch.cuda.empty_cache()
        # ────────────────────────────────────────────────


        with torch.no_grad():
            # Tắt FlashAttention (gây rác chữ trên kiến trúc Ampere/Hopper với mask dị biệt)
            # Bật Mem_Efficient (xFormers) để tiết kiệm RAM, tránh OOM trên L4 24GB thay vì dùng Math O(N^2)
            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=True, enable_math=True):
                if dataset == "samsum": 
                    output = model.generate(
                        **input_data,
                        max_new_tokens=max_gen,
                        num_beams=1,
                        do_sample=False,
                        temperature=1.0,
                        min_length=context_length+1,
                        eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
                        use_cache=True
                    )[0]
                else:
                    output = model.generate(
                        **input_data,
                        max_new_tokens=max_gen,
                        num_beams=1,
                        do_sample=False,
                        temperature=1.0,
                        use_cache=True
                    )[0]
        
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
            f.write('\n')

        # ─── VÁ LỖ HỔNG 1: XÓA BIẾN TẠM SAU KHI GENERATE XONG ───
        del input_data, output
        gc.collect()
        torch.cuda.empty_cache()
        # ────────────────────────────────────────────────────────

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer(path, model_name, device, config_params, load_8bit=False, load_4bit=False, load_awq=False, awq_model_path=None):
    
    if "LLaMA-2-7B-32K" in model_name or "LWM" in model_name or "longchat" in model_name or "TinyLlama" in model_name:

        config_path = awq_model_path if (load_awq and awq_model_path) else path
        config = LlamaConfig.from_pretrained(config_path)
        if load_awq and awq_model_path:
            orig_config = LlamaConfig.from_pretrained(path)
            if hasattr(orig_config, "rope_scaling"):
                config.rope_scaling = orig_config.rope_scaling
            if hasattr(orig_config, "max_position_embeddings"):
                config.max_position_embeddings = orig_config.max_position_embeddings
                
        config._flash_attn_2_enabled = False
        config._attn_implementation = "eager" if config_params["use_centroids"] else "sdpa"

        config.path_to_clusters_cosine = config_params["path_to_clusters_cosine"]
        config.use_centroids           = config_params["use_centroids"]
        config.hierarchical_lookup     = config_params["hierarchical_lookup"]
        config.percent_clusters        = config_params["percent_clusters"]
        config.percent_clusters_l2     = config_params["percent_clusters_l2"]
        config.percentile              = config_params["percentile"]
        config.percentile_lower        = config_params["percentile_lower"]
        config.obs_window              = config_params["obs_window"]
        config.value_aware_gamma       = config_params.get("value_aware_gamma", 0.3)
        config.use_value_aware = config_params.get("use_value_aware", True)


        print(f"Load {model_name} (AWQ: {load_awq}, 8-bit: {load_8bit}, 4-bit: {load_4bit}) | attn={config._attn_implementation}")

        if load_awq:
            model = AutoModelForCausalLM.from_pretrained(
                config_path,
                config=config,
                torch_dtype=torch.float16,
                device_map={"":device},
                low_cpu_mem_usage=True,
            )
        elif load_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=False,
            )
            model = LlamaForCausalLM.from_pretrained(
                path,
                config=config,
                quantization_config=quant_config,
                torch_dtype=torch.float16,
                device_map={"": device},
                low_cpu_mem_usage=True,
            )
        elif load_8bit:
            quant_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
            model = LlamaForCausalLM.from_pretrained(
                path,
                config=config,
                quantization_config=quant_config,
                torch_dtype=torch.float16,
                device_map={"": device},
                low_cpu_mem_usage=True,
            )
        else:
            model = LlamaForCausalLM.from_pretrained(
                path,
                config=config,
                torch_dtype=torch.float16,
                device_map={"": device},
                low_cpu_mem_usage=True,
            )

        # ─── VÁ LỖ HỔNG DEVICE CHO CÁC BUFFER (như inv_freq) ───
        for name, buffer in model.named_buffers():
            buffer.data = buffer.data.to(device)
        # ────────────────────────────────────────────────────────

        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)

    else:
        raise NotImplementedError(f"Model {model_name} chưa được hỗ trợ.")

    model = model.eval()
    return model, tokenizer

if __name__ == '__main__':
    seed_everything(42)
    


    args = parse_args()
    
    # Thanh phan (4) dieu khien bang bien moi truong nen lot qua chot
    # theo args ben duoi. value_centroids chi duoc nap o nhanh KHONG
    # hierarchical (modeling_llama.py), nen neu khong chan thi (4) se
    # im lang khong chay.
    if args.hierarchical_lookup and float(os.environ.get("SQ_VBLEND") or 0) > 0:
        raise NotImplementedError(
            "Thanh phan (4) chua ho tro hierarchical lookup: "
            "value_centroids chi duoc nap o nhanh khong-hierarchical.")
    if args.hierarchical_lookup and args.gamma != 0:
        raise NotImplementedError(
            "Value-aware chua ho tro hierarchical lookup. "
            "Dung --gamma 0 hoac bo --hierarchical_lookup."
        )
    
    # ─── VÁ LỖ HỔNG 2: ÉP CHẠY SINGLE THREAD ───
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    # ───────────────────────────────────────────

    model2path = json.load(open("config/model2path.json", "r"))
    # Chú ý: Đảm bảo file model2maxlen_pred.json có chứa "LLaMA-2-7B-32K": 7000
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    
    model_name = args.model
    max_length = model2maxlen[model_name]

    if args.task is not None:
        datasets = [args.task]
    else:
        if args.e:
            datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
        else:
            datasets = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", \
                        "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum", \
                        "lcc", "repobench-p"]

    config_params = {}
    config_params['use_centroids'] = args.use_centroids
    config_params['hierarchical_lookup'] = args.hierarchical_lookup
    config_params['percent_clusters'] = args.percent_clusters
    config_params['percent_clusters_l2'] = args.percent_clusters_l2
    config_params['percentile'] = args.percentile
    config_params['percentile_lower'] = args.percentile_lower
    config_params['obs_window'] = args.obs_window
    config_params['value_aware_gamma'] = args.gamma
    config_params['use_value_aware'] = bool(args.use_value_aware)

    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))

    # Cho phép đổi thư mục output qua biến môi trường PRED_DIR (mặc định: pred)
    base_dir = "pred_e" if args.e else "pred"
    PRED_DIR = os.environ.get("PRED_DIR", base_dir)
    if not os.path.exists(PRED_DIR): os.makedirs(PRED_DIR)

    for dataset in datasets:
        print('\n=========================================')
        print(f'dataset: {dataset}')
        print('=========================================')

        config_params['path_to_clusters_cosine'] = os.path.join(args.path_to_clusters, dataset) + os.sep
        
        if args.e:
            data = load_dataset('THUDM/LongBench', f"{dataset}_e", split='test')
        else:
            data = (load_dataset('json', data_files=os.path.join(os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')), 'SqueezedAttention-main/LongBench/data_val'), dataset + '.jsonl'), split='train') if dataset.endswith('_val') else load_dataset('THUDM/LongBench', dataset, split='test'))

        if not args.use_centroids:
            savepath = f"{PRED_DIR}/{model_name}_baseline"
        else:
            if args.hierarchical_lookup:
                savepath = f"{PRED_DIR}/{model_name}_PC1_{args.percent_clusters}_PERC1_{args.percentile}_PC2_{args.percent_clusters_l2}_PERC2_{args.percentile_lower}_lookup"
            else:
                savepath = f"{PRED_DIR}/{model_name}_PC{args.percent_clusters}_PERC{args.percentile}"

        if not os.path.exists(savepath): os.makedirs(savepath)
        out_path = savepath + f"/{dataset}.jsonl"

        prompt_format = dataset2prompt[dataset]
        prompt_only_format = dataset2prompt[dataset + '_prompt']
        max_gen = dataset2maxlen[dataset]
        
        data_all = [data_sample for data_sample in data]
        for i in range(len(data_all)):
            data_all[i]['different_prefix_index'] = i

        # Chỉ chạy N mẫu đầu khi chạy thử
        if args.limit is not None and args.limit > 0:
            data_all = data_all[:args.limit]
            print(f"[LIMIT] Chỉ chạy {len(data_all)} mẫu đầu tiên")

        # get_pred mở file ở chế độ "a" (append) cho từng mẫu, nên nếu không xoá
        # thì lần chạy mới sẽ NỐI vào kết quả cũ và eval.py chấm trên hỗn hợp
        # của hai lần chạy khác cấu hình.
        if os.path.exists(out_path) and not args.keep_old:
            os.remove(out_path)
            print(f"[CLEAN] Đã xoá kết quả cũ: {out_path}")

        # ─── GỌI HÀM TRỰC TIẾP, BỎ ĐA TIẾN TRÌNH ───
        get_pred(data_all, max_length, max_gen, prompt_format, prompt_only_format, dataset, device, model_name, model2path, out_path, config_params, args.load_8bit, args.load_4bit, args.load_awq, args.awq_model_path)
        # ───────────────────────────────────────────