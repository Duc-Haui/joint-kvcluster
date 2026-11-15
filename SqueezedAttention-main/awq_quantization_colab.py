import os
import json
import random
import inspect
import numpy as np
import torch
import transformers
import transformers.activations

# ─── SEED: cố định để lần lượng tử hoá sau này tái lập được ──────────────────
# AutoAWQ lấy ngẫu nhiên các mẫu hiệu chuẩn, nên không seed thì mỗi lần chạy ra
# một bộ trọng số khác nhau — không viết vào báo cáo được.
SEED = 42
os.environ.setdefault("PYTHONHASHSEED", str(SEED))
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Bỏ qua lỗi thiếu PytorchGELUTanh ở các bản transformers mới
if not hasattr(transformers.activations, 'PytorchGELUTanh'):
    class PytorchGELUTanh(torch.nn.Module):
        def forward(self, input):
            return torch.nn.functional.gelu(input, approximate="tanh")
    transformers.activations.PytorchGELUTanh = PytorchGELUTanh

import awq
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

MODEL_PATH = "/content/LLaMA-2-7B-32K"
QUANT_PATH = "/content/LLaMA-2-7B-32K-AWQ"

# ─── Độ dài chuỗi hiệu chuẩn ─────────────────────────────────────────────────
# Mặc định của AutoAWQ là 512 token, trong khi ngữ cảnh qasper có trung vị
# ~3.400 token và 100% mẫu dài hơn 512. AWQ chọn scale theo biên độ activation
# quan sát lúc hiệu chuẩn, nên hiệu chuẩn quá ngắn so với lúc chạy là điểm yếu
# dễ bị hỏi khi bảo vệ. 2048 là thoả hiệp giữa độ khớp và thời gian chạy.
# Đặt lại = 512 nếu muốn bám sát cấu hình mặc định của thư viện.
MAX_CALIB_SEQ_LEN = 2048
MAX_CALIB_SAMPLES = 128

quant_config = {
    "zero_point": True,
    "q_group_size": 128,   # chia hết 4096 và 11008 → không có nhóm lẻ
    "w_bit": 4,
    "version": "GEMM"
}

# Ghi lại phiên bản để đưa vào phần môi trường của báo cáo
print("torch:", torch.__version__,
      "| transformers:", transformers.__version__,
      "| autoawq:", getattr(awq, "__version__", "n/a"),
      "| seed:", SEED)

# safetensors: tự dò thay vì ép cứng False — thư mục model có thể chỉ có
# .safetensors, khi đó ép False sẽ gãy lúc nạp.
has_safetensors = any(f.endswith(".safetensors") for f in os.listdir(MODEL_PATH))
print(f"1. Đang tải model gốc từ: {MODEL_PATH} (safetensors={has_safetensors})")
model = AutoAWQForCausalLM.from_pretrained(
    MODEL_PATH,
    low_cpu_mem_usage=True,
    use_cache=False,
    safetensors=has_safetensors,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)

# Tên tham số hiệu chuẩn đổi giữa các phiên bản AutoAWQ, nên chỉ truyền những
# tham số mà bản đang cài thực sự khai báo — tránh TypeError giữa chừng.
wanted = {
    "max_calib_seq_len": MAX_CALIB_SEQ_LEN,
    "max_calib_samples": MAX_CALIB_SAMPLES,
}
supported = set(inspect.signature(model.quantize).parameters)
calib_kwargs = {k: v for k, v in wanted.items() if k in supported}
skipped = [k for k in wanted if k not in supported]
if skipped:
    print(f"   [chú ý] bản AutoAWQ này không có tham số {skipped} → dùng mặc định thư viện")

print(f"\n2. Đang tiến hành lượng tử hóa AWQ... (calib: {calib_kwargs or 'mặc định'})")
model.quantize(tokenizer, quant_config=quant_config, **calib_kwargs)

print(f"\n3. Đang lưu mô hình đã lượng tử hóa vào: {QUANT_PATH}")
os.makedirs(QUANT_PATH, exist_ok=True)
model.save_quantized(QUANT_PATH)
tokenizer.save_pretrained(QUANT_PATH)

# Kiểm tra sau khi lưu: rope_scaling phải còn trong config. Nếu mất, model 32K
# vẫn nạp bình thường nhưng hỏng âm thầm ở ngữ cảnh dài.
with open(os.path.join(QUANT_PATH, "config.json")) as f:
    saved_cfg = json.load(f)
print("   rope_scaling:", saved_cfg.get("rope_scaling"))
print("   max_position_embeddings:", saved_cfg.get("max_position_embeddings"))
if saved_cfg.get("rope_scaling") is None:
    print("   ⚠️  KHÔNG thấy rope_scaling — model 32K sẽ hỏng ở ngữ cảnh dài, kiểm tra lại config gốc!")

print("\n✅ HOÀN TẤT AWQ!")



# import json, shutil, os
# SRC, DST = '/content/LLaMA-2-7B-32K', '/content/LLaMA-2-7B-32K-AWQ'
# src = json.load(open(f'{SRC}/config.json'))
# dst = json.load(open(f'{DST}/config.json'))
# for k in ('rope_scaling', 'max_position_embeddings', 'rope_theta'):
#     if src.get(k) is not None and dst.get(k) != src.get(k):
#         print(f"khôi phục {k}: {dst.get(k)} -> {src.get(k)}")
#         dst[k] = src[k]
# json.dump(dst, open(f'{DST}/config.json', 'w'), indent=2)
# print("sau khi vá:", json.load(open(f'{DST}/config.json')).get('rope_scaling'))
# sau khi vá: {'factor': 8.0, 'type': 'linear'}


# for f in ['tokenizer.model','special_tokens_map.json','added_tokens.json','tokenizer_config.json','tokenizer.json']:
#     p = f'{SRC}/{f}'
#     if os.path.exists(p):
#         shutil.copy2(p, f'{DST}/{f}'); print('copied', f)

# copied tokenizer.model
# copied special_tokens_map.json
# copied added_tokens.json
# copied tokenizer_config.json
# copied tokenizer.json


