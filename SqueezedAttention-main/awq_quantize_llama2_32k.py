import os
import torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# Cấu hình đường dẫn
MODEL_PATH = "/media/an/720CDC460CDC074F/DucDang/SqueezedAttention/models/LLaMA-2-7B-32K"
QUANT_PATH = "/media/an/720CDC460CDC074F/DucDang/SqueezedAttention/models/LLaMA-2-7B-32K-AWQ"

# Cấu hình lượng tử hóa (AWQ 4-bit)
quant_config = {
    "zero_point": True,         # Sử dụng zero point để độ chính xác cao hơn
    "q_group_size": 128,        # Kích thước nhóm (phổ biến nhất, tương thích tốt nhất)
    "w_bit": 4,                 # Ép xuống 4-bit
    "version": "GEMM"           # GEMM là version tối ưu tốt nhất hiện tại cho inference
}

print(f"1. Đang tải model gốc từ: {MODEL_PATH}")
print("Có thể mất vài phút vì model 7B FP16 nặng khoảng 14GB...")
model = AutoAWQForCausalLM.from_pretrained(
    MODEL_PATH, 
    low_cpu_mem_usage=True, 
    use_cache=False
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)

print("\n2. Đang tiến hành lượng tử hóa AWQ (Calibration & Quantization)...")
print("Quá trình này sẽ tải tập dữ liệu mẫu (mặc định là pile-val) để đo lường token quan trọng.")
print("Quá trình này có thể mất từ 30 phút - 1 tiếng tùy tốc độ GPU...")
model.quantize(tokenizer, quant_config=quant_config)

print(f"\n3. Đang lưu mô hình đã lượng tử hóa vào: {QUANT_PATH}")
os.makedirs(QUANT_PATH, exist_ok=True)
model.save_quantized(QUANT_PATH)
tokenizer.save_pretrained(QUANT_PATH)

print("\n✅ HOÀN TẤT! Bây giờ bạn đã có thể dùng mô hình LLaMA-2-7B-32K-AWQ siêu nhanh!")
print("Trong run_all_experiments_integrated.sh, hãy đổi MODEL_PATH sang thư mục này và bỏ cờ --load_4bit đi.")
