"""
Script lượng tử hóa model LLaMA-2-7B-32K sang AWQ 4-bit.

Yêu cầu:
    pip install autoawq

Cách chạy:
    python quantize_awq.py

Thời gian: ~1-2 tiếng trên GPU (Quadro P5000)
Kết quả: Thư mục model AWQ tại OUTPUT_PATH (~3.5GB)
"""

import argparse
import os
import torch

# ─── PATCH: Fix lỗi "cannot import name 'BaseImageProcessor' from 'transformers'"
# AutoAWQ cần BaseImageProcessor từ top-level transformers namespace,
# nhưng lazy import của transformers 4.40.0 đôi khi không export nó.
# Ta import thủ công từ submodule rồi gắn vào namespace chính.
import transformers
from transformers.image_processing_utils import BaseImageProcessor
transformers.BaseImageProcessor = BaseImageProcessor
# ─── END PATCH ───

def main():
    parser = argparse.ArgumentParser(description="Quantize LLaMA-2-7B-32K to AWQ 4-bit")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/media/an/720CDC460CDC074F/DucDang/SqueezedAttention/models/LLaMA-2-7B-32K",
        help="Đường dẫn đến model gốc cần lượng tử hóa."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/media/an/720CDC460CDC074F/DucDang/SqueezedAttention/models/LLaMA-2-7B-32K-AWQ",
        help="Đường dẫn lưu model AWQ đã lượng tử hóa"
    )
    parser.add_argument(
        "--w_bit", type=int, default=4,
        help="Số bit cho weights (mặc định: 4)"
    )
    parser.add_argument(
        "--q_group_size", type=int, default=128,
        help="Kích thước nhóm lượng tử hóa (mặc định: 128)"
    )
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"  AWQ QUANTIZATION: LLaMA-2-7B-32K")
    print(f"{'='*60}")
    print(f"  Input model:  {args.model_path}")
    print(f"  Output path:  {args.output_path}")
    print(f"  Bits:         {args.w_bit}")
    print(f"  Group size:   {args.q_group_size}")
    print(f"  GPU:          {torch.cuda.get_device_name(0)}")
    print(f"  VRAM:         {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    print(f"{'='*60}")

    # 1. Import AutoAWQ
    try:
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer
    except Exception as e:
        print(f"❌ Lỗi khi import AutoAWQ: {e}")
        print("Vui lòng kiểm tra lại quá trình cài đặt.")
        return

    # 2. Load model FP16 gốc
    print("\n📥 Đang load model FP16 gốc...")
    model = AutoAWQForCausalLM.from_pretrained(
        args.model_path, 
        safetensors=False,
        low_cpu_mem_usage=True,
        use_cache=False
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    print("✅ Load xong!")

    # 3. Cấu hình AWQ
    quant_config = {
        "zero_point": True,
        "q_group_size": args.q_group_size,
        "w_bit": args.w_bit,
        "version": "GEMM"
    }

    # 4. Chạy lượng tử hóa (calibration + quantize)
    print(f"\n⏳ Đang lượng tử hóa AWQ (mất ~1-2 tiếng)...")
    print(f"   Calibration sẽ tự động dùng dataset 'pileval' từ MIT.")
    model.quantize(tokenizer, quant_config=quant_config)
    print("✅ Lượng tử hóa xong!")

    # 5. Lưu model
    print(f"\n💾 Đang lưu model AWQ tại: {args.output_path}")
    os.makedirs(args.output_path, exist_ok=True)
    model.save_quantized(args.output_path)
    tokenizer.save_pretrained(args.output_path)
    print("✅ Đã lưu xong!")

    # 6. Kiểm tra kích thước
    total_size = sum(
        os.path.getsize(os.path.join(args.output_path, f))
        for f in os.listdir(args.output_path)
        if os.path.isfile(os.path.join(args.output_path, f))
    )
    print(f"\n📊 Kích thước model AWQ: {total_size / (1024**3):.2f} GB")
    print(f"   (So với FP16 gốc: ~13.5 GB → tiết kiệm ~{(1 - total_size/(13.5*1024**3))*100:.0f}%)")
    print(f"\n{'='*60}")
    print(f"  HOÀN TẤT! Model AWQ sẵn sàng tại:")
    print(f"  {args.output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
