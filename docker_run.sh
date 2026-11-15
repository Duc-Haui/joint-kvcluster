#!/bin/bash
# ==============================================================================
# HƯỚNG DẪN BUILD & RUN DOCKER IMAGE
# ==============================================================================
# Yêu cầu máy host:
#   1. Docker Desktop hoặc Docker Engine
#   2. NVIDIA Container Toolkit (nvidia-docker2)
#   3. NVIDIA Driver >= 530 (cho CUDA 12.1)
# ==============================================================================

IMAGE_NAME="squeezed-attention"
CONTAINER_NAME="sq-runner"

# ─── Đường dẫn model trên máy HOST (THAY ĐỔI CHO PHÙ HỢP) ──────────────
# Đây là thư mục chứa model LLaMA trên máy mới
MODEL_DIR="${MODEL_DIR:-$HOME/models}"

# ==============================================================================
# BƯỚC 0: Cài đặt NVIDIA Container Toolkit (chỉ chạy 1 lần)
# ==============================================================================
install_nvidia_docker() {
    echo "🔧 Cài đặt NVIDIA Container Toolkit..."
    
    # Ubuntu/Debian
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
        sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    
    echo "✅ NVIDIA Container Toolkit đã cài xong!"
}

# ==============================================================================
# BƯỚC 1: BUILD DOCKER IMAGE
# ==============================================================================
build_image() {
    echo "🏗️  Building Docker image: ${IMAGE_NAME}..."
    echo "   (Quá trình này có thể mất 30-60 phút lần đầu)"
    echo ""
    
    docker build \
        --tag ${IMAGE_NAME}:latest \
        --progress=plain \
        .
    
    echo ""
    echo "✅ Build thành công! Image: ${IMAGE_NAME}:latest"
    docker images ${IMAGE_NAME}
}

# ==============================================================================
# BƯỚC 2: CHẠY CONTAINER (Interactive - vào terminal)
# ==============================================================================
run_interactive() {
    echo "🚀 Khởi chạy container (interactive mode)..."
    
    docker run -it --rm \
        --name ${CONTAINER_NAME} \
        --gpus all \
        --shm-size=16g \
        -v ${MODEL_DIR}:/workspace/models \
        -v $(pwd)/results:/workspace/results \
        ${IMAGE_NAME}:latest \
        /bin/bash
}

# ==============================================================================
# BƯỚC 3: CHẠY PIPELINE TỰ ĐỘNG (chạy xong tự tắt)
# ==============================================================================
run_pipeline() {
    echo "🚀 Chạy pipeline tự động..."
    
    # Tạo thư mục results trên host
    mkdir -p results
    
    docker run --rm \
        --name ${CONTAINER_NAME} \
        --gpus all \
        --shm-size=16g \
        -v ${MODEL_DIR}:/workspace/models \
        -v $(pwd)/results:/workspace/results \
        -e MODEL_PATH=/workspace/models/LLaMA-2-7B-32K \
        -e AWQ_MODEL_PATH=/workspace/models/LLaMA-2-7B-32K-AWQ \
        ${IMAGE_NAME}:latest \
        bash -c "cd /workspace/SqueezedAttention-main && ./run_all_experiments_integrated.sh"
}

# ==============================================================================
# MENU
# ==============================================================================
echo "========================================="
echo "  SQUEEZED ATTENTION - DOCKER MANAGER"
echo "========================================="
echo ""
echo "Chọn hành động:"
echo "  0) Cài NVIDIA Container Toolkit (chỉ lần đầu)"
echo "  1) Build Docker image"
echo "  2) Chạy container (interactive - vào terminal)"
echo "  3) Chạy pipeline tự động"
echo ""
read -p "Nhập số (0-3): " choice

case $choice in
    0) install_nvidia_docker ;;
    1) build_image ;;
    2) run_interactive ;;
    3) run_pipeline ;;
    *) echo "❌ Lựa chọn không hợp lệ" ;;
esac
