FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Bỏ qua các prompt tương tác trong quá trình cài đặt apt
ENV DEBIAN_FRONTEND=noninteractive

# Cài đặt các công cụ cơ bản
RUN apt-get update && apt-get install -y \
    wget \
    git \
    vim \
    curl \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt Miniconda
ENV MINICONDA_VERSION=latest
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p ${CONDA_DIR} && \
    rm ~/miniconda.sh

# Cập nhật biến môi trường cho Conda
ENV PATH=${CONDA_DIR}/bin:${PATH}

# Khởi tạo conda cho bash
RUN conda init bash

# Copy file environment.yml vào container để tạo môi trường
WORKDIR /app
COPY environment.yml /app/environment.yml

# Chấp nhận Điều khoản Dịch vụ của Anaconda (Tránh lỗi CondaToSNonInteractiveError)
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Cài đặt Mamba để tăng tốc độ tải và chống treo mạng
RUN conda install -n base -c conda-forge mamba -y && \
    conda clean -afy

# Tạo môi trường sq310 từ file environment.yml bằng Mamba
# Thêm lệnh clean ngay trong cùng layer để tránh lỗi "Read-only file system" do đầy dung lượng ảo trên Docker Windows (WSL2)
RUN mamba env create -f environment.yml && \
    mamba clean -afy && \
    rm -rf /opt/conda/pkgs/*

# Kích hoạt sẵn môi trường sq310 mỗi khi mở bash
RUN echo "conda activate sq310" >> ~/.bashrc
ENV PATH=${CONDA_DIR}/envs/sq310/bin:${PATH}

# Cài đặt đè numpy và cupy để khắc phục lỗi xung đột Numpy 2.0 và CuPy
# Đồng thời cài thêm thư viện runtime và nvrtc cho CUDA 12, sau đó xoá cache pip
RUN /bin/bash -c "source activate sq310 && \
    pip install 'numpy<2.0' cupy-cuda12x==12.3.0 nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12 && \
    pip cache purge && rm -rf /root/.cache/pip"

# Khởi chạy bash làm mặc định
CMD ["/bin/bash"]
