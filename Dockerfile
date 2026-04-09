# 1. 基础镜像（使用 NVIDIA CUDA 13.0 基础镜像，支持 Blackwell sm_120 架构）
FROM nvidia/cuda:13.0.0-cudnn-runtime-ubuntu22.04

# 2. 设置工作目录
WORKDIR /pain_detection

# 3. 更换为清华源（解决网络问题）
RUN sed -i 's@//.*archive.ubuntu.com@//mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list && \
    sed -i 's@//.*security.ubuntu.com@//mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list

# 4. 安装系统依赖和 Python
RUN apt update && apt install -y \
    libgl1-mesa-glx libglib2.0-0 build-essential \
    python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# 5. 安装 PyTorch 2.11.0（支持 Blackwell sm_120）
RUN pip3 install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu130

# 6. 安装其他Python依赖（换清华源，加速下载）
RUN pip3 install --no-cache-dir \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn \
    pytorchvideo \
    opencv-python \
    pandas \
    scikit-learn \
    timm \
    pillow \
    insightface \
    onnxruntime-gpu

CMD ["/bin/bash"]