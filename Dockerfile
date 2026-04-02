# 1. 基础镜像（PyTorch 2.5 + CUDA 12.4，支持新 GPU 架构）
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# 2. 设置工作目录
WORKDIR /pain_detection

# 3. 更换为清华源（解决网络问题）
RUN sed -i 's@//.*archive.ubuntu.com@//mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list && \
    sed -i 's@//.*security.ubuntu.com@//mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list

# 4. 安装系统依赖
RUN apt update && apt install -y libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# 4. 安装Python依赖（换清华源，加速下载）
RUN pip install --no-cache-dir \
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