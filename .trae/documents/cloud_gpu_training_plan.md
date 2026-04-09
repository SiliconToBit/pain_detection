# 云端GPU服务器训练模型实施计划

## 项目现状分析

当前项目已具备：
- 训练脚本：`code/train.py`（PyTorch实现）
- 模型定义：`code/model.py`（HRSingleModalModel）
- 数据集类：`code/datebase.py`（PainHRDataset）
- Dockerfile：已配置CUDA 13.0环境
- 运行脚本：`run_train.sh`（支持环境变量配置）

## 云端GPU训练方案

### 方案一：AutoDL（推荐 - 国内用户）

**优点**：价格便宜、国内访问快、按小时计费、预装深度学习环境

#### 步骤：

1. **租用实例**
   - 访问 [AutoDL](https://www.autodl.com/)
   - 选择镜像：PyTorch 2.0+ / Python 3.10+
   - GPU推荐：RTX 3090 / RTX 4090（性价比高）
   - 数据盘勾选（避免系统盘数据丢失）

2. **上传代码和数据**
   ```bash
   # 方式1：使用AutoDL的JupyterLab上传功能
   # 方式2：使用SCP/SFTP上传
   scp -r /home/gm/Workspace/ai-projects/pain_detection root@<服务器IP>:/root/
   ```

3. **安装依赖**
   ```bash
   cd /root/pain_detection
   pip install pandas scikit-learn
   ```

4. **运行训练**
   ```bash
   # 基础训练
   python code/train.py
   
   # 自定义参数训练
   NUM_EPOCHS=200 BATCH_SIZE=32 LEARNING_RATE=1e-4 python code/train.py
   ```

5. **下载模型**
   ```bash
   scp -r root@<服务器IP>:/root/pain_detection/checkpoints_hr ./
   ```

---

### 方案二：阿里云PAI-DSW

**优点**：企业级稳定、与阿里云生态集成

#### 步骤：

1. **创建DSW实例**
   - 进入阿里云PAI控制台
   - 创建DSW实例，选择GPU规格
   - 选择PyTorch官方镜像

2. **上传代码**（同方案一）

3. **训练执行**（同方案一）

---

### 方案三：使用Docker容器（通用方案）

适用于任何支持Docker的云服务器。

#### 步骤：

1. **构建镜像**
   ```bash
   cd /path/to/pain_detection
   docker build -t pain_detection:latest .
   ```

2. **运行容器**
   ```bash
   # 使用run_train.sh脚本
   NUM_EPOCHS=100 BATCH_SIZE=16 ./run_train.sh
   
   # 或直接运行
   docker run --gpus all \
     -v $(pwd):/pain_detection \
     -w /pain_detection \
     -e NUM_EPOCHS=100 \
     -e BATCH_SIZE=16 \
     pain_detection:latest \
     python3 code/train.py
   ```

---

### 方案四：Google Colab / Kaggle（免费方案）

**优点**：免费GPU、适合实验

#### 步骤：

1. **上传项目到Google Drive或Kaggle Dataset**

2. **创建Notebook**
   ```python
   # 挂载Google Drive
   from google.colab import drive
   drive.mount('/content/drive')
   
   # 进入项目目录
   %cd /content/drive/MyDrive/pain_detection
   
   # 安装依赖
   !pip install pandas scikit-learn
   
   # 运行训练
   !python code/train.py
   ```

---

## 实施步骤清单

### 阶段一：准备工作

- [ ] 1.1 整理项目依赖列表（创建requirements.txt）
- [ ] 1.2 确认数据文件完整性（data/hr/train_hr.csv, val_hr.csv）
- [ ] 1.3 测试本地训练脚本能否正常运行

### 阶段二：云服务器配置

- [ ] 2.1 选择云服务商并租用GPU实例
- [ ] 2.2 配置SSH连接（如需要）
- [ ] 2.3 上传项目代码和数据

### 阶段三：环境搭建

- [ ] 3.1 安装Python依赖
- [ ] 3.2 验证CUDA可用性（`python -c "import torch; print(torch.cuda.is_available())"`）
- [ ] 3.3 测试数据加载

### 阶段四：训练执行

- [ ] 4.1 启动训练（建议使用nohup或tmux后台运行）
- [ ] 4.2 监控训练进度
- [ ] 4.3 保存训练日志

### 阶段五：结果获取

- [ ] 5.1 下载训练好的模型文件
- [ ] 5.2 下载训练日志
- [ ] 5.3 本地验证模型效果

---

## 推荐配置参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| GPU | RTX 3090/4090 | 性价比高 |
| NUM_EPOCHS | 100-200 | 根据early stop调整 |
| BATCH_SIZE | 16-32 | 根据GPU显存调整 |
| LEARNING_RATE | 5e-4 | 默认值 |
| MODEL_ARCH | causal_gru | 实时架构 |
| FEATURE_MODE | enhanced | 增强特征 |

---

## 注意事项

1. **数据安全**：上传数据前确认无敏感信息
2. **成本控制**：训练完成后及时释放实例
3. **模型保存**：定期保存checkpoint，避免意外中断
4. **日志记录**：使用`tee`命令同时输出到终端和文件
   ```bash
   python code/train.py 2>&1 | tee train.log
   ```

---

## 快速启动命令

### AutoDL/普通云服务器
```bash
# 安装依赖
pip install torch torchvision pandas scikit-learn

# 后台训练
nohup python code/train.py > train.log 2>&1 &

# 查看日志
tail -f train.log
```

### 使用Docker
```bash
# 构建镜像
docker build -t pain_detection:latest .

# 训练
NUM_EPOCHS=100 ./run_train.sh
```