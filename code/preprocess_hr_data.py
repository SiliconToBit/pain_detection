# -*- coding: utf-8 -*-
"""
心率数据预处理脚本
将原始心率数据转换为模型训练所需的格式，并划分训练集和验证集
"""
import pandas as pd
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------------------------
# 配置参数
# -----------------------------------------------------------------------------
INPUT_CSV = "data/hr/pain_dataset_200P_4hz.csv"
TRAIN_OUTPUT = "data/hr/train_hr.csv"
VAL_OUTPUT = "data/hr/val_hr.csv"
SEQ_LEN = 30  # 心率序列长度（30个点，对应约7.5秒，采样率4Hz）
WINDOW_STRIDE = 1  # 滑窗步长（1 表示最大化样本数）
TEST_SIZE = 0.2  # 验证集比例
RANDOM_STATE = 42


def map_nrs_to_class(nrs_score):
    """将 pain_scale(1-8) 映射到 3 类: 0/1/2"""
    score = int(nrs_score)
    if 1 <= score <= 3:
        return 0
    if 4 <= score <= 6:
        return 1
    return 2


def build_subject_strata(df):
    """构建受试者级分层键，减小 train/val 的标签分布漂移。"""
    rows = []
    for person_id, person_data in df.groupby("person_ID", sort=False):
        mapped = person_data["pain_scale"].apply(map_nrs_to_class)
        class_counts = mapped.value_counts().reindex([0, 1, 2], fill_value=0)
        dominant_class = int(class_counts.idxmax())
        mean_pain = float(person_data["pain_scale"].mean())
        mean_bin = 0 if mean_pain < 4.0 else (1 if mean_pain < 6.0 else 2)

        rows.append(
            {
                "person_ID": person_id,
                "dominant_class": dominant_class,
                "mean_bin": mean_bin,
            }
        )

    subject_df = pd.DataFrame(rows)
    subject_df["stratum"] = (
        subject_df["dominant_class"].astype(str)
        + "_"
        + subject_df["mean_bin"].astype(str)
    )

    # 对样本过少（<2）的分层键降级到 dominant_class，避免 split 报错。
    key_counts = subject_df["stratum"].value_counts()
    rare_mask = subject_df["stratum"].map(key_counts) < 2
    subject_df.loc[rare_mask, "stratum"] = subject_df.loc[rare_mask, "dominant_class"].astype(str)
    return subject_df


def print_mapped_distribution(labels, prefix):
    mapped = pd.Series(labels).apply(map_nrs_to_class)
    dist = mapped.value_counts().reindex([0, 1, 2], fill_value=0)
    ratio = (dist / max(len(mapped), 1)).round(4)
    print(f"{prefix}映射后分布(0/1/2): {dist.to_dict()}")
    print(f"{prefix}映射后比例(0/1/2): {ratio.to_dict()}")

# -----------------------------------------------------------------------------
# 1. 加载原始数据
# -----------------------------------------------------------------------------
print("正在加载原始数据...")
df = pd.read_csv(INPUT_CSV)
print(f"原始数据形状: {df.shape}")
print(f"列名: {df.columns.tolist()}")
print(f"受试者数量: {df['person_ID'].nunique()}")
print(f"疼痛等级分布:\n{df['pain_scale'].value_counts().sort_index()}")

# -----------------------------------------------------------------------------
# 2. 按受试者划分训练集和验证集（避免数据泄露）
# -----------------------------------------------------------------------------
print("\n正在划分训练集和验证集...")
subject_df = build_subject_strata(df)

try:
    train_persons, val_persons = train_test_split(
        subject_df["person_ID"].values,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=subject_df["stratum"].values,
    )
except ValueError:
    # 极端情况下分层键仍可能不满足划分条件，回退到随机受试者切分。
    train_persons, val_persons = train_test_split(
        subject_df["person_ID"].values,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

print(f"训练集受试者: {len(train_persons)} 人")
print(f"验证集受试者: {len(val_persons)} 人")

# -----------------------------------------------------------------------------
# 3. 构建心率序列数据
# -----------------------------------------------------------------------------
def create_hr_sequences(df, person_ids, seq_len=30, stride=1):
    """
    为每个受试者构建心率序列
    
    参数:
        df: 原始数据DataFrame
        person_ids: 受试者ID列表
        seq_len: 序列长度
    
    返回:
        records: 序列样本列表（包含 person_ID、hr_sequence、nrs_label、受试者统计量）
    """
    records = []
    
    for person_id in person_ids:
        # 保持原始时间顺序，避免按 pain_scale 排序导致时序信息破坏。
        person_data = df[df["person_ID"] == person_id].reset_index(drop=True)

        hr_values = person_data["hr"].values
        pain_scales = person_data["pain_scale"].values
        person_hr_mean = float(hr_values.mean())
        person_hr_std = float(hr_values.std())
        if person_hr_std < 1e-6:
            person_hr_std = 1.0

        # 滑动窗口构建序列
        for i in range(0, len(hr_values) - seq_len + 1, stride):
            hr_seq = hr_values[i:i + seq_len]
            # 使用序列最后一个点的疼痛等级作为标签
            pain_label = pain_scales[i + seq_len - 1]

            records.append(
                {
                    "person_ID": person_id,
                    "hr_sequence": ",".join(map(str, hr_seq)),
                    "nrs_label": int(pain_label),
                    "person_hr_mean": person_hr_mean,
                    "person_hr_std": person_hr_std,
                }
            )

    return records

print("\n正在构建训练集序列...")
train_records = create_hr_sequences(
    df, train_persons, seq_len=SEQ_LEN, stride=WINDOW_STRIDE
)
print(f"训练集序列数量: {len(train_records)}")

print("\n正在构建验证集序列...")
val_records = create_hr_sequences(
    df, val_persons, seq_len=SEQ_LEN, stride=WINDOW_STRIDE
)
print(f"验证集序列数量: {len(val_records)}")
print_mapped_distribution([x["nrs_label"] for x in train_records], prefix="训练集")
print_mapped_distribution([x["nrs_label"] for x in val_records], prefix="验证集")

# -----------------------------------------------------------------------------
# 4. 保存为CSV文件
# -----------------------------------------------------------------------------
print("\n正在保存训练集...")
train_df = pd.DataFrame(train_records)
train_df.to_csv(TRAIN_OUTPUT, index=False)
print(f"训练集已保存到: {TRAIN_OUTPUT}")
print(f"训练集标签分布:\n{train_df['nrs_label'].value_counts().sort_index()}")

print("\n正在保存验证集...")
val_df = pd.DataFrame(val_records)
val_df.to_csv(VAL_OUTPUT, index=False)
print(f"验证集已保存到: {VAL_OUTPUT}")
print(f"验证集标签分布:\n{val_df['nrs_label'].value_counts().sort_index()}")

print("\n数据预处理完成！")
print(f"总计: 训练集 {len(train_df)} 条, 验证集 {len(val_df)} 条")