import pandas as pd
import numpy as np

# --- 数据准备 ---
# 创建作业中的数据集
data = {
    '编号': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    '年龄': [25, 35, 45, 20, 35, 52, 23, 40, 60, 48],
    '收入水平': ['中等', '高', '高', '低', '高', '中等', '低', '高', '中等', '高'],
    '信贷情况': ['一般', '好', '好', '一般', '一般', '好', '一般', '好', '一般', '一般'],
    '是否购买电脑': ['否', '否', '是', '否', '是', '是', '否', '是', '是', '是']
}
df = pd.DataFrame(data)
df.set_index('编号', inplace=True)

# 目标变量
target = '是否购买电脑'

# --- 核心函数 (根据PDF中的公式) ---

def calculate_gini(series):
    """
    计算一个数据集（Series）的基尼指数
    Gini(D) = 1 - sum(pk^2) 
    """
    if series.empty:
        return 0
    
    # 统计各类别的数量
    counts = series.value_counts()
    total = len(series)
    gini = 1.0
    
    # 遍历每个类别，计算 Gini
    for count in counts:
        p = count / total
        gini -= p**2
    return gini

def get_weighted_gini(series, left_mask, right_mask):
    """
    计算分裂后的加权基尼指数
    Gini_split = (|A1|/|A|) * Gini(A1) + (|A2|/|A|) * Gini(A2) 
    """
    total_n = len(series)
    
    # 根据掩码（mask）获取左右两个子集
    left_series = series[left_mask]
    right_series = series[right_mask]
    
    n_left = len(left_series)
    n_right = len(right_series)
    
    # 如果分裂导致一个子集为空，则这是一个无效分裂
    if n_left == 0 or n_right == 0:
        return float('inf') # 返回无穷大，确保不会被选为最佳
        
    # 计算加权 Gini
    gini_left = calculate_gini(left_series)
    gini_right = calculate_gini(right_series)
    
    weighted_gini = (n_left / total_n) * gini_left + (n_right / total_n) * gini_right
    return weighted_gini

# --- 作业任务 1 & 2: 处理连续特征“年龄” ---
print("--- 任务 1 & 2: 计算特征“年龄”的最佳分裂点和基尼指数 ---")

# 1. 对“年龄”进行排序
sorted_df = df.sort_values(by='年龄')

# 2. 找到唯一的年龄值
unique_ages = sorted_df['年龄'].unique()

best_gini = float('inf')
best_split_point = None

# 3. 计算潜在分裂点（相邻不同值的平均值）
# 这是处理连续值的标准方法，与PDF中尝试不同阈值的思想一致 [cite: 35]
potential_splits = []
for i in range(len(unique_ages) - 1):
    split_point = (unique_ages[i] + unique_ages[i+1]) / 2
    potential_splits.append(split_point)

print(f"“年龄”的潜在分裂点: {potential_splits}\n")

# 4. 遍历所有分裂点，计算Gini
for split in potential_splits:
    # CART是二叉树，所以分裂为 <= 和 >
    left_mask = df['年龄'] <= split
    right_mask = df['年龄'] > split
    
    current_gini = get_weighted_gini(df[target], left_mask, right_mask)
    
    print(f"  测试分裂点 {split}: Gini = {current_gini:.4f}")
    
    if current_gini < best_gini:
        best_gini = current_gini
        best_split_point = split

print("\n--- 任务 1 结果 ---")
print(f"对于特征“年龄”，确定的最佳划分点是: {best_split_point}")
print("--- 任务 2 结果 ---")
print(f"在该划分下的基尼指数 (Gini_split) 是: {best_gini:.4f}\n")


# --- 作业任务 3: 构建CART决策树的根节点分裂 ---
print("--- 任务 3: 确定根节点分裂 ---")
print("要确定根节点，我们必须比较所有特征的最佳Gini_split...")

# 我们已经有了“年龄”的最佳Gini
results = {
    '年龄': {'best_split': best_split_point, 'best_gini': best_gini}
}

# 1. 计算离散特征 "收入水平"
# CART对离散特征也进行二元分裂，需要测试所有可能的子集组合
# {低} vs {中等, 高}, {中等} vs {低, 高}, {高} vs {低, 中等}
feature = '收入水平'
best_discrete_gini = float('inf')
best_discrete_split = None

# 分区 1: {低} vs {中等, 高}
split_val = ['低']
left_mask = df[feature].isin(split_val)
right_mask = ~left_mask
gini_1 = get_weighted_gini(df[target], left_mask, right_mask)
print(f"  测试 '收入水平' 分裂 {{'低'}} vs {{'中等', '高'}}: Gini = {gini_1:.4f}")
if gini_1 < best_discrete_gini:
    best_discrete_gini = gini_1
    best_discrete_split = f"{split_val} vs others"

# 分区 2: {中等} vs {低, 高}
split_val = ['中等']
left_mask = df[feature].isin(split_val)
right_mask = ~left_mask
gini_2 = get_weighted_gini(df[target], left_mask, right_mask)
print(f"  测试 '收入水平' 分裂 {{'中等'}} vs {{'低', '高'}}: Gini = {gini_2:.4f}")
if gini_2 < best_discrete_gini:
    best_discrete_gini = gini_2
    best_discrete_split = f"{split_val} vs others"

# 分区 3: {高} vs {低, 中等}
split_val = ['高']
left_mask = df[feature].isin(split_val)
right_mask = ~left_mask
gini_3 = get_weighted_gini(df[target], left_mask, right_mask)
print(f"  测试 '收入水平' 分裂 {{'高'}} vs {{'低', '中等'}}: Gini = {gini_3:.4f}")
if gini_3 < best_discrete_gini:
    best_discrete_gini = gini_3
    best_discrete_split = f"{split_val} vs others"

results[feature] = {'best_split': best_discrete_split, 'best_gini': best_discrete_gini}
print(f"  '收入水平' 的最佳Gini_split: {best_discrete_gini:.4f}\n")


# 2. 计算离散特征 "信贷情况"
# 只有两个值 {一般} vs {好}，因此只有一种分裂
feature = '信贷情况'
split_val = ['一般']
left_mask = df[feature].isin(split_val)
right_mask = ~left_mask
gini_credit = get_weighted_gini(df[target], left_mask, right_mask)
print(f"  测试 '信贷情况' 分裂 {{'一般'}} vs {{'好'}}: Gini = {gini_credit:.4f}\n")
results[feature] = {'best_split': "{'一般'} vs {'好'}", 'best_gini': gini_credit}


# 3. 比较所有特征，确定根节点
root_feature = None
root_gini = float('inf')
root_split_condition = None

for feature, res in results.items():
    print(f"  特征 '{feature}' 的最小 Gini_split = {res['best_gini']:.4f}")
    if res['best_gini'] < root_gini:
        root_gini = res['best_gini']
        root_feature = feature
        root_split_condition = res['best_split']

print("\n--- 任务 3 结果 ---")
print(f"比较所有特征，'{root_feature}' 提供了最低的基尼指数 ({root_gini:.4f})。")
print(f"因此，CART决策树的根节点分裂是基于特征 '{root_feature}'，分裂条件为: '年龄 <= {root_split_condition}'")