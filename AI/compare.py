import numpy as np
import time
n = 10000       # 特征维度
w = np.random.randn(n)   # 权重向量
x = np.random.randn(n)   # 输入向量
b = np.random.randn()    # 偏置
# 1. 非向量化实现
def non_vectorized(w, x, b):
    f = 0
    for j in range(len(w)):
        f += w[j] * x[j]
    return f + b

# 2. 向量化实现
def vectorized(w, x, b):
    return np.dot(w, x) + b
#非向量化计时
start = time.perf_counter()  # 开始计时
result_non = non_vectorized(w, x, b)
end = time.perf_counter()    # 结束计时
time_non = end - start       # 用时
print(f"非向量化结果: {result_non:.6f}, 用时: {time_non:.6f} 秒")
#向量化计时
start = time.perf_counter()
result_vec = vectorized(w, x, b)
end = time.perf_counter()
time_vec = end - start
print(f"向量化结果:   {result_vec:.6f}, 用时: {time_vec:.6f} 秒")
diff = abs(result_non - result_vec)
speedup = time_non / time_vec
print(f"结果差异: {diff:.3e}")
print(f"加速比:   {speedup:.2f}x （向量化比非向量化快 {speedup:.2f} 倍）")
