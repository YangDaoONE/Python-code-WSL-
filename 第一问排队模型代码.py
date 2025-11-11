import math
import matplotlib.pyplot as plt

def erlang_c(c, rho):
    """
    M/M/C排队模型：
    计算Erlang-C公式（排队概率P_wait）。
    参数:
    - c: 服务台数量 C
    - rho: 系统负载率 ρ = λ / (C * μ)
    """
    if rho >= 1:
        return 1.0  # 系统不稳定
    numer = (c * rho) ** c / (math.factorial(c) * (1 - rho))
    denom = sum((c * rho) ** k / math.factorial(k) for k in range(c)) + numer
    return numer / denom

def compute_waiting_time(c, mu, lambd):
    """
    M/M/C排队模型：
    计算平均等待时间 Wq（小时）。
    参数:
    - c: 服务台数量 C
    - mu: 单个服务速率 μ
    - lambd: 到达率 λ

    返回:
    - Wq: 平均等待时间 (小时)
    - p_wait: 排队概率
    - rho: 负载率
    """
    rho = lambd / (c * mu)
    if rho >= 1:
        return float('inf'), 1.0, rho  # 系统不稳定
    p_wait = erlang_c(c, rho)
    Wq = p_wait / (c * mu - lambd)
    return Wq, p_wait, rho

def compute_N_vs_Wq(c, mu, m, T1, max_Wq_hours, step=100):
    """
    对于给定的服务点（C, μ, m），计算 (N, Wq) 列表直到Wq超过限制。
    参数:
    - c: 服务台数量
    - mu: 单个服务速率
    - m: 游客平均使用率
    - T1: 开放时间
    - max_Wq_hours: 最大允许等待时间 (小时)
    - step: N递增步长
    """
    max_N = int(c * mu * T1 / m)
    Ns, Wqs = [], []
    N = step
    while N <= max_N:
        lambd = (m * N) / T1
        Wq, _, _ = compute_waiting_time(c, mu, lambd)
        Ns.append(N)
        Wqs.append(Wq)
        if Wq > max_Wq_hours:
            break
        N += step
    return Ns, Wqs

# 参数设置
services = {
    "休息区": {"C": 12, "mu": 100, "m": 0.5},
    "商服点": {"C": 6, "mu": 60, "m": 0.15},
    "医疗点": {"C": 3, "mu": 10, "m": 0.005}
}
T1 = 8.5
max_Wq = 0.25  # 等待时间限制 0.25 小时
step = 100

# 中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 计算并绘图
plt.figure(figsize=(10, 6))

for name, params in services.items():
    C, mu, m = params["C"], params["mu"], params["m"]
    Ns, Wqs = compute_N_vs_Wq(C, mu, m, T1, max_Wq, step=step)
    plt.plot(Ns, Wqs, marker='o', markersize=4, linewidth=1.2, label=f'{name}')  # 更小的点和细线
    

# 画等待时间限制线
plt.axhline(y=max_Wq, color='r', linestyle='--', linewidth=1.2, label=f'等待时间限制: {max_Wq} 小时')
plt.xlabel('游客总数 N')
plt.ylabel('平均等待时间 $W_q$ (小时)')
plt.title('不同服务类型的游客总数 N 与 平均等待时间 $W_q$ 的关系 (M/M/C排队模型)')
plt.legend(fontsize=10)
plt.grid(True)
plt.show()
