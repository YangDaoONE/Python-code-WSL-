import numpy as np

# ========== 1️⃣ 路径点建模 ==========
path_points = np.array([
    [0.4, 4.6],
    [1.0, 4.2],
    [1.5, 3.5],
    [1.8, 2.8],
    [2.5, 2.2],
    [3.3, 2.0],
    [4.0, 2.8],
    [4.2, 3.6],
    [3.8, 4.4],
    [0.4, 4.6],
])

# 计算路径累计长度
cumulative_lengths = [0]
for i in range(1, len(path_points)):
    seg_len = np.linalg.norm(path_points[i] - path_points[i - 1])
    cumulative_lengths.append(cumulative_lengths[-1] + seg_len)
cumulative_lengths = np.array(cumulative_lengths)
total_length = cumulative_lengths[-1]
print(f"原始路径总长度: {total_length:.2f} km")

# ⭐️ 校正路径总长到18 km ⭐️
SCALE_FACTOR = 18 / total_length
path_points *= SCALE_FACTOR

# 重新计算累计长度（缩放后）
cumulative_lengths = [0]
for i in range(1, len(path_points)):
    seg_len = np.linalg.norm(path_points[i] - path_points[i - 1])
    cumulative_lengths.append(cumulative_lengths[-1] + seg_len)
cumulative_lengths = np.array(cumulative_lengths)
total_length = cumulative_lengths[-1]
print(f"路径总长度校正后: {total_length:.2f} km")

# ========== 2️⃣ 映射函数 ==========
def map_s_to_xy(s_array, path_points, cumulative_lengths):
    xy_positions = []
    for s in s_array:
        s = s % total_length
        idx = np.searchsorted(cumulative_lengths, s, side='right') - 1
        seg_start = path_points[idx]
        seg_end = path_points[idx + 1]
        seg_len = cumulative_lengths[idx + 1] - cumulative_lengths[idx]
        ratio = (s - cumulative_lengths[idx]) / seg_len if seg_len > 0 else 0
        xy = seg_start + ratio * (seg_end - seg_start)
        xy_positions.append(xy)
    return np.array(xy_positions)

# ========== 3️⃣ PSO 参数 ==========
NUM_PARTICLES = 30
NUM_ITERATIONS = 100
NUM_POSTS = 36
VELOCITY_MAX = 0.3  # km

w = 0.7
c1 = 1.5
c2 = 1.5

# 应急响应参数
sampling_step = 0.01  # 每 10 米采样
sampling_s = np.arange(0, total_length, sampling_step)
sampling_points = map_s_to_xy(sampling_s, path_points, cumulative_lengths)

# 初始化粒子群
particles = [np.sort(np.random.uniform(0, total_length, NUM_POSTS)) for _ in range(NUM_PARTICLES)]
velocities = [np.random.uniform(-VELOCITY_MAX, VELOCITY_MAX, NUM_POSTS) for _ in range(NUM_PARTICLES)]
p_best = particles.copy()
p_best_fitness = [np.inf] * NUM_PARTICLES
g_best = None
g_best_fitness = np.inf

# ========== 4️⃣ 适应度函数 ==========
def fitness_function(post_xy, sampling_points):
    distances = []
    for sp in sampling_points:
        min_dist = np.min(np.linalg.norm(post_xy - sp, axis=1))
        distances.append(min_dist)
    return max(distances)

# ========== 5️⃣ PSO 主循环 ==========
for iter in range(NUM_ITERATIONS):
    for i in range(NUM_PARTICLES):
        r1 = np.random.rand(NUM_POSTS)
        r2 = np.random.rand(NUM_POSTS)
        velocities[i] = (w * velocities[i] +
                         c1 * r1 * (p_best[i] - particles[i]) +
                         c2 * r2 * ((g_best if g_best is not None else particles[i]) - particles[i]))
        velocities[i] = np.clip(velocities[i], -VELOCITY_MAX, VELOCITY_MAX)
        particles[i] += velocities[i]
        particles[i] = np.mod(particles[i], total_length)
        particles[i] = np.sort(particles[i])

        post_xy = map_s_to_xy(particles[i], path_points, cumulative_lengths)
        fit = fitness_function(post_xy, sampling_points)

        if fit < p_best_fitness[i]:
            p_best[i] = particles[i].copy()
            p_best_fitness[i] = fit

    best_idx = np.argmin(p_best_fitness)
    if p_best_fitness[best_idx] < g_best_fitness:
        g_best = p_best[best_idx].copy()
        g_best_fitness = p_best_fitness[best_idx]

    if (iter + 1) % 10 == 0:
        print(f"迭代 {iter+1}/{NUM_ITERATIONS}, 当前最优适应度: {g_best_fitness:.3f} km")

# ========== 6️⃣ 输出最终结果 ==========
print("\n✅ 最优应急岗分布 (路径长度参数 s)：")
print(np.round(g_best, 3))

final_posts_xy = map_s_to_xy(g_best, path_points, cumulative_lengths)
print("\n✅ 对应岗位实际坐标 (X, Y in km)：")
for idx, (x, y) in enumerate(final_posts_xy):
    print(f"岗 {idx+1}: ({x:.3f}, {y:.3f}) km")

# 🚨 修正版响应时间输出（1.5 m/s = 0.09 km/min）
print(f"\n✅ 最终最大响应距离: {g_best_fitness:.3f} km (~{(g_best_fitness / 0.09):.2f} 分钟)")
