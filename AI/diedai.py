# x = 1.0              # 初值，可改为0~1内任意值
# tol = 1e-8
# for k in range(1, 1000):
#     x_new = (6 - x**3 - 0.5*exp(x)) / 5
#     if abs(x_new - x) < tol:
#         break
#     x = x_new
# x 即为近似解
import math

def iteration_method(x0, tol=1e-8, max_iter=1000):
    def phi(x):
        return (6 - x**3 - 0.5 * math.exp(x)) / 5

    print("迭代计算开始：")
    print(f"{'迭代次数':<8}{'x_k':<20}{'x_{k+1}':<20}{'误差':<20}")
    print("-" * 70)

    for k in range(max_iter):
        x1 = phi(x0)
        error = abs(x1 - x0)
        print(f"{k:<8}{x0:<20.10f}{x1:<20.10f}{error:<20.10e}")
        if error < tol:
            print("-" * 70)
            print(f"运行结果显示：从 x0 = {x0:.6f} 迭代开始，收敛于 x* = {x1:.10f}")
            print(f"共迭代 {k+1} 次")
            return x1
        x0 = x1

    print("未在最大迭代次数内收敛。")
    return None

# === 主程序 ===
x0 = 1.0
iteration_method(x0)
