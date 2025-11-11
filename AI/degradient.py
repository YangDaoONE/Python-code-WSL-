import math

# f, f'，以及目标 g(x)=0.5 f(x)^2（便于监控）
def f(x):  return x**3 + 0.5*math.exp(x) + 5*x - 6
def df(x): return 3*x**2 + 0.5*math.exp(x) + 5
def g(x):  return 0.5 * f(x)**2

def gradient_descent_root(x0, tol=1e-8, max_iter=1000, 
                          alpha0=1.0, beta=0.5, c=1e-4):
    """
    用梯度下降法最小化 g(x)=0.5*f(x)^2，从而求 f(x)=0 的根。
    - 回溯线搜索: 选择步长 eta 使得 Armijo 条件 g(x-eta*g') <= g(x)-c*eta*||g'||^2
    - 停止条件: |f(x)| < tol 或 |x_{k+1}-x_k| < tol
    """
    print("梯度下降法求根：")
    print(f"{'迭代次数':<8}{'x_k':<20}{'f(x_k)':<20}{'g(x_k)':<20}{'eta':<12}")
    print("-"*88)

    x = x0
    for k in range(max_iter):
        fx  = f(x)
        dfx = df(x)
        grad = fx * dfx                # g'(x)
        gx = 0.5 * fx*fx

        # Armijo 回溯线搜索
        eta = alpha0
        while True:
            x_new = x - eta * grad
            if g(x_new) <= gx - c * eta * (grad**2):
                break
            eta *= beta                 # 缩小步长

        print(f"{k:<8}{x:<20.10f}{fx:<20.10f}{gx:<20.10e}{eta:<12.3e}")

        # 更新
        if abs(fx) < tol:
            print("-"*88)
            print(f"运行结果显示：从 x0 = {x0:.6f} 迭代开始，稳定收敛于 x* = {x:.10f}")
            print(f"满足 |f(x*)| < {tol:g}，共迭代 {k} 次。")
            return x

        x_next = x - eta * grad
        if abs(x_next - x) < tol:
            x = x_next
            print("-"*88)
            print(f"运行结果显示：从 x0 = {x0:.6f} 迭代开始，稳定收敛于 x* = {x:.10f}")
            print(f"共迭代 {k+1} 次。")
            return x

        x = x_next

    print("\n 未在最大迭代次数内达到容限。最后近似解：", x)
    return x

# ==== 主程序 ====
x0 = 1.0
gradient_descent_root(x0)
