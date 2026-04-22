import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import networkx as nx
from scipy.linalg import solve_discrete_are

matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False

# ============================================================
# 0. 全局随机种子
# ============================================================
SEED = 42
np.random.seed(SEED)

# ============================================================
# 1. 系统参数设置
# ============================================================
n = 8       # 状态维度
m = 4       # 控制维度
N = 4       # 智能体数量
T = 500     # 迭代次数
tau_max = 3 # 最大通信延迟
eta_0 = 0.02  # 初始学习率（保守设置）
eta_min = 1e-4
mu = 0.001  # 零阶平滑参数（更小，减少估计偏差）

# 生成稳定系统矩阵（谱半径严格控制在0.7以内，留出足够稳定裕度）
np.random.seed(SEED)
A_raw = np.random.randn(n, n)
eigvals = np.linalg.eigvals(A_raw)
spectral_radius = np.max(np.abs(eigvals))
A = A_raw / spectral_radius * 0.7  # 谱半径=0.7，系统充分稳定

np.random.seed(SEED + 10)
B = np.random.randn(n, m) * 0.5

Q = np.eye(n)
R = np.eye(m)

print("=" * 60)
print("系统参数")
print("=" * 60)
print(f"状态维度 n={n}, 控制维度 m={m}, 智能体数 N={N}")
print(f"A 谱半径 = {np.max(np.abs(np.linalg.eigvals(A))):.4f}")
print(f"迭代次数 T={T}, tau_max={tau_max}")
print(f"初始学习率 eta_0={eta_0}, 平滑参数 mu={mu}")

# ============================================================
# 2. 集中式 DARE 求解最优解
# ============================================================
P_opt = solve_discrete_are(A, B, Q, R)
K_opt = np.linalg.inv(R + B.T @ P_opt @ B) @ B.T @ P_opt @ A
cost_opt = np.trace(P_opt)

print("\n" + "=" * 60)
print("集中式 DARE 最优解")
print("=" * 60)
print(f"最优 LQR 代价 C* = {cost_opt:.4f}")
print(f"K* 范数 ||K*||_F = {np.linalg.norm(K_opt, 'fro'):.4f}")

# ============================================================
# 3. LQR 代价函数（Lyapunov 迭代）
# ============================================================
def compute_lqr_cost(K, A, B, Q, R, max_iter=5000):
    A_cl = A - B @ K
    rho = np.max(np.abs(np.linalg.eigvals(A_cl)))
    if rho >= 1.0:
        return np.inf
    P = np.eye(n)
    Q_K = Q + K.T @ R @ K
    for _ in range(max_iter):
        P_new = Q_K + A_cl.T @ P @ A_cl
        if np.linalg.norm(P_new - P, 'fro') < 1e-12:
            break
        P = P_new
    return np.trace(P_new)

# ============================================================
# 4. 初始控制增益：小扰动 + 严格稳定性验证
# ============================================================
# 策略：在 K* 附近加小扰动，确保：
#   (1) 初始闭环谱半径 < 0.95
#   (2) 初始代价与 K* 有可见差距（热力图有色差）
#   (3) 不会因初始不稳定导致代价爆炸

np.random.seed(SEED + 99)
best_K_init = None
best_noise  = None
target_rho  = 0.93  # 目标初始谱半径上限

for noise_scale in np.arange(0.3, 0.05, -0.02):
    np.random.seed(SEED + 99)
    noise = np.random.randn(m, n) * noise_scale
    K_try = K_opt + noise
    A_cl_try = A - B @ K_try
    rho_try = np.max(np.abs(np.linalg.eigvals(A_cl_try)))
    if rho_try < target_rho:
        best_K_init = K_try.copy()
        best_noise  = noise_scale
        break

if best_K_init is None:
    # 兜底：直接用 K* 加极小扰动
    np.random.seed(SEED + 99)
    best_K_init = K_opt + np.random.randn(m, n) * 0.05
    best_noise  = 0.05

K_init = best_K_init.copy()
cost_init = compute_lqr_cost(K_init, A, B, Q, R)

print("\n" + "=" * 60)
print("初始化信息")
print("=" * 60)
print(f"扰动幅度 noise_scale      = {best_noise:.4f}")
print(f"初始 LQR 代价 C(K_0)     = {cost_init:.4f}")
print(f"||K_init - K*||_F        = {np.linalg.norm(K_init - K_opt, 'fro'):.4f}")
print(f"初始闭环谱半径            = {np.max(np.abs(np.linalg.eigvals(A - B @ K_init))):.4f}")

# ============================================================
# 5. 智能体分块（路径图）
# ============================================================
rows_per_agent = m // N  # 每个智能体负责1行
agent_rows = [list(range(i * rows_per_agent, (i + 1) * rows_per_agent))
              for i in range(N)]

neighbors = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2]}

print("\n" + "=" * 60)
print("智能体分块信息")
print("=" * 60)
for i in range(N):
    print(f"Agent {i+1}: 控制行 {agent_rows[i]}, "
          f"邻居: {[j+1 for j in neighbors[i]]}")

# ============================================================
# 6. 余弦退火学习率
# ============================================================
def cosine_lr(t, T, eta_0, eta_min=1e-4):
    return eta_min + 0.5 * (eta_0 - eta_min) * (1 + np.cos(np.pi * t / T))

# ============================================================
# 7. 异步 ZO-BCD 主循环（带投影回退机制）
# ============================================================
K = K_init.copy()
cost_history = [cost_init]

# 延迟缓冲区
delay_buffer = {i: [K.copy()] for i in range(N)}

np.random.seed(SEED + 200)

print("\n" + "=" * 60)
print("开始迭代训练...")
print("=" * 60)

for t in range(1, T + 1):
    eta_t = cosine_lr(t, T, eta_0, eta_min)

    for i in range(N):
        rows_i = agent_rows[i]

        # 异步延迟：随机选取缓冲区中的历史增益
        delay = np.random.randint(0, tau_max + 1)
        buf = delay_buffer[i]
        idx = max(0, len(buf) - 1 - delay)
        K_used = buf[idx].copy()

        # 零阶两点梯度估计
        U = np.zeros_like(K)
        U[rows_i, :] = np.random.randn(len(rows_i), n)
        U_norm = U / (np.linalg.norm(U, 'fro') + 1e-12)

        K_plus  = K_used + mu * U_norm
        K_minus = K_used - mu * U_norm

        c_plus  = compute_lqr_cost(K_plus,  A, B, Q, R)
        c_minus = compute_lqr_cost(K_minus, A, B, Q, R)

        if np.isinf(c_plus) or np.isinf(c_minus):
            continue

        # 梯度估计（仅更新本智能体负责的行）
        grad_scalar = (c_plus - c_minus) / (2 * mu)
        grad_est = np.zeros_like(K)
        grad_est[rows_i, :] = grad_scalar * U_norm[rows_i, :]

        # 自适应梯度裁剪
        grad_norm = np.linalg.norm(grad_est, 'fro')
        clip_thresh = 2.0
        if grad_norm > clip_thresh:
            grad_est = grad_est * (clip_thresh / grad_norm)

        # 候选更新
        K_new = K.copy()
        K_new[rows_i, :] -= eta_t * grad_est[rows_i, :]

        # 稳定性投影回退：若更新后不稳定则缩小步长重试
        accepted = False
        step = eta_t
        for _ in range(10):
            K_try = K.copy()
            K_try[rows_i, :] = K[rows_i, :] - step * grad_est[rows_i, :]
            rho_try = np.max(np.abs(np.linalg.eigvals(A - B @ K_try)))
            if rho_try < 0.9999:
                K = K_try
                accepted = True
                break
            step *= 0.5

        if accepted:
            delay_buffer[i].append(K.copy())
            if len(delay_buffer[i]) > tau_max + 2:
                delay_buffer[i].pop(0)

    # 记录本轮代价
    c_t = compute_lqr_cost(K, A, B, Q, R)
    cost_history.append(c_t if not np.isinf(c_t) else cost_history[-1])

    if t % 100 == 0:
        print(f"  迭代 t={t:4d} | C(K)={cost_history[-1]:.4f} | "
              f"C(K)-C*={cost_history[-1]-cost_opt:.4f} | "
              f"lr={eta_t:.2e}")

K_final = K.copy()

# ============================================================
# 8. 精确数值汇总输出
# ============================================================
cost_final = cost_history[-1]
abs_gap    = cost_final - cost_opt
rel_gap    = abs_gap / cost_opt * 100
total_drop_pct = (cost_history[0] - cost_final) / cost_history[0] * 100

print("\n" + "=" * 60)
print("训练结果汇总")
print("=" * 60)
print(f"初始代价   C(K_0)       = {cost_history[0]:.4f}")
print(f"t=100 代价 C(K_100)     = {cost_history[100]:.4f}")
print(f"最终代价   C(K_final)   = {cost_final:.4f}")
print(f"最优基准   C*           = {cost_opt:.4f}")
print(f"绝对差距   C(K)-C*      = {abs_gap:.4f}")
print(f"相对差距                = {rel_gap:.2f}%")
print(f"代价总下降幅度           = {total_drop_pct:.2f}%")
print(f"右图初始差距             = {cost_history[0] - cost_opt:.4e}")
print(f"右图最终差距             = {cost_final - cost_opt:.4e}")
print("-" * 60)
print(f"||K_final - K_init||_F = {np.linalg.norm(K_final - K_init, 'fro'):.4f}")
print(f"||K_final - K*||_F     = {np.linalg.norm(K_final - K_opt,  'fro'):.4f}")
print(f"||K_init  - K*||_F     = {np.linalg.norm(K_init  - K_opt,  'fro'):.4f}")
print(f"K_final 闭环谱半径      = {np.max(np.abs(np.linalg.eigvals(A - B @ K_final))):.4f}")

# ============================================================
# 9. 图1：通信拓扑图
# ============================================================
fig1, ax1 = plt.subplots(figsize=(5, 4))
G = nx.path_graph(N)
# 固定布局：水平排列，视觉清晰
pos = {i: (i, 0) for i in range(N)}
nx.draw_networkx_nodes(G, pos, node_size=1200,
                       node_color='steelblue', alpha=0.9, ax=ax1)
nx.draw_networkx_labels(G, pos,
                        labels={i: f'Agent {i+1}' for i in range(N)},
                        font_color='white', font_size=10, ax=ax1)
nx.draw_networkx_edges(G, pos, width=2.5,
                       edge_color='gray', ax=ax1)
ax1.set_title('多智能体通信拓扑图（路径图）', fontsize=12, pad=15)
ax1.axis('off')
ax1.set_xlim(-0.5, N - 0.5)
ax1.set_ylim(-0.5, 0.5)
plt.tight_layout()
plt.savefig('topology.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n[图1] topology.png 已保存")

# ============================================================
# 10. 图2：LQR 代价收敛曲线
# ============================================================
fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
fig2.suptitle('Async ZO-BCD 收敛性分析\n(IEEE TAC 2024, Jing et al.)',
              fontsize=13)

iters = np.arange(len(cost_history))

# 左图：线性坐标
ax_l = axes2[0]
ax_l.plot(iters, cost_history,
          color='steelblue', linewidth=1.8,
          marker='o', markersize=3, markevery=25,
          label='Async ZO-BCD（本文算法）')
ax_l.axhline(y=cost_opt, color='red', linestyle='--',
             linewidth=1.5, label=f'最优LQR基准 C*={cost_opt:.4f}')
ax_l.set_xlabel('迭代次数 t', fontsize=11)
ax_l.set_ylabel('LQR代价 C(K)', fontsize=11)
ax_l.set_title('LQR代价收敛曲线（线性坐标）', fontsize=11)
ax_l.legend(fontsize=9)
ax_l.grid(True, alpha=0.3)

# 右图：对数坐标
gap_history = [max(c - cost_opt, 1e-10) for c in cost_history]
ax_r = axes2[1]
ax_r.semilogy(iters, gap_history,
              color='green', linewidth=1.8,
              marker='s', markersize=3, markevery=25,
              label='C(K) - C*')
ax_r.set_xlabel('迭代次数 t', fontsize=11)
ax_r.set_ylabel('代价差距 C(K) - C*（对数坐标）', fontsize=11)
ax_r.set_title('收敛速率分析（对数坐标）', fontsize=11)
ax_r.legend(fontsize=9)
ax_r.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('convergence.png', dpi=150, bbox_inches='tight')
plt.show()
print("[图2] convergence.png 已保存")

# ============================================================
# 11. 图3：控制增益矩阵热力图
# ============================================================
fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))
fig3.suptitle('控制增益矩阵可视化', fontsize=13)

# 统一色彩范围（基于三矩阵最大绝对值）
vmax = max(np.abs(K_opt).max(),
           np.abs(K_init).max(),
           np.abs(K_final).max())

matrices = [K_init, K_final, K_opt]
titles   = ['初始增益 $K_{init}$',
            '训练后增益 $K_{final}$',
            '最优增益 $K^*$（DARE基准）']

for ax, mat, title in zip(axes3, matrices, titles):
    im = ax.imshow(mat, cmap='RdBu_r',
                   vmin=-vmax, vmax=vmax, aspect='auto')
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('状态维度', fontsize=10)
    ax.set_ylabel('控制维度', fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig('gain_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("[图3] gain_matrix.png 已保存")

print("\n全部完成！三张图片已保存至当前目录。")
