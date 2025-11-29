import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd

# 设置中文字体支持 (根据环境可能需要调整，这里使用无衬线字体作为回退)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class RankTunnelSim:
    """
    秩-风洞 (Rank Tunnel) 模拟器
    用于验证推理的热力学与重整化流理论
    """
    def __init__(self, dim=100, target_rank=20, model_capacity=1, mismatch_g=0.1):
        """
        初始化物理环境
        :param dim: 向量空间的维度 (Emb Dimension)
        :param target_rank: 问题的内禀秩 (Mass / Intrinsic Rank)
        :param model_capacity: 模型单步推理的秩限制 (流态介质的带宽)
        :param mismatch_g: 模型失配度/引力场 (g factor), 0=完美理解, 1=完全胡乱推理
        """
        self.dim = dim
        self.target_rank = target_rank
        self.model_capacity = model_capacity
        self.g = mismatch_g
        
        # 1. 生成高质量问题 (High Mass Object)
        # 这是一个随机生成的高秩矩阵 Target = U * S * V^T
        U_true, _ = np.linalg.qr(np.random.randn(dim, dim))
        V_true, _ = np.linalg.qr(np.random.randn(dim, dim))
        S_true = np.zeros(dim)
        S_true[:target_rank] = np.linspace(10, 1, target_rank) # 奇异值衰减
        self.Target = U_true @ np.diag(S_true) @ V_true.T
        
        # 2. 初始化状态
        self.Current_State = np.zeros((dim, dim))
        self.kv_cache_vectors = [] # 模拟流态介质 (Fluid Memory)
        self.residuals_history = []
        self.effective_ranks = []
        
    def effective_rank(self, matrix):
        """
        计算矩阵的有效秩 (Effective Rank / stable rank)
        基于奇异值的香农熵: exp(Entropy)
        """
        if matrix.shape[1] == 0: return 0
        _, s, _ = np.linalg.svd(matrix, full_matrices=False)
        # 归一化奇异值以作为概率分布
        p = s / np.sum(s) + 1e-12 # 避免log(0)
        entropy = -np.sum(p * np.log(p))
        return np.exp(entropy)

    def step(self):
        """
        执行一步推理 (One Time Step of Reasoning)
        机制：残差消除 + 噪声引入 (Renormalization Step)
        """
        # A. 计算当前残差 (Residual) - 尚未解决的问题部分
        R = self.Target - self.Current_State
        
        # B. 模型的有限关注 (Limited Attention / Capacity)
        # 模型只能捕捉残差中的主要成分 (Top-k SVD of Residual)
        U, S, Vt = np.linalg.svd(R, full_matrices=False)
        
        # 构造更新量 Delta (只取前 model_capacity 个分量)
        k = self.model_capacity
        
        # C. 引入模型失配 (The 'g' factor)
        # 完美的模型准确对齐 U 和 V，失配的模型会引入旋转误差
        noise_u = np.random.randn(*U[:, :k].shape) * self.g
        noise_v = np.random.randn(*Vt[:k, :].shape) * self.g
        
        U_noisy = U[:, :k] + noise_u
        Vt_noisy = Vt[:k, :] + noise_v
        
        # 重构更新步 Delta
        Delta = (U_noisy @ np.diag(S[:k]) @ Vt_noisy)
        
        # D. 更新状态 (Time Unfolding)
        self.Current_State += Delta
        
        # E. 记录流态介质 (存储这一步产生的特征向量，模拟 KV Cache)
        # 我们将 U_noisy 的列向量视为这一步产生的思维向量
        for i in range(k):
            self.kv_cache_vectors.append(U_noisy[:, i])
            
        # F. 计算指标
        self.residuals_history.append(np.linalg.norm(self.Target - self.Current_State))
        
        # 计算当前流态介质(KV Cache)的有效秩
        # 随着推理进行，如果模型“锁定”了规律，后续向量应与之前的线性相关，导致秩不再增长甚至下降（如果归一化）
        current_memory_matrix = np.column_stack(self.kv_cache_vectors)
        self.effective_ranks.append(self.effective_rank(current_memory_matrix))

    def run(self, steps=50):
        for _ in range(steps):
            self.step()

# --- 实验 1: 秩的呼吸 (Rank Breathing) & 锁定过程 ---
def plot_rank_breathing():
    print("Running Experiment 1: The 'Breathing' of Rank...")
    
    # 设定：一个中等难度的任务
    # g=0.05 保证能收敛
    sim = RankTunnelSim(dim=100, target_rank=15, model_capacity=1, mismatch_g=0.05)
    total_steps = 60
    sim.run(steps=total_steps)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # 绘制残差 (Thermodynamic Loss)
    color = 'tab:red'
    ax1.set_xlabel('Inference Time Steps (CoT Length)')
    ax1.set_ylabel('Thermodynamic Loss (Residual Norm)', color=color)
    ax1.plot(sim.residuals_history, color=color, linewidth=2, linestyle='--', label='Loss (Error)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    # 绘制有效秩 (Effective Rank)
    ax2 = ax1.twinx() 
    color = 'tab:blue'
    ax2.set_ylabel('Fluid Effective Rank (KV Cache Complexity)', color=color)
    ax2.plot(sim.effective_ranks, color=color, linewidth=3, label='Effective Rank')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # 标注相变区域
    # 当 Loss 降到初始值的 20% 以下，视为进入锁定区
    initial_loss = sim.residuals_history[0]
    lock_step = next((i for i, x in enumerate(sim.residuals_history) if x < initial_loss * 0.2), total_steps)
    
    if lock_step < total_steps:
        plt.axvline(x=lock_step, color='green', linestyle=':', alpha=0.8)
        plt.text(lock_step + 1, sim.effective_ranks[lock_step], 'Locking Phase\n(Solidification)', color='green', fontweight='bold')

    plt.title('Experiment A: The Rank Breathing (Phase Duality)\nSolid -> Fluid (Expansion) -> Solid (Crystallization)')
    fig.tight_layout()
    plt.savefig('rank_breathing.png')
    plt.close()

# --- 实验 2: 相变图谱 (Phase Diagram) ---
def plot_phase_diagram():
    print("Running Experiment 2: Generating Phase Diagram...")
    
    # 调整变量范围，增加分辨率
    mass_levels = np.arange(1, 40, 1)  # 问题难度 (Mass)
    time_steps = np.arange(1, 80, 1)   # 推理时间 (增加到80以展示完整相变)
    
    phase_map = np.zeros((len(mass_levels), len(time_steps)))
    
    for i, mass in enumerate(mass_levels):
        for j, t in enumerate(time_steps):
            # 统一参数：dim=100, g=0.05 (与实验1保持一致，确保可解)
            sim = RankTunnelSim(dim=100, target_rank=mass, model_capacity=1, mismatch_g=0.05)
            sim.run(steps=t)
            
            initial_norm = np.linalg.norm(sim.Target)
            final_norm = sim.residuals_history[-1]
            ratio = final_norm / initial_norm
            
            # 存储 Residual Ratio (0.0 = Solved, 1.0 = Unsolved)
            phase_map[i, j] = ratio

    plt.figure(figsize=(10, 8))
    
    # 修正 Colormap: 使用 'magma'
    # 在 'magma' 中: 0.0 是黑色 (Black), 1.0 是亮色 (Bright)
    # 这符合直觉: 黑色 = 冷却/固态/锁定 (Solved), 亮色 = 高热/流态/混沌 (Unsolved)
    extent = [time_steps[0], time_steps[-1], mass_levels[0], mass_levels[-1]]
    
    # 显示热力图
    im = plt.imshow(phase_map, aspect='auto', cmap="magma", vmin=0, vmax=1.0, 
               origin='lower', extent=extent)
    
    plt.colorbar(im, label='Residual Ratio (Black=Locked/Solid, Bright=Fluid/Chaos)')
    
    # 添加等高线，明确画出 "相变边界" (例如 10% 残差线)
    # 注意: contour 需要 X, Y 网格
    X, Y = np.meshgrid(time_steps, mass_levels)
    CS = plt.contour(X, Y, phase_map, levels=[0.2], colors='cyan', linestyles='dashed', linewidths=2)
    plt.clabel(CS, inline=1, fontsize=10, fmt='Locking Boundary')

    plt.xlabel('Inference Time (Compute/Heat)')
    plt.ylabel('Problem Mass (Intrinsic Rank)')
    plt.title('Phase Diagram: The Rank Tunnel\n(Black Region = Crystallized/Solved)')
    
    plt.savefig('phase_diagram.png')
    plt.close()

if __name__ == "__main__":
    np.random.seed(42)
    plot_rank_breathing()
    plot_phase_diagram()
    print("Simulation Complete. Images generated.")
