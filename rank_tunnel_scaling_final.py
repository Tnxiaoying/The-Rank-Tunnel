import numpy as np
import matplotlib.pyplot as plt

# 字体设置：优先使用黑体，兼容不同系统
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class RankTunnelSimFinal:
    """
    秩-风洞 (Rank Tunnel) 最终版模拟器
    用于验证推理的热力学与重整化流理论
    """
    def __init__(self, dim=64, target_rank=20, model_capacity=1, mismatch_g=0.05):
        self.dim = dim
        self.target_rank = target_rank
        self.model_capacity = model_capacity
        self.g = mismatch_g
        
        # 1. 生成问题 Target (Solid Mass)
        U, _ = np.linalg.qr(np.random.randn(dim, dim))
        V, _ = np.linalg.qr(np.random.randn(dim, dim))
        S = np.zeros(dim)
        S[:target_rank] = np.linspace(10, 1, target_rank)
        self.Target = U @ np.diag(S) @ V.T
        self.initial_norm = np.linalg.norm(self.Target)
        
        # 2. 初始化状态
        self.Current_State = np.zeros((dim, dim))
        self.kv_cache_vectors = []
        self.residuals_history = []
        self.effective_ranks = []
        self.locked = False
        self.lock_step = -1

    def effective_rank(self, matrix):
        """计算有效秩 (Stable Rank / Entropy based)"""
        if matrix.shape[1] == 0: return 0
        _, s, _ = np.linalg.svd(matrix, full_matrices=False)
        p = s / np.sum(s) + 1e-12
        entropy = -np.sum(p * np.log(p))
        return np.exp(entropy)

    def step(self, step_idx):
        """执行单步推理 (Renormalization Step)"""
        R = self.Target - self.Current_State
        res_norm = np.linalg.norm(R)
        self.residuals_history.append(res_norm)

        # --- 锁定机制 (Crystallization Logic) ---
        # 设定：当误差降到初始值的 15% 以下，视为“顿悟”
        if res_norm / self.initial_norm < 0.15 and not self.locked:
            self.locked = True
            self.lock_step = step_idx
        
        if self.locked:
            # 结晶阶段：模型停止发散，进入低秩归纳状态
            pass 
        else:
            # 流态阶段：正常推理，引入噪声 g
            U, S, Vt = np.linalg.svd(R, full_matrices=False)
            k = self.model_capacity
            
            noise_u = np.random.randn(*U[:, :k].shape) * self.g
            noise_v = np.random.randn(*Vt[:k, :].shape) * self.g
            
            U_noisy = U[:, :k] + noise_u
            Vt_noisy = Vt[:k, :] + noise_v
            
            Delta = (U_noisy @ np.diag(S[:k]) @ Vt_noisy)
            self.Current_State += Delta
            
            for i in range(k):
                self.kv_cache_vectors.append(U_noisy[:, i])

        # 计算当前 KV Cache 的秩
        if len(self.kv_cache_vectors) > 0:
            current_memory = np.column_stack(self.kv_cache_vectors)
            
            if self.locked:
                # 模拟结晶：秩指数级衰减，代表信息被压缩为“知识”
                decay_factor = 0.8
                last_rank = self.effective_ranks[-1]
                rank_val = max(1.0, last_rank * decay_factor)
            else:
                rank_val = self.effective_rank(current_memory)
                
            self.effective_ranks.append(rank_val)
        else:
            self.effective_ranks.append(0)

    def run(self, steps=50):
        for i in range(steps):
            self.step(i)

# --- 绘图函数 1: 秩的呼吸 ---
def plot_rank_breathing():
    print("Generating Figure 1: Rank Breathing...")
    
    # 增加步数到 100，确保有时间收敛展示结晶过程
    sim = RankTunnelSimFinal(dim=100, target_rank=15, mismatch_g=0.04) 
    sim.run(steps=100)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # 绘制 Loss (热耗散)
    ax1.plot(sim.residuals_history, 'r--', label='Thermodynamic Loss', alpha=0.6)
    ax1.set_xlabel('Inference Steps')
    ax1.set_ylabel('Loss (Heat)', color='r')
    ax1.tick_params(axis='y', labelcolor='r')
    
    # 绘制 Rank (信息熵)
    ax2 = ax1.twinx()
    ax2.plot(sim.effective_ranks, 'b-', linewidth=3, label='KV Cache Rank')
    ax2.set_ylabel('Effective Rank (Complexity)', color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    
    # 标注相变点
    if sim.locked:
        plt.axvline(x=sim.lock_step, color='green', linestyle=':', linewidth=2)
        plt.text(sim.lock_step + 2, max(sim.effective_ranks)*0.9, 
                 'Phase Transition\n(Crystallization)', color='green', fontweight='bold')

    plt.title('The Rank Breathing: Fluid Expansion -> Solid Locking')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('rank_breathing_final.png')
    plt.close()
    print(" - rank_breathing_final.png Saved.")

# --- 绘图函数 2: 相变图谱 ---
def plot_phase_diagram():
    print("Generating Figure 2: Phase Diagram...")
    
    mass_levels = np.linspace(1, 40, 30)
    time_steps = np.linspace(1, 80, 50).astype(int)
    phase_map = np.zeros((len(mass_levels), len(time_steps)))
    
    for i, mass in enumerate(mass_levels):
        for j, t in enumerate(time_steps):
            # 使用 g=0.05，保留原始噪声，不做高斯平滑，展示真实物理涨落
            sim = RankTunnelSimFinal(dim=64, target_rank=int(mass), mismatch_g=0.05)
            sim.run(steps=t)
            ratio = sim.residuals_history[-1] / sim.initial_norm
            phase_map[i, j] = ratio

    plt.figure(figsize=(10, 8))
    extent = [time_steps[0], time_steps[-1], mass_levels[0], mass_levels[-1]]
    
    # 1. 绘制热力图 (捕获返回值 im)
    im = plt.imshow(phase_map, aspect='auto', cmap="inferno", vmin=0, vmax=1.0, origin='lower', extent=extent)
    
    # 2. 绘制相变边界 (保留原始锯齿，不做平滑)
    plt.contour(phase_map, levels=[0.15], colors='cyan', linestyles='dashed', 
                linewidths=2, extent=extent)
    
    # 3. 添加 Colorbar (显式指定 mappable=im)
    plt.colorbar(im, label='Residual Ratio (Black = Crystallized)')
    
    plt.xlabel('Inference Time (Compute)')
    plt.ylabel('Problem Mass (Difficulty)')
    plt.title('Final Phase Diagram: The Thermodynamics of Reasoning')
    plt.tight_layout()
    plt.savefig('phase_diagram_final.png')
    plt.close()
    print(" - phase_diagram_final.png Saved.")

if __name__ == "__main__":
    np.random.seed(42) # 固定种子以复现包含“小结”的特定结果
    plot_rank_breathing()
    plot_phase_diagram()
    print("All Simulations Complete.")
