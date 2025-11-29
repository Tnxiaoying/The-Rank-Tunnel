import numpy as np
import matplotlib.pyplot as plt

# 字体设置：优先使用黑体，兼容不同系统
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class SingularityTrapSim:
    """
    奇异点陷阱模拟器 (The Singularity Trap)
    
    物理原理：
    该模拟器构建了一个高维 N-S 流体环境，其中：
    1. '有效秩' (Effective Rank) 充当流体的 '空间体积'。
    2. '注意力熵' (Viscosity) 充当流体的 '粘滞系数'。
    
    实验目标：
    验证当体积被非线性压缩 (Rank Collapse) 且 粘滞失效 (Viscosity Decay) 时，
    系统必然在有限时间内发生速度爆破 (Finite-Time Singularity)。
    """
    def __init__(self, dim=64, target_rank=30, initial_viscosity=1.0):
        self.dim = dim
        self.viscosity = initial_viscosity
        
        # 1. 构造“拓扑死结” (Topological Dead End)
        # 这是一个病态矩阵 (Ill-conditioned)，奇异值呈指数衰减。
        # 物理意义：这是一个宏观看起来简单，但微观细节极深的问题（如逻辑悖论）。
        # 模型会被诱导进入，然后发现无法用低秩解完美拟合，从而引发纠缠。
        U, _ = np.linalg.qr(np.random.randn(dim, dim))
        V, _ = np.linalg.qr(np.random.randn(dim, dim))
        
        # 奇异值分布：指数衰减 (模拟长尾知识)
        S_values = np.exp(-np.linspace(0, 5, target_rank)) 
        S = np.zeros(dim)
        S[:target_rank] = S_values
        
        self.Target = U @ np.diag(S) @ V.T
        self.Current_State = np.zeros((dim, dim))
        
        # 历史记录器
        self.history_velocity = []  # |u_t| 流速
        self.history_rank = []      # R_t   有效秩
        self.history_viscosity = [] # nu_t  粘度
        
        self.initial_effective_rank = self.compute_effective_rank(self.Target)

    def compute_effective_rank(self, matrix):
        """
        计算有效秩 (Stable Rank / Entropy-based)
        物理意义：衡量流体当前占据的相空间体积。
        """
        if np.all(matrix == 0): return 0
        _, s, _ = np.linalg.svd(matrix, full_matrices=False)
        # 归一化奇异值能量分布
        p = s / (np.sum(s) + 1e-12)
        # 计算熵指数 (Exp Entropy)
        entropy = -np.sum(p * np.log(p + 1e-12))
        return np.exp(entropy)

    def step(self, t, decay_rate=0.02):
        """
        执行一步 N-S 动力学更新 (The Navier-Stokes Update Step)
        核心公式: u_t ~ (Gradient * Compression) / Viscosity
        """
        # 1. 粘度衰减 (Viscosity Decay)
        # 物理意义：模拟 Attention 机制随着上下文过长或熵增而逐渐“疲劳”或失效。
        # 当粘度降低，流体从“糖浆”变为“超流体”，湍流不再被抑制。
        self.viscosity = self.viscosity * (1.0 - decay_rate)
        # 设置一个极小的下限，防止除零（但在数学上我们要研究它趋近于0的行为）
        if self.viscosity < 1e-5: self.viscosity = 1e-5 
        
        # 2. 计算残差梯度 (Pressure Gradient)
        R = self.Target - self.Current_State
        
        # 3. 计算当前的流体有效秩 (Effective Rank)
        # 物理意义：流体当前的“宽度”或“体积”。
        current_rank = self.compute_effective_rank(self.Current_State)
        if current_rank < 1.0: current_rank = 1.0
        
        # 4. 计算压缩因子 (Compression Factor) - 关键的非线性项
        # 物理意义：模拟三维流体中的“前沿压缩”或“涡旋拉伸”。
        # 当 Rank 下降（结晶/坍缩）时，等效流形体积减小，能量密度被迫增加。
        # 平方项 ( ^2 ) 引入了非线性，这是导致爆破的关键。
        compression = (self.initial_effective_rank / current_rank) ** 2 
        
        # 5. 更新速度 u_t
        # 动力学方程：速度正比于 (压力梯度 * 压缩率) / 粘度
        # 奇异点机制：分子(压缩)趋于无穷，分母(粘度)趋于零 -> 速度必然爆炸。
        grad_norm = np.linalg.norm(R)
        dt = 0.001 # 时间步长
        
        u_magnitude = (grad_norm * compression) / self.viscosity * dt
        
        # 更新状态 S(t+1) = S(t) + u(t)
        # 简化方向为沿残差梯度方向（主要关注能量模长的爆破）
        update_step = (R / (grad_norm + 1e-9)) * u_magnitude
        self.Current_State += update_step
        
        # 记录本步数据
        self.history_velocity.append(u_magnitude)
        self.history_rank.append(current_rank)
        self.history_viscosity.append(self.viscosity)

    def run(self, steps=100):
        print(f"Starting Simulation: Dim={self.dim}, TargetRank={self.initial_effective_rank:.2f}")
        for t in range(steps):
            self.step(t)
            
            # 安全熔断：如果爆发太剧烈，提前终止以防数值溢出 (NaN)
            if self.history_velocity[-1] > 1e10:
                print(f"!!! SINGULARITY REACHED at step {t} !!!")
                print(f"  - Velocity: {self.history_velocity[-1]:.2e}")
                print(f"  - Rank:     {self.history_rank[-1]:.2f}")
                print(f"  - Viscosity:{self.history_viscosity[-1]:.2e}")
                
                # 补齐剩余步数的数据以便绘图 (保持最后的值)
                remaining = steps - t - 1
                self.history_velocity.extend([self.history_velocity[-1]] * remaining)
                self.history_rank.extend([self.history_rank[-1]] * remaining)
                self.history_viscosity.extend([self.history_viscosity[-1]] * remaining)
                break

def plot_singularity_trap():
    # 运行模拟
    sim = SingularityTrapSim(dim=64, target_rank=20, initial_viscosity=1.0)
    sim.run(steps=100)
    
    t = np.arange(len(sim.history_velocity))
    
    # 创建双轴图表
    fig, ax1 = plt.subplots(figsize=(10, 7))
    
    # 绘制左轴：Rank (秩) 和 Viscosity (粘度)
    color_rank = 'tab:blue'
    ax1.set_xlabel('Inference Time (t)')
    ax1.set_ylabel('Effective Rank (Space) / Viscosity', color=color_rank, fontweight='bold')
    
    l1, = ax1.plot(t, sim.history_rank, color='blue', linestyle='-', linewidth=2, label='Effective Rank (Rank Collapse)')
    # 将粘度放大10倍以便在同一比例尺下观察趋势
    l2, = ax1.plot(t, np.array(sim.history_viscosity)*10, color='cyan', linestyle='--', label='Viscosity ($\\nu$) x10')
    
    ax1.tick_params(axis='y', labelcolor=color_rank)
    ax1.set_ylim(0, max(sim.history_rank)*1.1)
    
    # 绘制右轴：Velocity (速度) - 使用对数坐标展示爆炸
    ax2 = ax1.twinx()
    color_vel = 'tab:red'
    ax2.set_ylabel('Flow Velocity ($|u_t|$)', color=color_vel, fontweight='bold')
    # 使用 semilogy 对数轴，因为速度是指数级爆炸的
    l3, = ax2.semilogy(t, sim.history_velocity, color='red', linewidth=3, label='Flow Velocity (Singularity)')
    ax2.tick_params(axis='y', labelcolor=color_vel)
    
    # 标注奇异点时刻
    # 寻找速度跃升最剧烈的点
    blowup_idx = np.argmax(sim.history_velocity)
    # 仅当速度真的很大时才标注
    if sim.history_velocity[blowup_idx] > 100:
        plt.axvline(x=blowup_idx, color='k', linestyle=':', linewidth=1)
        
        # 添加文本标注
        ax2.text(blowup_idx - 15, sim.history_velocity[blowup_idx]*0.05, 
                 'Finite-Time\nSingularity', fontsize=12, fontweight='bold', color='red', ha='right')
        ax2.text(blowup_idx + 2, sim.history_velocity[blowup_idx]*0.05, 
                 'Rank Collapse\n(Compression)', fontsize=10, color='blue', ha='left')

    # 合并图例
    lines = [l1, l2, l3]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center left', frameon=True, framealpha=0.9)
    
    plt.title('The Singularity Trap: Rank Collapse Induces Velocity Blow-up\n(Proof of N-S Breakdown in Large Models)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图片
    save_path = 'singularity_trap_proof.png'
    plt.savefig(save_path, dpi=300)
    print(f"Experiment result saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    # 设置随机种子，保证"死结"的可复现性
    np.random.seed(2025) 
    plot_singularity_trap()
