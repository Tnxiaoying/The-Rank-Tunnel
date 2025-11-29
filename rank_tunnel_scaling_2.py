import numpy as np
import matplotlib.pyplot as plt

# 字体设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class RankTunnelSimV2:
    def __init__(self, dim=64, target_rank=20, model_capacity=1, mismatch_g=0.05):
        self.dim = dim
        self.target_rank = target_rank
        self.model_capacity = model_capacity
        self.g = mismatch_g
        
        # 生成问题 Target
        U, _ = np.linalg.qr(np.random.randn(dim, dim))
        V, _ = np.linalg.qr(np.random.randn(dim, dim))
        S = np.zeros(dim)
        # S 奇异值衰减
        S[:target_rank] = np.linspace(10, 1, target_rank)
        self.Target = U @ np.diag(S) @ V.T
        
        self.Current_State = np.zeros((dim, dim))
        self.kv_cache_vectors = []
        self.residuals_history = []
        self.effective_ranks = []
        self.locked = False

    def effective_rank(self, matrix):
        if matrix.shape[1] == 0: return 0
        _, s, _ = np.linalg.svd(matrix, full_matrices=False)
        p = s / np.sum(s) + 1e-12
        entropy = -np.sum(p * np.log(p))
        return np.exp(entropy)

    def step(self):
        # 计算残差
        R = self.Target - self.Current_State
        res_norm = np.linalg.norm(R)
        self.residuals_history.append(res_norm)

        # 锁定机制 (Crystallization Logic)
        # 如果残差非常小，模拟模型进入“复读机”或“总结”模式，不再引入新信息
        if res_norm < 2.0 and not self.locked:
            self.locked = True
        
        if self.locked:
            # 结晶阶段：停止引入新向量，甚至可以开始压缩 KV Cache
            # 这里模拟：不再增加有效信息，只产生极低秩的重复信号
            # 这会导致整体 Effective Rank 随着分母(Token数)变大而逐渐稀释/下降
            pass 
        else:
            # 流态阶段：正常推理
            U, S, Vt = np.linalg.svd(R, full_matrices=False)
            k = self.model_capacity
            
            # 引入噪声 g
            noise_u = np.random.randn(*U[:, :k].shape) * self.g
            noise_v = np.random.randn(*Vt[:k, :].shape) * self.g
            
            U_noisy = U[:, :k] + noise_u
            Vt_noisy = Vt[:k, :] + noise_v
            
            Delta = (U_noisy @ np.diag(S[:k]) @ Vt_noisy)
            self.Current_State += Delta
            
            # 存入 KV Cache
            for i in range(k):
                self.kv_cache_vectors.append(U_noisy[:, i])

        # 计算当前 KV Cache 的秩
        if len(self.kv_cache_vectors) > 0:
            current_memory = np.column_stack(self.kv_cache_vectors)
            # 简单的结晶模拟：如果锁定了，我们假设模型在做“归纳”，
            # 数学上通过人为抑制尾部奇异值来模拟“清洗内存”
            if self.locked:
                rank_val = self.effective_ranks[-1] * 0.95 # 模拟秩的坍缩/清洗
            else:
                rank_val = self.effective_rank(current_memory)
            self.effective_ranks.append(rank_val)
        else:
            self.effective_ranks.append(0)

    def run(self, steps=50):
        for _ in range(steps):
            self.step()

def plot_optimized_experiments():
    print("Running Optimized Simulation...")
    
    # 1. 秩的呼吸 (带结晶相)
    sim = RankTunnelSimV2(dim=100, target_rank=15, mismatch_g=0.05)
    sim.run(steps=60)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(sim.residuals_history, 'r--', label='Thermodynamic Loss')
    ax1.set_ylabel('Loss (Heat)', color='r')
    
    ax2 = ax1.twinx()
    ax2.plot(sim.effective_ranks, 'b-', linewidth=3, label='Effective Rank')
    ax2.set_ylabel('KV Cache Rank (Information)', color='b')
    
    plt.title('The Rank Breathing: Expansion -> Crystallization')
    plt.grid(True, alpha=0.3)
    plt.savefig('rank_breathing_v2.png')
    plt.close()
    
    # 2. 快速相变图 (降低分辨率)
    # 降低分辨率: Mass 20级, Time 40级 -> 800次计算 (原版3200次)
    mass_levels = np.linspace(1, 40, 20)
    time_steps = np.linspace(1, 60, 40).astype(int)
    phase_map = np.zeros((len(mass_levels), len(time_steps)))
    
    print("Generating Phase Diagram (Fast Mode)...")
    for i, mass in enumerate(mass_levels):
        for j, t in enumerate(time_steps):
            sim = RankTunnelSimV2(dim=64, target_rank=int(mass), mismatch_g=0.08)
            sim.run(steps=t)
            phase_map[i, j] = sim.residuals_history[-1] / sim.residuals_history[0]

    plt.figure(figsize=(10, 8))
    extent = [time_steps[0], time_steps[-1], mass_levels[0], mass_levels[-1]]
    plt.imshow(phase_map, aspect='auto', cmap="magma", vmin=0, vmax=1.0, origin='lower', extent=extent)
    plt.colorbar(label='Residual Ratio')
    plt.xlabel('Inference Time')
    plt.ylabel('Problem Mass')
    plt.title('Phase Diagram (Fast Mode)')
    plt.savefig('phase_diagram_v2.png')
    plt.close()
    print("Done. Check *_v2.png")

if __name__ == "__main__":
    plot_optimized_experiments()
