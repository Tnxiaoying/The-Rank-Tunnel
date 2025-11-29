import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# 字体设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class SimplePCA:
    """
    一个简单的 PCA 实现，移除 sklearn 依赖
    使用 SVD 进行降维
    """
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None

    def fit(self, X):
        # 中心化数据
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # 计算 SVD
        # X_centered = U * S * Vt
        # V 的前 n_components 行就是主成分
        # 注意: np.linalg.svd 返回的是 Vt
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        self.components_ = Vt[:self.n_components]
        
    def transform(self, X):
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_.T)

class FluidRankTunnel:
    """
    Rank Tunnel - Fluid Dynamics Edition
    模拟推理作为流体动力学过程：层流(Zero-shot) vs 湍流(CoT)
    """
    def __init__(self, dim=64, target_rank=20, model_capacity=1, viscosity=1.0):
        """
        :param viscosity: 粘性 (nu). 
           - High Viscosity (Low Temp) -> Laminar Flow (Conservative)
           - Low Viscosity (High Temp) -> Turbulent Flow (Creative/Hallucinatory)
        """
        self.dim = dim
        self.target_rank = target_rank
        self.model_capacity = model_capacity
        # 温度越高，粘性越低 (T ~ 1/nu)
        # 我们用 viscosity 来控制噪声幅度: noise ~ 1/viscosity
        self.viscosity = max(0.01, viscosity) 
        self.g = 0.05 / self.viscosity # 基础引力常数受粘性调节
        
        # 1. 生成固体障碍物 (The Solid Mass)
        U, _ = np.linalg.qr(np.random.randn(dim, dim))
        V, _ = np.linalg.qr(np.random.randn(dim, dim))
        S = np.zeros(dim)
        S[:target_rank] = np.linspace(10, 1, target_rank)
        self.Target = U @ np.diag(S) @ V.T
        self.initial_norm = np.linalg.norm(self.Target)
        
        # 2. 流体状态
        self.Current_State = np.zeros((dim, dim))
        self.history_states = [np.zeros(dim*dim)] # 用于PCA轨迹
        self.velocities = [] # u_t
        self.kinetic_energies = [] # E_k
        self.vorticities = [] # Omega (Curvature)
        
        self.locked = False

    def step(self):
        # 计算残差 (Pressure Gradient)
        R = self.Target - self.Current_State
        res_norm = np.linalg.norm(R)
        
        if res_norm / self.initial_norm < 0.15:
            self.locked = True
            # 锁定后，速度趋于0，不再产生湍流
            self.velocities.append(0)
            self.kinetic_energies.append(0)
            self.vorticities.append(0)
            self.history_states.append(self.Current_State.flatten())
            return

        # N-S 动力学模拟
        U, S, Vt = np.linalg.svd(R, full_matrices=False)
        k = self.model_capacity
        
        # 引入热涨落 (Thermal Fluctuations / Turbulence)
        # 噪声幅度与粘性成反比
        noise_scale = self.g 
        noise_u = np.random.randn(*U[:, :k].shape) * noise_scale
        noise_v = np.random.randn(*Vt[:k, :].shape) * noise_scale
        
        U_noisy = U[:, :k] + noise_u
        Vt_noisy = Vt[:k, :] + noise_v
        
        # 计算流速 u_t (Update Vector)
        Delta = (U_noisy @ np.diag(S[:k]) @ Vt_noisy)
        
        # 更新状态 S_{t+1} = S_t + u_t
        self.Current_State += Delta
        
        # --- 流体力学指标记录 ---
        
        # 1. 流速 magnitude
        u_mag = np.linalg.norm(Delta)
        self.velocities.append(u_mag)
        
        # 2. 动能 E_k = 0.5 * |u|^2
        self.kinetic_energies.append(0.5 * u_mag**2)
        
        # 3. 涡度/曲率 (Tortuosity/Vorticity)
        # 计算本次更新向量与上一次更新向量的夹角变化
        # 如果一直沿直线走，夹角为0，涡度低；如果反复横跳，涡度高
        if len(self.history_states) > 1:
            prev_state = self.history_states[-1].reshape(self.dim, self.dim)
            prev_prev_state = self.history_states[-2].reshape(self.dim, self.dim)
            u_prev = prev_state - prev_prev_state
            
            # Cosine Similarity
            norm_prod = np.linalg.norm(Delta) * np.linalg.norm(u_prev)
            if norm_prod > 1e-9:
                cos_theta = np.sum(Delta * u_prev) / norm_prod
                # 涡度定义为方向偏离程度: 1 - cos_theta (0=直线, 2=反向)
                vorticity = 1.0 - cos_theta 
            else:
                vorticity = 0
        else:
            vorticity = 0
        self.vorticities.append(vorticity)
        
        self.history_states.append(self.Current_State.flatten())

    def run(self, steps=100):
        for _ in range(steps):
            self.step()

# --- 实验 1: 思维流线图 (Streamlines of Thought) ---
# 验证：层流 (Laminar) vs 湍流 (Turbulent)
def plot_streamlines():
    print("Generating Figure 1: Streamlines of Thought (PCA)...")
    
    # 场景 A: 简单问题 (Low Mass) + 高粘性 -> 层流 (Zero-shot like)
    sim_laminar = FluidRankTunnel(dim=64, target_rank=5, viscosity=2.0)
    sim_laminar.run(steps=50)
    
    # 场景 B: 困难问题 (High Mass) + 低粘性 -> 湍流 (CoT like)
    sim_turbulent = FluidRankTunnel(dim=64, target_rank=30, viscosity=0.5)
    sim_turbulent.run(steps=80)
    
    # PCA 降维投影到 2D 平面
    data_laminar = np.array(sim_laminar.history_states)
    data_turbulent = np.array(sim_turbulent.history_states)
    
    # 使用自定义 SimplePCA
    pca = SimplePCA(n_components=2)
    # 混合数据训练 PCA 以统一坐标系
    combined = np.vstack([data_laminar, data_turbulent])
    pca.fit(combined)
    
    traj_lam = pca.transform(data_laminar)
    traj_tur = pca.transform(data_turbulent)
    
    plt.figure(figsize=(10, 8))
    
    # 绘制层流轨迹
    plt.plot(traj_lam[:, 0], traj_lam[:, 1], 'b-o', markersize=4, label='Laminar Flow (Low Mass, High Viscosity)', alpha=0.7)
    # 标注起点终点
    plt.text(traj_lam[0,0], traj_lam[0,1], 'Start', color='b', fontweight='bold')
    plt.text(traj_lam[-1,0], traj_lam[-1,1], 'Solution', color='b', fontweight='bold')
    
    # 绘制湍流轨迹
    plt.plot(traj_tur[:, 0], traj_tur[:, 1], 'r-^', markersize=4, label='Turbulent Flow (High Mass, Low Viscosity)', alpha=0.7)
    plt.text(traj_tur[0,0], traj_tur[0,1], 'Start', color='r', fontweight='bold')
    # 湍流可能未完全到达终点或路径极其曲折
    
    plt.title('Streamlines of Thought: Laminar vs Turbulent Reasoning Paths')
    plt.xlabel('Principal Component 1 (State Space)')
    plt.ylabel('Principal Component 2 (State Space)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('fluid_streamlines.png')
    plt.close()

# --- 实验 2: 寻找漩涡 - 涡度相变图 (Vorticity Phase Diagram) ---
# 重绘 Phase Diagram，但这次颜色代表 "最大涡度"，以寻找 Blue Knot
def plot_vorticity_map():
    print("Generating Figure 2: Vorticity Phase Map (Hunting the Vortex)...")
    
    mass_levels = np.linspace(1, 40, 30)
    time_steps = np.linspace(1, 80, 50).astype(int)
    vorticity_map = np.zeros((len(mass_levels), len(time_steps)))
    
    for i, mass in enumerate(mass_levels):
        for j, t in enumerate(time_steps):
            # 使用标准粘性
            sim = FluidRankTunnel(dim=64, target_rank=int(mass), viscosity=1.0)
            sim.run(steps=t)
            
            # 记录该次推理中的 最大涡度 (Max Vorticity)
            # 这代表了思维过程中最“纠结”的时刻
            if len(sim.vorticities) > 0:
                max_vort = np.max(sim.vorticities)
            else:
                max_vort = 0
            vorticity_map[i, j] = max_vort

    plt.figure(figsize=(10, 8))
    extent = [time_steps[0], time_steps[-1], mass_levels[0], mass_levels[-1]]
    
    # 使用 'viridis' 或 'plasma' 来高亮高涡度区域
    plt.imshow(vorticity_map, aspect='auto', cmap="plasma", origin='lower', extent=extent)
    plt.colorbar(label='Max Vorticity (Curvature of Thought)')
    
    # 绘制之前的锁定边界作为参考
    # 我们预期高涡度区域应该出现在边界附近 (Blue Knot 所在处)
    plt.contour(vorticity_map, levels=[0.5], colors='cyan', linestyles='dashed', linewidths=1, extent=extent)
    
    plt.xlabel('Inference Time')
    plt.ylabel('Problem Mass')
    plt.title('Vorticity Phase Diagram: Locating the "Cognitive Vortices"')
    plt.savefig('vorticity_phase_diagram.png')
    plt.close()

# --- 实验 3: 能量级联 (Energy Cascade) ---
# 验证 Kolmogorov Spectrum: Log-Log 频谱分析
def plot_energy_cascade():
    print("Generating Figure 3: Energy Cascade Spectrum...")
    
    # 运行一个长时间的湍流模拟
    sim = FluidRankTunnel(dim=128, target_rank=60, viscosity=0.3) # 低粘性，高难度
    sim.run(steps=200)
    
    # 获取速度序列 (动能变化的平方根)
    velocities = np.array(sim.velocities)
    # 去除锁定后的零值
    velocities = velocities[velocities > 1e-6]
    
    if len(velocities) < 10:
        print("Simulation converged too fast for spectrum analysis.")
        return

    # FFT 频谱分析
    N = len(velocities)
    yf = fft(velocities)
    xf = fftfreq(N, 1)[:N//2]
    power = 2.0/N * np.abs(yf[0:N//2])
    
    plt.figure(figsize=(8, 6))
    
    # Log-Log Plot
    plt.loglog(xf[1:], power[1:], 'b-', label='Simulation Spectrum')
    
    # 绘制理论参考线 (Kolmogorov -5/3 律 或类似标度律)
    # 这里我们只是示意，斜率可能不同，但应为幂律衰减
    if len(xf) > 5:
        ref_x = xf[int(N/10):int(N/2)]
        # 假设斜率 k
        ref_y = ref_x**(-1.5) * (power[int(N/10)] / ref_x[0]**(-1.5))
        plt.loglog(ref_x, ref_y, 'r--', label='Power Law Reference ($k^{-1.5}$)')
    
    plt.xlabel('Frequency (1/Time Step)')
    plt.ylabel('Energy Density (Spectral Power)')
    plt.title('Energy Cascade of Reasoning: The Spectrum of Thought')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.savefig('energy_cascade.png')
    plt.close()

if __name__ == "__main__":
    np.random.seed(42)
    plot_streamlines()
    plot_vorticity_map()
    plot_energy_cascade()
    print("Fluid Dynamics Simulation Complete.")
