import numpy as np
import matplotlib.pyplot as plt

# 字体设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def plot_fractal_barrier():
    # 1. 定义田氏临界方程
    def p_critical(d):
        return 5 + 24 / (d - 3)

    # 生成数据点
    d_right = np.linspace(3.05, 6, 200) # d > 3 的区域 (超临界区)
    d_left = np.linspace(1, 2.95, 200)  # d < 3 的区域 (次临界区)
    
    p_right = p_critical(d_right)
    p_left = p_critical(d_left)

    fig, ax = plt.subplots(figsize=(10, 8))

    # 2. 绘制理论曲线
    ax.plot(d_right, p_right, 'k-', linewidth=2.5, label='Tian Fractal Criticality: $p(d) = 5 + \\frac{24}{d-3}$')
    ax.plot(d_left, p_left, 'k--', linewidth=1, alpha=0.5) # 左侧虚线，表示非物理区

    # 3. 绘制垂直渐近线 (d=3 的无限壁垒)
    ax.axvline(x=3, color='red', linestyle='-', linewidth=3, alpha=0.3)
    ax.text(3.05, 80, 'The Infinite Barrier\n(3D Navier-Stokes)', color='red', fontsize=12, fontweight='bold')

    # 4. 标注历史锚点 (Wei et al.)
    ax.plot(4, 29, 'bo', markersize=10, label='Wei et al. (2024): d=4, p=29')
    ax.plot(5, 17, 'go', markersize=10, label='Wei et al. (2024): d=5, p=17')
    
    # 添加注释
    ax.annotate('Blow-up Possible\n(Front Compression)', xy=(4, 29), xytext=(4.5, 35),
                arrowprops=dict(facecolor='black', shrink=0.05))

    # 5. 绘制 "维度隧道效应" (Dimensional Tunneling)
    # 模拟 AI 的轨迹：从高维 (d=6) 迅速坍缩穿过 d=3
    tunnel_x = np.linspace(6, 1.5, 100)
    # AI 的非线性强度是有限的 (比如 p=20)
    tunnel_y = np.ones_like(tunnel_x) * 20 
    
    ax.arrow(5.8, 20, -4.0, 0, head_width=2, head_length=0.2, fc='purple', ec='purple', linewidth=3, 
             label='AI Dimensional Tunneling\n(Rank Collapse)', zorder=10)
    
    ax.text(2.0, 22, 'Tunneling via\nRank Collapse', color='purple', fontweight='bold', ha='center')

    # 设置图表属性
    ax.set_ylim(0, 100)
    ax.set_xlim(1, 6)
    ax.set_xlabel('Dimension ($d$)', fontsize=12)
    ax.set_ylabel('Critical Nonlinearity ($p$)', fontsize=12)
    ax.set_title('The Mathematical Barrier & Dimensional Tunneling\n(Why 3D N-S is Hard & How AI Cheats)', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # 填充颜色
    ax.fill_between(d_right, p_right, 100, color='gray', alpha=0.1, label='Stable Region')

    plt.tight_layout()
    plt.savefig('fractal_barrier_tunneling.png')
    print("理论验证图 'fractal_barrier_tunneling.png' 已生成。")
    plt.show()

if __name__ == "__main__":
    plot_fractal_barrier()
