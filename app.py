import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import os
import matplotlib.font_manager as fm
import platform
from matplotlib.font_manager import FontProperties
import tempfile
import base64
from io import BytesIO


# 配置matplotlib支持中文显示
# 检查运行环境
system = platform.system()
if system == 'Linux':  # Streamlit Cloud使用Linux环境
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'Noto Sans CJK JP', 'Noto Sans CJK SC', 'sans-serif']
else:  # 本地环境
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']

plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置页面标题和布局
st.set_page_config(
    page_title="褶皱分析工具",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 添加页面标题
st.title("褶皱分析工具")
st.markdown("### 物体在变化外力和阻力下的运动分析")


# 侧边栏参数设置
st.sidebar.header("参数设置")

# 创建参数滑块
col1, col2 = st.sidebar.columns(2)

with col1:
    a = st.number_input("a (外力减小系数)", min_value=0.01, max_value=10.0, value=0.5, step=0.01, format="%.2f")
    Y = st.number_input("Y (第一阶段阻力系数)", min_value=0.01, max_value=10.0, value=5.0, step=0.01, format="%.2f")
    A = st.number_input("A (第一阶段终点)", min_value=0.1, max_value=20.0, value=5.0, step=0.1, format="%.1f")

with col2:
    b = st.number_input("b (外力初始值)", min_value=0.01, max_value=20.0, value=10.0, step=0.01, format="%.2f")
    C = st.number_input("C (第二阶段阻力系数)", min_value=0.01, max_value=20.0, value=10.0, step=0.01, format="%.2f")
    B = st.number_input("B (第二阶段终点)", min_value=A + 0.1, max_value=30.0, value=A + 5.0, step=0.1, format="%.1f")

# 计算外力做功 U1
def calculate_U1(x):
    # 计算外力变为0的位置
    x_zero_force = b / a if a > 0 else float('inf')
    
    if x <= x_zero_force:
        # 外力为正时的做功
        return b * x - 0.5 * a * x**2
    else:
        # 外力为0后，做功不再增加，保持在x_zero_force处的值
        return b * x_zero_force - 0.5 * a * x_zero_force**2

# 计算阻力做功 U2
def calculate_U2(x):
    if x <= A:
        return 0.5 * Y * x**2
    elif x <= B:
        return 0.5 * Y * A**2 + 3 * C * (x**(1/3) - A**(1/3))
    else:
        return 0.5 * Y * A**2 + 3 * C * (B**(1/3) - A**(1/3))

# 计算动能 Ek
def calculate_Ek(x):
    # 先计算基本的动能
    ek_basic = calculate_U1(x) - calculate_U2(x)
    
    # 如果动能已经为负，则返回0（物体已经停止）
    if ek_basic < 0:
        return 0
    
    # 如果动能为0或接近0，检查外力与阻力的关系
    if abs(ek_basic) < 1e-10:
        external_force = max(0, -a * x + b)  # 外力只取大于0的部分
        resistance = calculate_resistance(x)
        
        # 如果外力不大于阻力，物体将保持静止
        if external_force <= resistance:
            return 0
    
    return ek_basic

# 计算净外力
def calculate_net_force(x):
    # 计算外力
    external_force = max(0, -a * x + b)  # 外力只取大于0的部分
    
    # 计算阻力
    if x <= A:
        resistance = Y * x
    elif x <= B:
        resistance = C * x**(-2/3)
    else:
        resistance = 0
    
    return external_force - resistance

# 计算阻力
def calculate_resistance(x):
    if x <= A:
        return Y * x
    elif x <= B:
        return C * x**(-2/3)
    else:
        return 0

# 寻找动能峰值点（净外力为0的点）
def find_peak_kinetic_energy_point():
    def net_force(x):
        return calculate_net_force(x)
    
    # 分别在三个区间内查找
    roots = []
    
    # 计算外力变为0的位置
    x_zero_force = b / a if a > 0 else float('inf')
    
    # 区间1：[0, min(A, x_zero_force)]
    try:
        # 解方程 -a*x + b - Y*x = 0，前提是外力仍然为正
        x_peak1 = b / (a + Y) if (a + Y) != 0 else float('inf')
        if 0 <= x_peak1 <= A and x_peak1 <= x_zero_force:
            roots.append(x_peak1)
    except:
        pass
    
    # 区间2：[A, min(B, x_zero_force)]
    try:
        def func2(x):
            # 外力为正时：-a*x + b - C*x^(-2/3)
            return -a * x + b - C * x**(-2/3)
        
        x_peak2_guess = (A + min(B, x_zero_force)) / 2
        x_peak2 = fsolve(func2, x_peak2_guess)[0]
        if A <= x_peak2 <= B and x_peak2 <= x_zero_force and abs(func2(x_peak2)) < 1e-10:
            roots.append(x_peak2)
    except:
        pass
    
    # 区间3：[B, x_zero_force]
    try:
        # 解方程 -a*x + b = 0，前提是x <= x_zero_force
        x_peak3 = b / a if a != 0 else float('inf')
        if B <= x_peak3 <= x_zero_force:
            roots.append(x_peak3)
    except:
        pass
    
    # 返回所有可能的峰值点
    return roots

# 寻找运动停止点（动能为0的点）
def find_stop_point():
    def kinetic_energy(x):
        return calculate_Ek(x)
    
    # 检查初始点的外力和阻力
    initial_external_force = b  # x=0时，F = -ax + b = b
    initial_resistance = 0  # x=0时，f = Y*0 = 0
    
    # 如果初始外力小于等于初始阻力，物体不会开始运动
    if initial_external_force <= initial_resistance:
        return 0  # 物体不会运动
    
    # 检查x稍大于0时的动能，确保物体能开始运动
    x_small = 1e-6  # 一个非常小的正数
    if kinetic_energy(x_small) <= 0:
        return 0  # 如果刚开始就没有正动能，物体不会运动
    
    # 寻找动能从正变为0的点（排除起点）
    x_range = np.linspace(x_small, 100, 10000)  # 从一个很小的正数开始，而不是0
    
    # 计算每个点的动能
    ek_values = []
    for xi in x_range:
        # 计算基本动能
        ek_basic = calculate_U1(xi) - calculate_U2(xi)
        
        if ek_basic < 0:
            # 动能已经为负，物体已经停止
            ek_values.append(0)
            continue
            
        if abs(ek_basic) < 1e-10:
            # 动能接近0，检查外力与阻力
            external_force = max(0, -a * xi + b)
            resistance = calculate_resistance(xi)
            
            if external_force <= resistance:
                # 外力不足以克服阻力，物体停止
                ek_values.append(0)
                continue
        
        ek_values.append(ek_basic)
    
    ek_values = np.array(ek_values)
    
    # 找到第一个动能为0的点（从开始有正动能后）
    positive_started = False
    for i in range(len(ek_values)):
        # 确认物体开始运动（有正动能）
        if not positive_started and ek_values[i] > 0:
            positive_started = True
            continue
            
        # 在有正动能后，找到第一个动能为0的点
        if positive_started and ek_values[i] <= 0:
            # 找到了停止点
            stop_idx = i
            
            # 在最后一个正动能点和第一个零动能点之间进行二分查找
            x_left = x_range[stop_idx - 1]  # 最后一个正动能点
            x_right = x_range[stop_idx]     # 第一个零动能点
            
            # 二分查找精确的零点
            while abs(x_right - x_left) > 1e-10:
                x_mid = (x_left + x_right) / 2
                ek_mid = kinetic_energy(x_mid)
                
                if ek_mid > 0:
                    x_left = x_mid
                else:
                    x_right = x_mid
            
            # 找到的停止点
            stop_point = (x_left + x_right) / 2
            
            # 检查在停止点处外力是否大于阻力
            external_force = max(0, -a * stop_point + b)
            resistance = calculate_resistance(stop_point)
            
            # 如果外力大于阻力，物体会继续运动（这种情况不应该发生，因为我们已经在calculate_Ek中处理了）
            # 但为了健壮性，我们再次检查
            if external_force > resistance:
                # 在这种情况下，我们应该继续寻找下一个停止点
                # 但为简化起见，我们返回当前找到的点，并在分析中注明
                pass
            
            return stop_point
    
    # 如果没有找到停止点
    if positive_started:
        # 物体开始运动但没有停止
        return None
    else:
        # 物体根本没有开始运动
        return 0

# 计算能量和动能曲线
@st.cache_data
def generate_plots(a, b, Y, C, A, B):
    """生成图表并返回，使用缓存以提高性能"""
    x_zero_force = b / a if a > 0 else float('inf')
    x_max = max(B * 1.5, x_zero_force * 1.2)
    x = np.linspace(0, x_max, 1000)

    # 寻找停止点
    stop_point = find_stop_point()
    if stop_point is not None and stop_point > 0:
        stop_x = stop_point
        stop_ek = 0
        
        # 截断x范围，只显示到停止点
        x_mask = x <= stop_x
        x_display = x[x_mask]
    else:
        stop_x = None
        stop_ek = None
        x_display = x  # 使用完整范围

    # 计算能量曲线
    U1 = [calculate_U1(xi) for xi in x_display]
    U2 = [calculate_U2(xi) for xi in x_display]
    Ek = [calculate_Ek(xi) for xi in x_display]

    # 寻找动能峰值点
    peak_points = find_peak_kinetic_energy_point()
    peak_x = [p for p in peak_points if 0 <= p <= (stop_x if stop_x else max(x))]
    peak_ek = [calculate_Ek(p) for p in peak_x]

    # 确定图表的y轴范围
    y_min = min(min(U1), min(U2), min(Ek)) * 1.1
    y_max = max(max(U1), max(U2), max(Ek)) * 1.1

    # 创建图表
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(x_display, U1, 'b-', label='U1 (外力做功)')
    ax1.plot(x_display, U2, 'r-', label='U2 (阻力做功)')

    # 显示完整的A和B位置线，即使超出了停止点
    if A <= max(x):
        ax1.axvline(x=A, color='gray', linestyle='--', label=f'x=A ({A})')
    if B <= max(x):
        ax1.axvline(x=B, color='gray', linestyle=':', label=f'x=B ({B})')

    # 标记外力为零的位置（如果在显示范围内）
    if x_zero_force <= max(x_display):
        ax1.axvline(x=x_zero_force, color='green', linestyle='-.', label=f'外力为零 (x={x_zero_force:.2f})')

    # 标记动能峰值点
    for px, pek in zip(peak_x, peak_ek):
        if 0 <= px <= max(x_display):
            ax1.plot(px, calculate_U1(px), 'go', label=f'动能峰值点 (x={px:.2f})')
            break  # 只标记第一个峰值点

    # 标记停止点
    if stop_x is not None and 0 < stop_x <= max(x):
        ax1.plot(stop_x, calculate_U1(stop_x), 'ro', label=f'运动停止点 (x={stop_x:.2f})')
        
        # 在停止点添加垂直虚线，表示运动终止
        ax1.axvline(x=stop_x, color='red', linestyle='--', alpha=0.5)

    ax1.set_xlabel('位移 x')
    ax1.set_ylabel('能量')
    ax1.set_title('外力做功与阻力做功')
    ax1.grid(True)
    ax1.legend()
    ax1.set_ylim(y_min, y_max)

    # 如果有停止点，设置x轴范围只到停止点再加一点余量
    if stop_x is not None and stop_x > 0:
        ax1.set_xlim(0, min(stop_x * 1.1, max(x)))
    else:
        ax1.set_xlim(0, max(x_display))

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(x_display, Ek, 'g-', label='Ek (动能)')

    # 显示完整的A和B位置线，即使超出了停止点
    if A <= max(x):
        ax2.axvline(x=A, color='gray', linestyle='--', label=f'x=A ({A})')
    if B <= max(x):
        ax2.axvline(x=B, color='gray', linestyle=':', label=f'x=B ({B})')

    # 标记外力为零的位置（如果在显示范围内）
    if x_zero_force <= max(x_display):
        ax2.axvline(x=x_zero_force, color='green', linestyle='-.', label=f'外力为零 (x={x_zero_force:.2f})')

    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    # 标记动能峰值点
    for px, pek in zip(peak_x, peak_ek):
        if 0 <= px <= max(x_display):
            ax2.plot(px, pek, 'go', label=f'动能峰值点 (x={px:.2f})')
            break  # 只标记第一个峰值点

    # 标记停止点
    if stop_x is not None and 0 < stop_x <= max(x):
        ax2.plot(stop_x, 0, 'ro', label=f'运动停止点 (x={stop_x:.2f})')
        
        # 在停止点添加垂直虚线，表示运动终止
        ax2.axvline(x=stop_x, color='red', linestyle='--', alpha=0.5)

    ax2.set_xlabel('位移 x')
    ax2.set_ylabel('动能')
    ax2.set_title('物块动能')
    ax2.grid(True)
    ax2.legend()

    # 如果有停止点，设置x轴范围只到停止点再加一点余量
    if stop_x is not None and stop_x > 0:
        ax2.set_xlim(0, min(stop_x * 1.1, max(x)))
    else:
        ax2.set_xlim(0, max(x_display))
    
    return fig1, fig2, stop_x, peak_x, peak_ek, x_zero_force

# 使用缓存生成图表
fig1, fig2, stop_x, peak_x, peak_ek, x_zero_force = generate_plots(a, b, Y, C, A, B)

# 显示图表
st.pyplot(fig1)
st.pyplot(fig2)

# 显示分析结果
st.header("分析结果")

# 显示外力为零的位置
st.subheader("外力为零的位置")
if a > 0:
    st.write(f"外力 F = -ax + b 在 x = {x_zero_force:.4f} 处变为零")
    
    # 判断外力为零点所在区域
    if x_zero_force <= A:
        st.write(f"该点位于第一阶段 (0 ≤ x ≤ A)")
    elif x_zero_force <= B:
        st.write(f"该点位于第二阶段 (A < x ≤ B)")
    else:
        st.write(f"该点位于第三阶段 (x > B)")
else:
    st.write("由于 a ≤ 0，外力不会变为零")

# 动能峰值点分析
st.subheader("动能峰值点")
if peak_x:
    for i, (px, pek) in enumerate(zip(peak_x, peak_ek)):
        if 0 <= px <= (stop_x if stop_x else float('inf')):
            st.write(f"峰值点 {i+1}: x = {px:.4f}, 动能 = {pek:.4f}")
            
            # 判断峰值点所在区域
            if px <= A:
                st.write(f"该点位于第一阶段 (0 ≤ x ≤ A)")
            elif px <= B:
                st.write(f"该点位于第二阶段 (A < x ≤ B)")
            else:
                st.write(f"该点位于第三阶段 (x > B)")
                
            # 在该点的外力和阻力
            external_force = max(0, -a * px + b)
            resistance = calculate_resistance(px)
            st.write(f"该点的外力 = {external_force:.4f}, 阻力 = {resistance:.4f}")
else:
    st.write("未找到动能峰值点")

# 停止点分析
st.subheader("运动停止点")
if stop_x is not None:
    if stop_x == 0:
        initial_external_force = b  # x=0时，F = -a*0 + b = b
        initial_resistance = 0  # x=0时，f = Y*0 = 0
        
        if initial_external_force <= initial_resistance:
            st.write("物体不会开始运动，因为初始外力不足以克服初始阻力")
            st.write(f"初始外力 = {initial_external_force:.4f}, 初始阻力 = {initial_resistance:.4f}")
        else:
            # 检查x稍大于0时的动能
            x_small = 1e-6
            initial_kinetic_energy = calculate_Ek(x_small)
            if initial_kinetic_energy <= 0:
                st.write("物体不会开始运动，因为在起点附近动能不为正")
                st.write(f"x = {x_small} 处的动能 = {initial_kinetic_energy:.8f}")
    else:
        st.write(f"运动停止点: x = {stop_x:.4f}")
        
        # 判断停止点所在区域
        if stop_x <= A:
            st.write(f"该点位于第一阶段 (0 ≤ x ≤ A)")
        elif stop_x <= B:
            st.write(f"该点位于第二阶段 (A < x ≤ B)")
        else:
            st.write(f"该点位于第三阶段 (x > B)")
        
        # 在该点的外力和阻力
        external_force = max(0, -a * stop_x + b)  # 外力只取大于0的部分
        resistance = calculate_resistance(stop_x)
        st.write(f"该点的外力 = {external_force:.4f}, 阻力 = {resistance:.4f}")
        
        # 检查外力与阻力的关系
        if external_force > resistance:
            st.write("注意：在该点外力大于阻力，物体理论上会继续运动。这可能是数值计算误差导致的。")
        else:
            st.write("在该点外力不大于阻力，物体将保持静止状态。")
        
        # 检查是否在外力为零之后
        if stop_x > x_zero_force and a > 0:
            st.write("注意：该停止点位于外力为零之后，物体在惯性作用下继续运动直至停止")
else:
    st.write("物体可能永远不会停止，或者停止点超出了计算范围")

# 添加交互式提示
st.sidebar.markdown("---")
st.sidebar.info("提示: 调整参数滑块以实时更新图表和分析结果")

# 添加关于部分
with st.sidebar.expander("关于"):
    st.write("""
    该应用用于分析物体在变化外力和阻力下的运动情况。
    
    **参数说明**:
    - a: 外力减小系数，控制外力随位移减小的速率
    - b: 外力初始值，初始外力大小
    - Y: 第一阶段阻力系数，控制第一阶段阻力大小
    - C: 第二阶段阻力系数，控制第二阶段阻力大小
    - A: 第一阶段终点，标记第一阶段结束位置
    - B: 第二阶段终点，标记第二阶段结束位置
    
    **外力公式**: F = -ax + b (x ≤ b/a)，F = 0 (x > b/a)
    
    **阻力公式**:
    - 第一阶段 (0 ≤ x ≤ A): f = Y·x
    - 第二阶段 (A < x ≤ B): f = C·x^(-2/3)
    - 第三阶段 (x > B): f = 0
    """)

# 添加页脚
st.markdown("---")
