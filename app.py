import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import platform
import pandas as pd
from scipy.integrate import solve_ivp

# 初始化session_state中的参数，如果不存在的话
if 'a' not in st.session_state:
    st.session_state.a = 40.0
if 'b' not in st.session_state:
    st.session_state.b = 50.0
if 'Y' not in st.session_state:
    st.session_state.Y = 5.0
if 'C' not in st.session_state:
    st.session_state.C = 10.0
if 't_max' not in st.session_state:
    st.session_state.t_max = 5.0
if 'user_presets' not in st.session_state:
    st.session_state.user_presets = {}

# 定义回调函数，用于在一个控件更改时更新另一个控件
def update_a():
    st.session_state.a = st.session_state.a_slider
    
def update_a_slider():
    st.session_state.a_slider = st.session_state.a_number
    
def update_b():
    st.session_state.b = st.session_state.b_slider
    
def update_b_slider():
    st.session_state.b_slider = st.session_state.b_number
    
def update_Y():
    st.session_state.Y = st.session_state.Y_slider
    
def update_Y_slider():
    st.session_state.Y_slider = st.session_state.Y_number
    
def update_C():
    st.session_state.C = st.session_state.C_slider
    
def update_C_slider():
    st.session_state.C_slider = st.session_state.C_number
    
def update_t_max():
    st.session_state.t_max = st.session_state.t_max_slider
    
def update_t_max_slider():
    st.session_state.t_max_slider = st.session_state.t_max_number

# 加载预设参数的函数
def load_preset():
    preset_name = st.session_state.preset_select
    if preset_name in st.session_state.user_presets:
        preset = st.session_state.user_presets[preset_name]
        st.session_state.a = preset["a"]
        st.session_state.b = preset["b"]
        st.session_state.Y = preset["Y"]
        st.session_state.C = preset["C"]
        st.session_state.t_max = preset["t_max"]
        
        # 同步滑动条和数值输入框
        st.session_state.a_slider = preset["a"]
        st.session_state.a_number = preset["a"]
        st.session_state.b_slider = preset["b"]
        st.session_state.b_number = preset["b"]
        st.session_state.Y_slider = preset["Y"]
        st.session_state.Y_number = preset["Y"]
        st.session_state.C_slider = preset["C"]
        st.session_state.C_number = preset["C"]
        st.session_state.t_max_slider = preset["t_max"]
        st.session_state.t_max_number = preset["t_max"]

# 保存当前参数为新预设
def save_current_preset():
    preset_name = st.session_state.new_preset_name
    if preset_name:  # 只要名称不为空就可以保存
        st.session_state.user_presets[preset_name] = {
            "a": st.session_state.a,
            "b": st.session_state.b,
            "Y": st.session_state.Y,
            "C": st.session_state.C,
            "t_max": st.session_state.t_max
        }
        # 清空输入框
        st.session_state.new_preset_name = ""

# 重置为默认参数
def reset_to_default():
    # 使用硬编码的默认值
    default_a = 40.0
    default_b = 50.0
    default_Y = 5.0
    default_C = 10.0
    default_t_max = 5.0
    
    st.session_state.a = default_a
    st.session_state.b = default_b
    st.session_state.Y = default_Y
    st.session_state.C = default_C
    st.session_state.t_max = default_t_max
    
    # 同步滑动条和数值输入框
    st.session_state.a_slider = default_a
    st.session_state.a_number = default_a
    st.session_state.b_slider = default_b
    st.session_state.b_number = default_b
    st.session_state.Y_slider = default_Y
    st.session_state.Y_number = default_Y
    st.session_state.C_slider = default_C
    st.session_state.C_number = default_C
    st.session_state.t_max_slider = default_t_max
    st.session_state.t_max_number = default_t_max

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

# 固定的阻力区域边界
A = 2  # 第一阶段终点
B = 4  # 第二阶段终点

# 侧边栏参数设置
st.sidebar.header("参数输入")

# 参数预设选择
st.sidebar.markdown("### 自定义预设")

# 如果有用户预设，显示选择框
if st.session_state.user_presets:
    st.sidebar.selectbox(
        "选择已保存的预设",
        options=list(st.session_state.user_presets.keys()),
        key="preset_select",
        on_change=load_preset
    )

# 保存当前参数为新预设
preset_col1, preset_col2 = st.sidebar.columns([3, 1])
with preset_col1:
    st.text_input("输入新预设名称", key="new_preset_name", placeholder="我的预设")
with preset_col2:
    st.button("保存", on_click=save_current_preset)

# 重置按钮
st.sidebar.button("重置为默认参数", on_click=reset_to_default)

st.sidebar.markdown("---")

# a 参数 (外力系数)
st.sidebar.markdown("### a (外力系数)")
a_col1, a_col2 = st.sidebar.columns([3, 2])
with a_col1:
    a = st.slider('滑动调节', 0.0, 100.0, st.session_state.a, step=0.1, key="a_slider", on_change=update_a)
with a_col2:
    a = st.number_input('精确值', 0.0, 100.0, st.session_state.a, step=0.1, format="%.1f", key="a_number", on_change=update_a_slider)

# b 参数 (外力常数)
st.sidebar.markdown("### b (外力常数)")
b_col1, b_col2 = st.sidebar.columns([3, 2])
with b_col1:
    b = st.slider('滑动调节', 0.0, 100.0, st.session_state.b, step=0.1, key="b_slider", on_change=update_b)
with b_col2:
    b = st.number_input('精确值', 0.0, 100.0, st.session_state.b, step=0.1, format="%.1f", key="b_number", on_change=update_b_slider)

# Y 参数 (第一阶段阻力系数)
st.sidebar.markdown("### Y (第一阶段阻力系数)")
Y_col1, Y_col2 = st.sidebar.columns([3, 2])
with Y_col1:
    Y = st.slider('滑动调节', 0.01, 10.0, st.session_state.Y, step=0.01, key="Y_slider", on_change=update_Y)
with Y_col2:
    Y = st.number_input('精确值', 0.01, 10.0, st.session_state.Y, step=0.01, format="%.2f", key="Y_number", on_change=update_Y_slider)

# C 参数 (第二阶段阻力系数)
st.sidebar.markdown("### C (第二阶段阻力系数)")
C_col1, C_col2 = st.sidebar.columns([3, 2])
with C_col1:
    C = st.slider('滑动调节', 0.01, 20.0, st.session_state.C, step=0.01, key="C_slider", on_change=update_C)
with C_col2:
    C = st.number_input('精确值', 0.01, 20.0, st.session_state.C, step=0.01, format="%.2f", key="C_number", on_change=update_C_slider)

# 质量固定为1
m = 1.0

# 最大模拟时间参数
st.sidebar.markdown("### t_max (最大模拟时间)")
t_max_col1, t_max_col2 = st.sidebar.columns([3, 2])
with t_max_col1:
    t_max = st.slider('滑动调节', 0.1, 50.0, st.session_state.t_max, step=0.1, key="t_max_slider", on_change=update_t_max)
with t_max_col2:
    t_max = st.number_input('精确值', 0.1, 50.0, st.session_state.t_max, step=0.1, format="%.1f", key="t_max_number", on_change=update_t_max_slider)

# 保存当前参数到session_state
st.session_state.a = a
st.session_state.b = b
st.session_state.Y = Y
st.session_state.C = C
st.session_state.t_max = t_max

# 显示固定参数
st.sidebar.markdown("---")
st.sidebar.markdown("**固定参数**")
st.sidebar.markdown(f"m (质量) = {m}")
st.sidebar.markdown(f"A (第一阶段终点) = {A}")
st.sidebar.markdown(f"B (第二阶段终点) = {B}")

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
        # 第一阶段 (0 ≤ x ≤ A=2): f = Y·x
        return 0.5 * Y * x**2
    elif x <= B:
        # 第二阶段 (A=2 < x ≤ B=4): f = C·x^(-2/3)
        return 2*Y + 3*C * (x**(1/3) - A**(1/3))
    else:
        # 第三阶段 (x > B=4): f = 0
        return 2*Y + 3*C * (B**(1/3) - A**(1/3))

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
    resistance = calculate_resistance(x)
    
    return external_force - resistance

# 计算阻力
def calculate_resistance(x):
    if x <= A:
        # 第一阶段 (0 ≤ x ≤ A=2): f = Y·x
        return Y * x
    elif x <= B:
        # 第二阶段 (A=2 < x ≤ B=4): f = C·x^(-2/3)
        return C * x**(-2/3)
    else:
        # 第三阶段 (x > B=4): f = 0
        return 0

# 添加关于部分
with st.sidebar.expander("关于"):
    st.write("""
    该应用用于分析物体在变化外力和阻力下的运动情况。
    
    **参数说明**:
    - a: 外力减小系数，控制外力随位移减小的速率
    - b: 外力初始值，初始外力大小
    - Y: 第一阶段阻力系数，控制第一阶段阻力大小
    - C: 第二阶段阻力系数，控制第二阶段阻力大小
    - m: 物体质量，固定为 1
    - A: 第一阶段终点，固定为 2
    - B: 第二阶段终点，固定为 4
    
    **外力公式**: F = -ax + b (x ≤ b/a)，F = 0 (x > b/a)
    
    **阻力公式**:
    - 第一阶段 (0 ≤ x ≤ 2): f = Y·x
    - 第二阶段 (2 < x ≤ 4): f = C·x^(-2/3)
    - 第三阶段 (x > 4): f = 0
    """)

# 添加页脚
st.markdown("---")
st.markdown("### 物理模型说明")
st.markdown("""
针对物体质量 m=1，初始位移 x=0，初始速度 v=0，外力 F = -ax + b（其中 a 和 b 为正定量参数），F≥0（即0 ≤ x ≤ b/a），以及阻力分区域定义：
- 在 0 ≤ x ≤ 2 区域，阻力 f = Y·x（Y 为正定量参数）
- 在 2 < x ≤ 4 区域，阻力 f = C·x^(-2/3)（C 为正定量参数）
- 在 x > 4 区域，阻力为 0

外力做功 U1 = b·x - (a/2)·x^2（当 x ≤ b/a）

阻力做功 U2:
- 当 0 ≤ x ≤ 2: U2 = (Y/2)·x^2
- 当 2 < x ≤ 4: U2 = 2Y + 3C·(x^(1/3) - 2^(1/3))
- 当 x > 4: U2 = 2Y + 3C·(4^(1/3) - 2^(1/3))

动能 Ek = U1 - U2
""")

# 缓存时间模式的计算结果
@st.cache_data
def solve_motion_over_time(a, b, Y, C, A, B, m, t_max):
    # 初始条件检查
    initial_force = b  # x=0时的外力
    initial_resistance = 0  # x=0时的阻力
    
    if initial_force <= initial_resistance:
        return np.array([0]), np.array([0]), np.array([0]), np.array([0]), np.array([0]), np.array([0])

    def model(t, y):
        x, v = y
        
        # 如果速度非常小，检查是否应该停止
        if abs(v) < 1e-6:
            force_at_stop = max(0, -a * x + b)
            resistance_at_stop = calculate_resistance(x)
            if force_at_stop <= resistance_at_stop:
                return [0, 0]  # 速度和加速度都为0
        
        # 计算外力（只取正值）
        force = max(0, -a * x + b)
        
        # 计算阻力
        resistance = calculate_resistance(x)
        
        # 计算净力
        net_force = force - resistance
        
        # 只有在净力为正且速度为正，或净力为负时才应用加速度
        # 这可以防止在速度降为0后，如果净力为负，物体反向运动
        if net_force <= 0 and v <= 0:
            dxdt = 0
            dvdt = 0
        else:
            dxdt = v
            dvdt = net_force / m
        
        return [dxdt, dvdt]

    # 定义停止事件：当速度变为0时
    def event_velocity_zero(t, y):
        return y[1]  # 速度 v
    event_velocity_zero.terminal = True  # 触发时终止积分
    event_velocity_zero.direction = -1  # 从正到负

    # 初始状态
    y0 = [0, 0]  # 初始位移x=0, 初始速度v=0
    t_span = [0, t_max]
    
    sol = solve_ivp(
        model, 
        t_span, 
        y0, 
        dense_output=True, 
        events=event_velocity_zero,
        max_step=0.01
    )

    t = sol.t
    x = sol.y[0]
    v = sol.y[1]
    
    # 如果没有足够的数据点，返回空数组
    if len(t) <= 1:
        return np.array([0]), np.array([0]), np.array([0]), np.array([0]), np.array([0]), np.array([0])

    # 计算各个物理量
    # 计算外力（只取正值）
    force = np.zeros_like(x)
    for i in range(len(x)):
        force[i] = max(0, -a * x[i] + b)
    
    # 计算阻力
    resistance = np.zeros_like(x)
    for i in range(len(x)):
        resistance[i] = calculate_resistance(x[i])
    
    # 计算功
    work_force = np.zeros_like(x)
    work_resistance = np.zeros_like(x)
    
    # 计算每个时间点的U1和U2
    for i in range(len(x)):
        work_force[i] = calculate_U1(x[i])
        work_resistance[i] = calculate_U2(x[i])

    # 计算动能
    kinetic_energy = 0.5 * m * v**2

    return t, x, v, work_force, work_resistance, kinetic_energy

def plot_results(x_axis_data, work_force, work_resistance, kinetic_energy, xlabel, find_max_ke=True):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(x_axis_data, work_force, label='外力功')
    ax.plot(x_axis_data, work_resistance, label='阻力功')
    ax.plot(x_axis_data, kinetic_energy, label='动能')

    # 如果横坐标是时间，我们需要找到物体通过特定位置的时间点
    if xlabel.startswith("时间"):
        # 确保我们有对应的位移数据
        if 'x' in globals() and len(x_axis_data) == len(x):
            # 找到物体通过A和B位置的时间点
            idx_A = np.where(x >= A)[0]
            if len(idx_A) > 0:
                t_A = x_axis_data[idx_A[0]]
                ax.axvline(x=t_A, color='gray', linestyle='--', alpha=0.7)
                ax.text(t_A, ax.get_ylim()[1]*0.95, f't={t_A:.2f}s\n物体通过x=A={A}', 
                        rotation=90, verticalalignment='top')
            
            idx_B = np.where(x >= B)[0]
            if len(idx_B) > 0:
                t_B = x_axis_data[idx_B[0]]
                ax.axvline(x=t_B, color='gray', linestyle='--', alpha=0.7)
                ax.text(t_B, ax.get_ylim()[1]*0.95, f't={t_B:.2f}s\n物体通过x=B={B}', 
                        rotation=90, verticalalignment='top')
            
            # # 找到物体通过外力为零点的时间
            # x_zero_force = b / a if a > 0 else float('inf')
            # idx_zero_force = np.where(x >= x_zero_force)[0]
            # if len(idx_zero_force) > 0:
            #     t_zero_force = x_axis_data[idx_zero_force[0]]
            #     ax.axvline(x=t_zero_force, color='green', linestyle='-.', alpha=0.7)
            #     ax.text(t_zero_force, ax.get_ylim()[1]*0.95, f't={t_zero_force:.2f}s\n物体通过外力为零点\nx={x_zero_force:.2f}', 
            #             rotation=90, verticalalignment='top')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel('能量 / 功 (J)')
    ax.legend()
    ax.grid(True)
    ax.axhline(0, color='gray', linewidth=0.8)

    # 寻找关键点
    # 1. 动能峰值
    if find_max_ke and len(kinetic_energy) > 0 and np.max(kinetic_energy) > 0:
        max_ke_index = np.argmax(kinetic_energy)
        max_ke = kinetic_energy[max_ke_index]
        ax.annotate(f'动能峰值\n({x_axis_data[max_ke_index]:.2f}, {max_ke:.2f})',
                    xy=(x_axis_data[max_ke_index], max_ke),
                    xytext=(x_axis_data[max_ke_index], max_ke + 0.1 * max_ke),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                    )

    # 2. 运动停止点 (动能从正变为0)
    positive_ke_indices = np.where(kinetic_energy > 0)[0]
    if len(positive_ke_indices) > 0:
        stop_index = positive_ke_indices[-1]
        if stop_index is not None and stop_index < len(x_axis_data) -1 :
             stop_x = x_axis_data[stop_index]
             ax.plot(stop_x, 0, 'ro', markersize=8, label=f'停止点 (x={stop_x:.2f})')
             ax.annotate(f'运动停止\n({stop_x:.2f}, 0)',
                        xy=(stop_x, 0),
                        xytext=(stop_x, 10),
                         arrowprops=dict(facecolor='red', shrink=0.05, width=1, headwidth=8),
                        )
    
    st.pyplot(fig)

# 执行时间变化模式逻辑
if b <= Y:
    st.warning("初始外力 F(0) 小于或等于初始静摩擦力 Y，物体无法开始运动。")
else:
    t, x, v, work_force, work_resistance, kinetic_energy = solve_motion_over_time(a, b, Y, C, A, B, m, t_max)
    if len(t) <= 1:
        st.info("物体在给定参数下未能移动或立即停止。")
    else:
        # 绘制能量-时间图
        st.subheader("能量-时间关系")
        plot_results(t, work_force, work_resistance, kinetic_energy, xlabel="时间 t (s)")
        
        # 绘制位移-时间图和速度-时间图
        st.subheader("位移和速度随时间变化")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # 位移-时间图
        ax1.plot(t, x, 'b-', label='位移')
        ax1.set_ylabel('位移 (m)')
        ax1.grid(True)
        ax1.legend()
        
        # 标记A和B位置
        max_x = max(x)
        if A <= max_x:
            ax1.axhline(y=A, color='gray', linestyle='--', alpha=0.7)
            ax1.text(0, A, f'A={A}', verticalalignment='center')
        
        if B <= max_x:
            ax1.axhline(y=B, color='gray', linestyle='--', alpha=0.7)
            ax1.text(0, B, f'B={B}', verticalalignment='center')
        
        # 计算外力为零的位置
        x_zero_force = b / a if a > 0 else float('inf')
        if x_zero_force <= max_x:
            ax1.axhline(y=x_zero_force, color='green', linestyle='-.', alpha=0.7)
            ax1.text(0, x_zero_force, f'外力为零\nx={x_zero_force:.2f}', verticalalignment='center')
        
        # 速度-时间图
        ax2.plot(t, v, 'r-', label='速度')
        ax2.set_xlabel('时间 (s)')
        ax2.set_ylabel('速度 (m/s)')
        ax2.grid(True)
        ax2.legend()
        
        # 标记速度为零的点
        zero_v_indices = np.where(np.abs(v) < 1e-6)[0]
        if len(zero_v_indices) > 0:
            for idx in zero_v_indices:
                if idx > 0:  # 排除初始点
                    ax2.plot(t[idx], v[idx], 'ko', markersize=6)
                    ax2.annotate(f'v=0, t={t[idx]:.2f}',
                                xy=(t[idx], v[idx]),
                                xytext=(t[idx], v[idx] - 0.2),
                                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                                )
                    break  # 只标记第一个速度为零的点
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.subheader("运动详情")
        col1, col2, col3 = st.columns(3)
        
        max_ke_idx = np.argmax(kinetic_energy)
        stop_idx = -1 # 最后一个点
        
        col1.metric("运动总时长", f"{t[stop_idx]:.2f} s")
        col2.metric("最大移动距离", f"{x[stop_idx]:.2f} m")
        col3.metric("最大速度", f"{np.max(v):.2f} m/s", f"在 t={t[np.argmax(v)]:.2f}s 时达到")
        
        # 计算外力为零的位置
        x_zero_force = b / a if a > 0 else float('inf')
        
        # 添加更详细的分析结果
        st.subheader("物理分析")
        
        # 分析动能峰值
        if len(kinetic_energy) > 0 and np.max(kinetic_energy) > 0:
            max_ke_idx = np.argmax(kinetic_energy)
            max_ke_t = t[max_ke_idx]
            max_ke_x = x[max_ke_idx]
            max_ke = kinetic_energy[max_ke_idx]
            
            st.write(f"**动能峰值**：{max_ke:.4f} J，出现在 t = {max_ke_t:.4f} s, x = {max_ke_x:.4f} m")
            
            # 判断峰值点所在区域
            if max_ke_x <= A:
                st.write(f"该点位于第一阶段阻力区域 (0 ≤ x ≤ {A})")
            elif max_ke_x <= B:
                st.write(f"该点位于第二阶段阻力区域 ({A} < x ≤ {B})")
            else:
                st.write(f"该点位于第三阶段阻力区域 (x > {B})")
            
            # 在该点的外力和阻力
            force_at_peak = max(0, -a * max_ke_x + b)
            resistance_at_peak = calculate_resistance(max_ke_x)
            st.write(f"该点的外力 = {force_at_peak:.4f} N，阻力 = {resistance_at_peak:.4f} N")
        
        # 分析运动停止点
        if len(x) > 1:
            stop_x = x[-1]
            stop_t = t[-1]
            
            st.write(f"**运动停止点**：x = {stop_x:.4f} m，t = {stop_t:.4f} s")
            
            # 判断停止点所在区域
            if stop_x <= A:
                st.write(f"物体停止在第一阶段阻力区域 (0 ≤ x ≤ {A})")
            elif stop_x <= B:
                st.write(f"物体停止在第二阶段阻力区域 ({A} < x ≤ {B})")
            else:
                st.write(f"物体停止在第三阶段阻力区域 (x > {B})")
            
            # 在该点的外力和阻力
            force_at_stop = max(0, -a * stop_x + b)
            resistance_at_stop = calculate_resistance(stop_x)
            st.write(f"停止点的外力 = {force_at_stop:.4f} N，阻力 = {resistance_at_stop:.4f} N")
            
            # 检查外力与阻力的关系
            if force_at_stop > resistance_at_stop:
                st.warning("注意：在停止点外力大于阻力，物体理论上会继续运动。这可能是数值计算误差导致的。")
            else:
                st.success("在停止点外力不大于阻力，物体将保持静止状态。")
        
        # 分析物体经过的区域
        if len(x) > 1:
            max_x = np.max(x)
            st.write("**物体经过的区域**：")
            
            if max_x <= A:
                st.write(f"物体仅在第一阶段阻力区域运动 (0 ≤ x ≤ {A})")
            elif max_x <= B:
                st.write(f"物体经过第一阶段和第二阶段阻力区域 (0 ≤ x ≤ {B})")
            else:
                st.write(f"物体经过所有三个阻力区域 (x > {B})")
            
            # 检查是否超过外力为零点
            if x_zero_force < max_x and a > 0:
                st.write(f"物体在惯性作用下超过了外力为零点 (x = {x_zero_force:.4f} m)")
        
        st.subheader("随时间变化的详细数据")
        df = pd.DataFrame({
            '时间 (s)': t,
            '位移 (m)': x,
            '速度 (m/s)': v,
            '动能 (J)': kinetic_energy,
            '外力功 (J)': work_force,
            '阻力功 (J)': work_resistance
        })
        st.dataframe(df)
