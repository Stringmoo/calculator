"""
计算工具模块，包含物理模型计算的各种函数
"""
def calculate_resistance(x, Y, C, A, B):
    """计算阻力"""
    if x <= A:
        # 第一阶段 (0 ≤ x ≤ A): f = Y·x
        return Y * x
    elif x <= B:
        # 第二阶段 (A < x ≤ B): f = C·x^(-2/3)
        return C * x**(-2/3)
    else:
        # 第三阶段 (x > B): f = 0
        return 0
    
def calculate_net_force(x, a, b, Y, C, A, B):
    """计算净外力"""
    # 计算外力
    external_force = max(0, -a * x + b)  # 外力只取大于0的部分

    # 计算阻力
    resistance = calculate_resistance(x, Y, C, A, B)

    return external_force - resistance
    
def calculate_U1(x, a, b):
    """
    计算外力做功 U1
    
    参数:
        x: 位移
        a: 外力减小系数
        b: 外力初始值
    
    返回:
        外力做功值
    """
    # 计算外力变为0的位置
    x_zero_force = b / a if a > 0 else float('inf')
    
    if x <= x_zero_force:
        # 外力为正时的做功
        return b * x - 0.5 * a * x**2
    else:
        # 外力为0后，做功不再增加，保持在x_zero_force处的值
        return b * x_zero_force - 0.5 * a * x_zero_force**2

def calculate_U2(x, Y, C, A, B):
    """
    计算阻力做功 U2
    
    参数:
        x: 位移
        Y: 第一阶段阻力系数
        C: 第二阶段阻力系数
        A: 第一阶段终点
        B: 第二阶段终点
    
    返回:
        阻力做功值
    """
    if x <= A:
        # 第一阶段 (0 ≤ x ≤ A): f = Y·x
        return 0.5 * Y * x**2
    elif x <= B:
        # 第二阶段 (A < x ≤ B): f = C·x^(-2/3)
        return 2*Y + 3*C * (x**(1/3) - A**(1/3))
    else:
        # 第三阶段 (x > B): f = 0
        return 2*Y + 3*C * (B**(1/3) - A**(1/3))


