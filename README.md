# 褶皱分析工具

这是一个基于Streamlit开发的物理受力分析工具，用于分析物体在变化外力和阻力下的运动情况。

## 功能特点

- 分析物体在变化外力和阻力下的运动情况
- 计算并可视化外力做功、阻力做功和动能曲线
- 自动寻找动能峰值点和运动停止点
- 提供详细的运动分析结果

## 参数说明

- **a**: 外力减小系数，控制外力随位移减小的速率
- **b**: 外力初始值，初始外力大小
- **Y**: 第一阶段阻力系数，控制第一阶段阻力大小
- **C**: 第二阶段阻力系数，控制第二阶段阻力大小
- **A**: 第一阶段终点，标记第一阶段结束位置
- **B**: 第二阶段终点，标记第二阶段结束位置

## 公式说明

### 外力公式
- F = -ax + b (x ≤ b/a)
- F = 0 (x > b/a)

### 阻力公式
- 第一阶段 (0 ≤ x ≤ A): f = Y·x
- 第二阶段 (A < x ≤ B): f = C·x^(-2/3)
- 第三阶段 (x > B): f = 0

## 本地运行

1. 克隆此仓库
2. 安装依赖：`pip install -r requirements.txt`
3. 运行应用：`streamlit run app.py`

## 在线部署

可以使用Streamlit Cloud免费部署此应用：

1. 在[Streamlit Cloud](https://streamlit.io/cloud)上创建账号
2. 连接你的GitHub仓库
3. 选择此项目并部署

## 中文字体支持

应用已针对不同操作系统环境进行了中文字体支持优化：

- Windows: 使用SimHei、Microsoft YaHei字体
- macOS: 使用Arial Unicode MS、PingFang SC字体
- Linux: 自动检测并使用可用的中文字体

## 依赖项

- streamlit>=1.22.0
- numpy>=1.20.0
- matplotlib>=3.5.0
- scipy>=1.7.0
- pandas>=1.3.0

