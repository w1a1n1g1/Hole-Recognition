# 工业零件孔洞检测 —— 基于 Canny 边缘检测的参数对比实验

对工业零件（刹车盘）图像进行 Canny 边缘检测，通过 5 组双阈值参数组合的对比实验筛选最优检测参数，为孔洞/缺陷自动检测提供可复现的参数选型依据。

## 项目结构

```
zy2/
├── canny_edge_detection.py  # 主实验代码
├── 实验记录表格.md          # 实验记录和分析
├── README.md               # 项目说明
├── brake_disc.jpg          # 实验素材图像
└── results/                # 结果保存目录
    └── ...                 # 边缘检测结果图像
```

## 实验目的

1. 掌握Canny边缘检测算法的核心原理及完整实现流程
2. 熟练使用OpenCV实现工业零件图像的边缘检测
3. 掌握Canny算法双阈值的调整方法，分析阈值对边缘检测效果的影响
4. 学会保存边缘检测结果，撰写规范的实验记录

## 环境要求

- Python 3.8~3.10
- OpenCV 4.5.0及以上版本
- NumPy 1.21.0及以上版本
- Matplotlib 3.4.0及以上版本

## 安装依赖

```bash
pip install opencv-python numpy matplotlib
```

## 实验步骤

1. **准备实验素材**：将工业零件图像（brake_disc.jpg）放置在项目根目录
2. **运行实验代码**：执行canny_edge_detection.py脚本
3. **查看结果**：在results目录中查看不同参数组合的边缘检测结果
4. **填写实验记录**：根据实验结果填写实验记录表格

## 代码说明

- `load_image()`：读取灰度图像
- `canny_edge_detection()`：执行Canny边缘检测
- `save_result()`：保存边缘检测结果
- `main()`：主函数，执行不同参数组合的实验

## 实验参数

- 高斯滤波：3×3核，σ=1
- 双阈值组合：
  - 组合1：threshold1=30, threshold2=90
  - 组合2：threshold1=40, threshold2=100
  - 组合3：threshold1=50, threshold2=150
  - 组合4：threshold1=60, threshold2=120
  - 组合5：threshold1=20, threshold2=80

## 预期结果

- 生成不同参数组合的边缘检测结果图像
- 分析阈值对边缘检测效果的影响
- 验证高斯滤波对去噪的重要性
- 筛选出最优的边缘检测参数
