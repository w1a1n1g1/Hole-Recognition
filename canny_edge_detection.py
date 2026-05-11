import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 创建结果保存目录
os.makedirs('results', exist_ok=True)

# 读取图像
def load_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    return img

# Canny边缘检测
def canny_edge_detection(img, blur_kernel=(3, 3), sigma=1, threshold1=50, threshold2=150, use_blur=True):
    # 高斯滤波去噪
    if use_blur:
        blurred = cv2.GaussianBlur(img, blur_kernel, sigma)
    else:
        blurred = img
    # Canny边缘检测
    edges = cv2.Canny(blurred, threshold1, threshold2)
    return edges

# 保存结果
def save_result(original, edges, filename, params):
    # 创建子图
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    # 显示原始图像
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('原始图像')
    axes[0].axis('off')
    # 显示边缘检测结果
    axes[1].imshow(edges, cmap='gray')
    axes[1].set_title(f'Canny边缘检测\n参数: threshold1={params["threshold1"]}, threshold2={params["threshold2"]}\nblur={params["use_blur"]}')
    axes[1].axis('off')
    # 保存图像
    save_path = f'results/{filename}'
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path

# 主函数
def main():
    # 图像路径
    image_paths = [
        '2.png'
    ]
    
    # 阈值组合
    threshold_combinations = [
        {"threshold1": 30, "threshold2": 90},   # 组合1: 3倍
        {"threshold1": 40, "threshold2": 100},  # 组合2: 2.5倍
        {"threshold1": 50, "threshold2": 150},  # 组合3: 3倍
        {"threshold1": 60, "threshold2": 120},  # 组合4: 2倍
        {"threshold1": 20, "threshold2": 80}    # 组合5: 4倍
    ]
    
    # 处理每张图像
    for img_path in image_paths:
        try:
            # 加载图像
            img = load_image(img_path)
            print(f"处理图像: {img_path}")
            
            # 测试不同阈值组合
            for i, params in enumerate(threshold_combinations, 1):
                # 执行Canny边缘检测（使用高斯滤波）
                edges = canny_edge_detection(
                    img,
                    threshold1=params["threshold1"],
                    threshold2=params["threshold2"],
                    use_blur=True
                )
                # 保存结果
                filename = f'{os.path.splitext(img_path)[0]}_combo{i}_blur.jpg'
                save_path = save_result(img, edges, filename, {**params, "use_blur": True})
                print(f"  保存结果: {save_path}")
            
            # 噪声影响实验（关闭高斯滤波）
            best_params = threshold_combinations[2]  # 使用组合3作为最优参数
            edges_no_blur = canny_edge_detection(
                img,
                threshold1=best_params["threshold1"],
                threshold2=best_params["threshold2"],
                use_blur=False
            )
            filename = f'{os.path.splitext(img_path)[0]}_no_blur.jpg'
            save_path = save_result(img, edges_no_blur, filename, {**best_params, "use_blur": False})
            print(f"  保存无滤波结果: {save_path}")
            
        except Exception as e:
            print(f"处理图像 {img_path} 时出错: {e}")

if __name__ == "__main__":
    main()
