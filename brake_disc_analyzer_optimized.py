import cv2
import numpy as np
import os

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 创建结果保存目录
results_dir = os.path.join(current_dir, 'results')
os.makedirs(results_dir, exist_ok=True)

class BrakeDiscAnalyzer:
    def __init__(self):
        pass
    
    def preprocess_image(self, img):
        # 高斯滤波去噪
        blurred = cv2.GaussianBlur(img, (3, 3), 0)
        # 自适应二值化 - 调整参数以提高圆孔与背景的对比度
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
        # 形态学操作 - 先膨胀后腐蚀，增强圆孔区域
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_ERODE, kernel)
        return binary
    
    def detect_circles(self, img, binary):
        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        circles = []
        height, width = img.shape
        margin = 10  # 进一步减少边缘边距，确保检测到靠近边缘的圆孔
        
        for contour in contours:
            # 计算轮廓面积
            area = cv2.contourArea(contour)
            # 计算轮廓周长
            perimeter = cv2.arcLength(contour, True)
            # 跳过太小或太大的轮廓
            if area < 5 or area > 800:  # 进一步调整面积范围，确保检测到所有真正的圆孔
                continue
            # 计算圆形度
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
                # 只保留圆形度较高的轮廓
                if circularity > 0.55:  # 进一步降低圆形度阈值，以检测到更多接近圆形的物体
                    # 拟合椭圆
                    if len(contour) >= 5:
                        ellipse = cv2.fitEllipse(contour)
                        center, axes, angle = ellipse
                        # 计算半径
                        radius = max(axes) / 2
                        # 计算椭圆的长宽比，确保是接近圆形的
                        aspect_ratio = min(axes) / max(axes)
                        if aspect_ratio > 0.7:  # 进一步降低长宽比阈值，以检测到更多接近圆形的物体
                            # 过滤边缘的轮廓
                            if (center[0] > margin and center[0] < width - margin and
                                center[1] > margin and center[1] < height - margin):
                                circles.append((center, radius, contour))
        
        return circles
    
    def separate_circles(self, circles, image_shape):
        if not circles:
            return [], []
        
        large_circles = []
        small_circles = []
        
        # 选择整个图像中最大的圆作为大孔
        circles_sorted = sorted(circles, key=lambda x: x[1], reverse=True)
        
        # 确保有足够大的圆作为参考孔
        if circles_sorted:
            # 选择最大的圆作为大孔
            large_circles.append(circles_sorted[0])
            
            # 其他圆作为小孔
            for circle in circles:
                if circle != large_circles[0]:
                    # 确保小孔有一定大小
                    if circle[1] > 2.0:  # 最小半径阈值
                        small_circles.append(circle)
        
        return large_circles, small_circles
    
    def calculate_distances(self, large_circles, small_circles, image_shape):
        results = []
        reference_circle = None
        
        if not large_circles or not small_circles:
            return results, reference_circle
        
        # 使用大圆孔（最大的圆）作为参考孔
        reference_circle = large_circles[0]
        ref_center = reference_circle[0]
        ref_radius = reference_circle[1]
        
        for i, small_circle in enumerate(small_circles):
            small_center = small_circle[0]
            small_radius = small_circle[1]
            
            # 计算距离
            distance = np.sqrt((small_center[0] - ref_center[0]) ** 2 + (small_center[1] - ref_center[1]) ** 2)
            
            # 计算相对大小（相对于参考孔）
            relative_size = small_radius / ref_radius
            
            results.append({
                'index': i + 1,
                'center': small_center,
                'radius': small_radius,
                'distance_from_reference': distance,
                'relative_size': relative_size
            })
        
        return results, reference_circle
    
    def visualize_results(self, img, large_circles, small_circles, distances, reference_circle):
        # 创建彩色图像用于绘制
        color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # 合并所有圆孔并统一编号
        all_circles = []
        if large_circles:
            all_circles.extend(large_circles)
        if small_circles:
            all_circles.extend(small_circles)
        
        # 绘制所有圆孔
        for i, circle in enumerate(all_circles):
            center, radius, _ = circle
            
            # 绘制圆圈
            if circle == reference_circle:
                # 参考孔用绿色标记
                cv2.circle(color_img, (int(center[0]), int(center[1])), int(radius), (0, 255, 0), 2)
            else:
                # 其他孔用红色标记
                cv2.circle(color_img, (int(center[0]), int(center[1])), int(radius), (0, 0, 255), 2)
            
            # 绘制十字标记
            cross_size = 5
            cv2.line(color_img, (int(center[0]) - cross_size, int(center[1])), 
                     (int(center[0]) + cross_size, int(center[1])), (0, 255, 0), 2)
            cv2.line(color_img, (int(center[0]), int(center[1]) - cross_size), 
                     (int(center[0]), int(center[1]) + cross_size), (0, 255, 0), 2)
            
            # 标注编号和坐标
            text = f"# {i+1} ({int(center[0])}, {int(center[1])})"
            cv2.putText(color_img, text, (int(center[0]) + 10, int(center[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return color_img
    
    def analyze(self, image_path):
        # 加载灰度图像
        gray_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if gray_img is None:
            print(f"无法读取图像: {image_path}")
            return
        
        # 预处理图像
        binary = self.preprocess_image(gray_img)
        
        # 检测圆孔
        circles = self.detect_circles(gray_img, binary)
        
        # 区分大圆孔和小圆孔
        large_circles, small_circles = self.separate_circles(circles, gray_img.shape)
        
        # 计算距离和大小
        distances, reference_circle = self.calculate_distances(large_circles, small_circles, gray_img.shape)
        
        # 可视化结果
        result_img = self.visualize_results(gray_img, large_circles, small_circles, distances, reference_circle)
        
        # 保存结果
        image_name = os.path.basename(image_path)
        result_path = os.path.join(results_dir, f"{os.path.splitext(image_name)[0]}_optimized_result.jpg")
        cv2.imwrite(result_path, result_img)
        
        # 输出分析结果
        print("===== 刹车盘圆孔分析结果（优化版）=====")
        if reference_circle:
            ref_center = reference_circle[0]
            ref_radius = reference_circle[1]
            print(f"参考孔: 中心=({ref_center[0]:.1f}, {ref_center[1]:.1f}), 半径={ref_radius:.1f}")
            print(f"检测到 {len(large_circles)} 个大圆孔和 {len(small_circles)} 个小圆孔")
            
            if distances:
                print("\n小圆孔分析结果:")
                for result in distances[:10]:  # 显示前10个结果
                    print(f"  圆孔 #{result['index']}: 中心=({result['center'][0]:.1f}, {result['center'][1]:.1f}), "
                          f"半径={result['radius']:.1f}, 距离参考孔={result['distance_from_reference']:.1f}, "
                          f"相对大小={result['relative_size']:.2f}")
                
                if len(distances) > 10:
                    print(f"... 还有 {len(distances) - 10} 个小圆孔未显示")
        else:
            print("未检测到参考孔")
        
        print(f"\n分析结果已保存到: {result_path}")

if __name__ == "__main__":
    # 分析当前目录下的2.png图像
    image_path = os.path.join(current_dir, "2.png")
    if os.path.exists(image_path):
        analyzer = BrakeDiscAnalyzer()
        analyzer.analyze(image_path)
    else:
        print(f"图像文件不存在: {image_path}")
        print("请确保2.png文件在当前目录中")
