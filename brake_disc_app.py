import cv2
import numpy as np
from tkinter import Tk, Button, Label, Frame, filedialog, messagebox, Scrollbar, Text
from PIL import Image, ImageTk
import os

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 创建结果保存目录
results_dir = os.path.join(current_dir, 'results')
os.makedirs(results_dir, exist_ok=True)

class BrakeDiscAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("刹车盘圆孔分析工具")
        self.root.geometry("1100x750")
        self.root.configure(bg="#f0f0f0")
        
        # 初始化变量
        self.image_path = None
        self.original_image = None
        self.result_image = None
        self.analysis_results = []
        self.reference_circle = None
        
        # 创建界面
        self.create_widgets()
    
    def create_widgets(self):
        # 顶部标题区域
        title_frame = Frame(self.root, bg="#2c3e50", height=60)
        title_frame.pack(fill="x")
        title_label = Label(title_frame, text="刹车盘圆孔分析系统", font=("Arial", 18, "bold"), fg="white", bg="#2c3e50")
        title_label.pack(pady=15)
        
        # 顶部按钮区域
        button_frame = Frame(self.root, bg="#f0f0f0")
        button_frame.pack(pady=15)
        
        # 选择图像按钮
        self.select_button = Button(button_frame, text="选择图像", command=self.select_image, width=15, font=("Arial", 10), bg="#3498db", fg="white", relief="flat", padx=10, pady=5)
        self.select_button.pack(side="left", padx=10)
        
        # 分析按钮
        self.analyze_button = Button(button_frame, text="分析圆孔", command=self.analyze_image, width=15, font=("Arial", 10), bg="#27ae60", fg="white", relief="flat", padx=10, pady=5, state="disabled")
        self.analyze_button.pack(side="left", padx=10)
        
        # 保存结果按钮
        self.save_button = Button(button_frame, text="保存结果", command=self.save_result, width=15, font=("Arial", 10), bg="#f39c12", fg="white", relief="flat", padx=10, pady=5, state="disabled")
        self.save_button.pack(side="left", padx=10)
        
        # 图像显示区域
        image_frame = Frame(self.root, bg="#f0f0f0")
        image_frame.pack(pady=10)
        
        # 原始图像
        original_frame = Frame(image_frame, bg="white", relief="raised", borderwidth=2)
        original_frame.pack(side="left", padx=10, pady=5)
        
        self.original_label = Label(original_frame, text="原始图像", font=("Arial", 10, "bold"), bg="white")
        self.original_label.pack(pady=5)
        
        self.original_image_label = Label(original_frame, width=400, height=400, bg="#e0e0e0")
        self.original_image_label.pack(padx=5, pady=5)
        
        # 结果图像
        result_frame = Frame(image_frame, bg="white", relief="raised", borderwidth=2)
        result_frame.pack(side="left", padx=10, pady=5)
        
        self.result_label = Label(result_frame, text="分析结果", font=("Arial", 10, "bold"), bg="white")
        self.result_label.pack(pady=5)
        
        self.result_image_label = Label(result_frame, width=400, height=400, bg="#e0e0e0")
        self.result_image_label.pack(padx=5, pady=5)
        
        # 分析结果区域
        result_text_frame = Frame(self.root, bg="#f0f0f0")
        result_text_frame.pack(pady=10, fill="x", padx=20)
        
        self.result_text_label = Label(result_text_frame, text="分析结果:", font=("Arial", 10, "bold"), bg="#f0f0f0")
        self.result_text_label.pack(anchor="w")
        
        # 添加滚动条的文本框
        text_frame = Frame(result_text_frame, bg="white", relief="sunken", borderwidth=1)
        text_frame.pack(fill="both", expand=True, pady=5)
        
        scrollbar = Scrollbar(text_frame)
        scrollbar.pack(side="right", fill="y")
        
        self.result_text = Text(text_frame, wrap="word", yscrollcommand=scrollbar.set, height=8, font=("Arial", 9))
        self.result_text.pack(fill="both", expand=True)
        scrollbar.config(command=self.result_text.yview)
        
        # 底部状态区域
        status_frame = Frame(self.root, bg="#2c3e50", height=30)
        status_frame.pack(fill="x", side="bottom")
        
        self.status_label = Label(status_frame, text="就绪", font=("Arial", 9), fg="white", bg="#2c3e50")
        self.status_label.pack(anchor="w", padx=10, pady=5)
    
    def select_image(self):
        # 选择图像文件
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        
        if file_path:
            self.image_path = file_path
            # 加载并显示图像
            self.load_and_display_image()
            # 启用分析按钮
            self.analyze_button.config(state="normal")
            # 禁用保存按钮
            self.save_button.config(state="disabled")
            # 更新状态
            self.status_label.config(text="已选择图像: " + os.path.basename(file_path))
    
    def load_and_display_image(self):
        # 加载图像
        self.original_image = cv2.imread(self.image_path)
        
        # 调整图像大小以适应显示
        height, width = self.original_image.shape[:2]
        max_size = 400
        if height > max_size or width > max_size:
            scale = max_size / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            self.original_image = cv2.resize(self.original_image, (new_width, new_height))
        
        # 转换为PIL图像并显示
        img_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        tk_img = ImageTk.PhotoImage(image=pil_img)
        
        self.original_image_label.config(image=tk_img)
        self.original_image_label.image = tk_img
    
    def preprocess_image(self, img):
        # 高斯滤波去噪
        blurred = cv2.GaussianBlur(img, (3, 3), 0)
        # 自适应二值化 - 调整参数以提高检测灵敏度
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 13, 2)
        # 形态学操作 - 减小核大小，保留更多细节
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        return binary
    
    def detect_circles(self, img, binary):
        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        circles = []
        height, width = img.shape
        margin = 10  # 进一步减少边缘边距，从15像素减少到10像素
        
        for contour in contours:
            # 计算轮廓面积
            area = cv2.contourArea(contour)
            # 计算轮廓周长
            perimeter = cv2.arcLength(contour, True)
            # 跳过太小的轮廓
            if area < 8:  # 进一步降低最小面积阈值，从10减少到8
                continue
            # 计算圆形度
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
                # 只保留圆形度较高的轮廓
                if circularity > 0.55:  # 进一步降低圆形度阈值，从0.6减少到0.55
                    # 拟合椭圆
                    if len(contour) >= 5:
                        ellipse = cv2.fitEllipse(contour)
                        center, axes, angle = ellipse
                        # 计算半径
                        radius = max(axes) / 2
                        
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
    
    def analyze_image(self):
        if not self.image_path:
            messagebox.showerror("错误", "请先选择图像")
            return
        
        try:
            # 更新状态
            self.status_label.config(text="正在分析图像...")
            self.root.update()
            
            # 加载灰度图像
            gray_img = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
            
            # 预处理图像
            binary = self.preprocess_image(gray_img)
            
            # 检测圆孔
            circles = self.detect_circles(gray_img, binary)
            
            # 区分大圆孔和小圆孔
            large_circles, small_circles = self.separate_circles(circles, gray_img.shape)
            
            # 计算距离和大小
            distances, reference_circle = self.calculate_distances(large_circles, small_circles, gray_img.shape)
            self.analysis_results = distances
            self.reference_circle = reference_circle
            
            # 可视化结果
            result_img = self.visualize_results(gray_img, large_circles, small_circles, distances, reference_circle)
            
            # 调整结果图像大小
            height, width = result_img.shape[:2]
            max_size = 400
            if height > max_size or width > max_size:
                scale = max_size / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                result_img = cv2.resize(result_img, (new_width, new_height))
            
            self.result_image = result_img
            
            # 显示结果图像
            img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            tk_img = ImageTk.PhotoImage(image=pil_img)
            
            self.result_image_label.config(image=tk_img)
            self.result_image_label.image = tk_img
            
            # 显示分析结果
            self.display_analysis_results(large_circles, small_circles, distances, reference_circle)
            
            # 启用保存按钮
            self.save_button.config(state="normal")
            
            # 更新状态
            self.status_label.config(text="分析完成")
            
        except Exception as e:
            messagebox.showerror("错误", f"分析失败: {str(e)}")
            self.status_label.config(text="分析失败")
    
    def display_analysis_results(self, large_circles, small_circles, distances, reference_circle):
        # 清空文本框
        self.result_text.delete(1.0, "end")
        
        if not reference_circle:
            self.result_text.insert("end", "未检测到参考孔")
            return
        
        ref_center = reference_circle[0]
        ref_radius = reference_circle[1]
        
        result_text = f"参考孔: 中心=({ref_center[0]:.1f}, {ref_center[1]:.1f}), 半径={ref_radius:.1f}\n"
        result_text += f"检测到 {len(large_circles)} 个大圆孔和 {len(small_circles)} 个小圆孔\n\n"
        
        if distances:
            result_text += "小圆孔分析结果:\n"
            for result in distances[:15]:  # 显示前15个结果
                result_text += f"  圆孔 #{result['index']}: 中心=({result['center'][0]:.1f}, {result['center'][1]:.1f}), "
                result_text += f"半径={result['radius']:.1f}, 距离参考孔={result['distance_from_reference']:.1f}, "
                result_text += f"相对大小={result['relative_size']:.2f}\n"
            
            if len(distances) > 15:
                result_text += f"... 还有 {len(distances) - 15} 个小圆孔未显示"
        else:
            result_text += "未检测到小圆孔"
        
        self.result_text.insert("end", result_text)
    
    def save_result(self):
        if not self.result_image:
            messagebox.showerror("错误", "请先分析图像")
            return
        
        # 保存结果图像
        save_path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*")]
        )
        
        if save_path:
            try:
                cv2.imwrite(save_path, self.result_image)
                messagebox.showinfo("成功", f"结果已保存到: {save_path}")
                self.status_label.config(text="结果已保存")
            except Exception as e:
                messagebox.showerror("错误", f"保存失败: {str(e)}")
                self.status_label.config(text="保存失败")

if __name__ == "__main__":
    root = Tk()
    app = BrakeDiscAnalyzer(root)
    root.mainloop()
