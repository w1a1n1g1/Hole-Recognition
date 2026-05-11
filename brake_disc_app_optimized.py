import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
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

class BrakeDiscApp:
    def __init__(self, root):
        self.root = root
        self.root.title("刹车盘圆孔分析系统")
        self.root.geometry("1000x700")
        self.root.configure(bg="#f0f0f0")
        
        # 初始化分析器
        self.analyzer = BrakeDiscAnalyzer()
        
        # 存储当前图像路径
        self.image_path = None
        self.gray_img = None
        self.results = None
        self.reference_circle = None
        
        # 创建界面元素
        self.create_widgets()
    
    def create_widgets(self):
        # 标题栏
        title_frame = tk.Frame(self.root, bg="#2c3e50", height=60)
        title_frame.pack(fill=tk.X, padx=10, pady=10)
        
        title_label = tk.Label(title_frame, text="刹车盘圆孔分析系统", font=("Arial", 20, "bold"), fg="white", bg="#2c3e50")
        title_label.pack(pady=10)
        
        # 主框架
        main_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧图像显示区域
        left_frame = tk.Frame(main_frame, bg="#f0f0f0")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        
        # 原始图像标题
        original_title = tk.Label(left_frame, text="原始图像", font=("Arial", 12, "bold"), bg="#f0f0f0")
        original_title.pack(pady=5)
        
        # 原始图像显示
        self.original_canvas = tk.Canvas(left_frame, bg="white", relief=tk.RAISED, bd=2)
        self.original_canvas.pack(fill=tk.BOTH, expand=True)
        
        # 右侧结果显示区域
        right_frame = tk.Frame(main_frame, bg="#f0f0f0")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)
        
        # 结果图像标题
        result_title = tk.Label(right_frame, text="分析结果", font=("Arial", 12, "bold"), bg="#f0f0f0")
        result_title.pack(pady=5)
        
        # 结果图像显示
        self.result_canvas = tk.Canvas(right_frame, bg="white", relief=tk.RAISED, bd=2)
        self.result_canvas.pack(fill=tk.BOTH, expand=True)
        
        # 底部按钮和结果区域
        bottom_frame = tk.Frame(self.root, bg="#f0f0f0")
        bottom_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # 按钮框架
        button_frame = tk.Frame(bottom_frame, bg="#f0f0f0")
        button_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        # 选择图像按钮
        self.select_button = tk.Button(button_frame, text="选择图像", command=self.select_image, 
                                     width=15, height=2, bg="#3498db", fg="white", 
                                     font=("Arial", 10, "bold"), relief=tk.FLAT)
        self.select_button.pack(side=tk.LEFT, padx=10)
        
        # 分析按钮
        self.analyze_button = tk.Button(button_frame, text="分析圆孔", command=self.analyze_image, 
                                      width=15, height=2, bg="#27ae60", fg="white", 
                                      font=("Arial", 10, "bold"), relief=tk.FLAT)
        self.analyze_button.pack(side=tk.LEFT, padx=10)
        
        # 保存结果按钮
        self.save_button = tk.Button(button_frame, text="保存结果", command=self.save_result, 
                                   width=15, height=2, bg="#e67e22", fg="white", 
                                   font=("Arial", 10, "bold"), relief=tk.FLAT)
        self.save_button.pack(side=tk.LEFT, padx=10)
        
        # 结果显示区域
        result_frame = tk.Frame(bottom_frame, bg="#f0f0f0")
        result_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=5)
        
        # 结果文本框
        result_label = tk.Label(result_frame, text="分析结果", font=("Arial", 12, "bold"), bg="#f0f0f0")
        result_label.pack(pady=5)
        
        self.result_text = tk.Text(result_frame, height=10, width=80, font=("Arial", 10), 
                                 bg="#f8f9fa", relief=tk.SUNKEN, bd=2)
        self.result_text.pack(fill=tk.BOTH, expand=True)
        
        # 滚动条
        scrollbar = tk.Scrollbar(self.result_text, command=self.result_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_text.config(yscrollcommand=scrollbar.set)
        
        # 状态栏
        status_frame = tk.Frame(self.root, bg="#34495e", height=30)
        status_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.status_label = tk.Label(status_frame, text="就绪", font=("Arial", 10), fg="white", bg="#34495e")
        self.status_label.pack(pady=5, padx=10, anchor=tk.W)
    
    def select_image(self):
        # 打开文件对话框选择图像
        file_path = filedialog.askopenfilename(
            filetypes=[("图像文件", "*.png;*.jpg;*.jpeg;*.bmp")]
        )
        
        if file_path:
            self.image_path = file_path
            # 加载灰度图像
            self.gray_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if self.gray_img is not None:
                # 显示原始图像
                self.display_image(self.original_canvas, self.gray_img)
                self.status_label.config(text=f"已加载图像: {os.path.basename(file_path)}")
                # 清空结果
                self.result_text.delete(1.0, tk.END)
                self.result_canvas.delete("all")
            else:
                messagebox.showerror("错误", "无法读取图像文件")
    
    def display_image(self, canvas, img):
        # 计算图像显示尺寸
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        if canvas_width == 1 or canvas_height == 1:
            # 等待窗口完全加载
            self.root.update_idletasks()
            canvas_width = canvas.winfo_width()
            canvas_height = canvas.winfo_height()
        
        # 调整图像大小
        h, w = img.shape[:2]
        aspect_ratio = w / h
        
        if canvas_width / canvas_height > aspect_ratio:
            new_height = canvas_height
            new_width = int(new_height * aspect_ratio)
        else:
            new_width = canvas_width
            new_height = int(new_width / aspect_ratio)
        
        resized_img = cv2.resize(img, (new_width, new_height))
        
        # 转换为PIL图像
        if len(resized_img.shape) == 2:
            pil_img = Image.fromarray(resized_img, mode="L")
        else:
            pil_img = Image.fromarray(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB))
        
        # 转换为Tkinter图像
        tk_img = ImageTk.PhotoImage(image=pil_img)
        
        # 显示图像
        canvas.delete("all")
        canvas.create_image((canvas_width - new_width) // 2, (canvas_height - new_height) // 2, 
                          anchor=tk.NW, image=tk_img)
        
        # 保存图像引用，避免被垃圾回收
        canvas.image = tk_img
    
    def analyze_image(self):
        if self.gray_img is None:
            messagebox.showinfo("提示", "请先选择图像")
            return
        
        self.status_label.config(text="分析中...")
        self.root.update()
        
        try:
            # 预处理图像
            binary = self.analyzer.preprocess_image(self.gray_img)
            
            # 检测圆孔
            circles = self.analyzer.detect_circles(self.gray_img, binary)
            
            # 区分大圆孔和小圆孔
            large_circles, small_circles = self.analyzer.separate_circles(circles, self.gray_img.shape)
            
            # 计算距离和大小
            self.results, self.reference_circle = self.analyzer.calculate_distances(large_circles, small_circles, self.gray_img.shape)
            
            # 可视化结果
            result_img = self.analyzer.visualize_results(self.gray_img, large_circles, small_circles, self.results, self.reference_circle)
            
            # 显示结果图像
            self.display_image(self.result_canvas, result_img)
            
            # 显示分析结果
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "===== 刹车盘圆孔分析结果 =====\n")
            
            if self.reference_circle:
                ref_center = self.reference_circle[0]
                ref_radius = self.reference_circle[1]
                self.result_text.insert(tk.END, f"参考孔: 中心=({ref_center[0]:.1f}, {ref_center[1]:.1f}), 半径={ref_radius:.1f}\n")
                self.result_text.insert(tk.END, f"检测到 {len(large_circles)} 个大圆孔和 {len(small_circles)} 个小圆孔\n\n")
                
                if self.results:
                    self.result_text.insert(tk.END, "小圆孔分析结果:\n")
                    for result in self.results[:15]:  # 显示前15个结果
                        self.result_text.insert(tk.END, f"  圆孔 #{result['index']}: 中心=({result['center'][0]:.1f}, {result['center'][1]:.1f}), "
                                              f"半径={result['radius']:.1f}, 距离参考孔={result['distance_from_reference']:.1f}, "
                                              f"相对大小={result['relative_size']:.2f}\n")
                    
                    if len(self.results) > 15:
                        self.result_text.insert(tk.END, f"... 还有 {len(self.results) - 15} 个小圆孔未显示\n")
            else:
                self.result_text.insert(tk.END, "未检测到参考孔\n")
            
            self.status_label.config(text="分析完成")
            
        except Exception as e:
            messagebox.showerror("错误", f"分析过程中出错: {str(e)}")
            self.status_label.config(text="分析失败")
    
    def save_result(self):
        if self.gray_img is None or self.reference_circle is None:
            messagebox.showinfo("提示", "请先分析图像")
            return
        
        try:
            # 检查results目录是否存在
            if not os.path.exists(results_dir):
                os.makedirs(results_dir, exist_ok=True)
                self.status_label.config(text="已创建results目录")
            
            # 检查image_path是否存在
            if not hasattr(self, 'image_path') or self.image_path is None:
                messagebox.showerror("错误", "图像路径不存在")
                return
            
            # 重新生成结果图像
            binary = self.analyzer.preprocess_image(self.gray_img)
            circles = self.analyzer.detect_circles(self.gray_img, binary)
            large_circles, small_circles = self.analyzer.separate_circles(circles, self.gray_img.shape)
            result_img = self.analyzer.visualize_results(self.gray_img, large_circles, small_circles, self.results, self.reference_circle)
            
            # 检查result_img是否生成成功
            if result_img is None:
                messagebox.showerror("错误", "无法生成结果图像")
                return
            
            # 保存结果
            try:
                image_name = os.path.basename(self.image_path)
                result_path = os.path.join(results_dir, f"{os.path.splitext(image_name)[0]}_result.jpg")
                
                # 尝试保存
                success = cv2.imwrite(result_path, result_img)
                if success:
                    messagebox.showinfo("成功", f"分析结果已保存到: {result_path}")
                    self.status_label.config(text=f"结果已保存: {os.path.basename(result_path)}")
                else:
                    messagebox.showerror("错误", f"保存失败: 无法写入文件到 {result_path}")
                    self.status_label.config(text="保存失败: 无法写入文件")
            except Exception as save_error:
                messagebox.showerror("错误", f"保存路径处理出错: {str(save_error)}")
                self.status_label.config(text="保存失败: 路径错误")
            
        except Exception as e:
            messagebox.showerror("错误", f"保存过程中出错: {str(e)}")
            self.status_label.config(text="保存失败")

if __name__ == "__main__":
    root = tk.Tk()
    app = BrakeDiscApp(root)
    root.mainloop()
