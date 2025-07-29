import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os
import glob
import numpy as np

class ImageManagerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("图片和NPY文件管理器")
        self.root.geometry("800x600")
        self.root.resizable(True, True)

        # 设置中文字体
        self.style = ttk.Style()
        self.style.configure('.', font=('SimHei', 10))

        # 存储图片路径和当前索引
        self.image_paths = []
        self.current_index = -1
        self.image_dir = ''
        self.channel_dirs = []  # 存储4个通道文件夹路径
        self.grouped_images = {}  # {文件名: [通道1路径, 通道2路径, 通道3路径, 通道4路径]}
        self.image_names = []  # 所有共有图片文件名列表

        # 创建界面
        self.create_widgets()

    def create_widgets(self):
        # 创建工具栏
        self.toolbar = ttk.Frame(self.root, padding=5)
        self.toolbar.pack(side=tk.TOP, fill=tk.X)

        # 选择文件夹按钮
        ttk.Button(self.toolbar, text="选择图片文件夹", command=self.select_directory).pack(side=tk.LEFT, padx=5)

        # 删除按钮
        self.delete_btn = ttk.Button(self.toolbar, text="删除当前图片及NPY", command=self.delete_current_files, state=tk.DISABLED)
        self.delete_btn.pack(side=tk.LEFT, padx=5)

        # 上一张按钮
        self.prev_btn = ttk.Button(self.toolbar, text="上一张图片", command=self.previous_image, state=tk.DISABLED)
        self.prev_btn.pack(side=tk.LEFT, padx=5)

        # 下一张按钮
        self.next_btn = ttk.Button(self.toolbar, text="下一张图片", command=self.next_image, state=tk.DISABLED)
        self.next_btn.pack(side=tk.LEFT, padx=5)

        # 状态栏
        self.status_var = tk.StringVar()
        self.status_var.set("请选择包含PNG图片的文件夹")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # 图片显示区域
        self.image_frame = ttk.Frame(self.root)
        self.image_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.image_label = ttk.Label(self.image_frame, text="请选择图片文件夹以开始浏览", anchor=tk.CENTER)
        self.image_label.pack(fill=tk.BOTH, expand=True)

    def select_directory(self):
        parent_dir = filedialog.askdirectory(title="选择包含4个通道文件夹的父目录")
        if parent_dir:
            # 获取父目录下的所有子目录作为通道文件夹
            self.channel_dirs = [os.path.join(parent_dir, d) for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]
            
            # 确保有4个通道文件夹
            if len(self.channel_dirs) != 4:
                messagebox.showwarning("警告", f"所选目录下找到{len(self.channel_dirs)}个子目录，需要 exactly 4个通道文件夹")
                return
            
            self.load_grouped_images()
            if self.grouped_images:
                self.image_names = list(self.grouped_images.keys())
                self.current_index = 0
                self.update_ui_state()
                self.display_current_images()
            else:
                messagebox.showinfo("提示", "所选通道文件夹中未找到共有的PNG图片")
                self.status_var.set("所选通道文件夹中未找到共有的PNG图片")

    def load_grouped_images(self):
        # 从4个通道文件夹加载所有PNG图片并按文件名分组
        self.grouped_images = {}
        channel_images = []
        
        # 收集每个通道的图片
        for channel_dir in self.channel_dirs:
            images = glob.glob(os.path.join(channel_dir, "*.png"))
            # 按文件名建立索引
            img_dict = {os.path.basename(img): img for img in images}
            channel_images.append(img_dict)
            
        # 找到所有通道共有的图片文件名
        if channel_images and all(channel_images):
            common_filenames = set(channel_images[0].keys())
            for img_dict in channel_images[1:]:
                common_filenames.intersection_update(img_dict.keys())
            
            # 按文件名排序并构建分组字典
            for filename in sorted(common_filenames):
                self.grouped_images[filename] = [img_dict[filename] for img_dict in channel_images]
            
        self.status_var.set(f"找到 {len(self.grouped_images)} 组共有的PNG图片")

    def display_current_images(self):
        if 0 <= self.current_index < len(self.image_names):
            filename = self.image_names[self.current_index]
            image_paths = self.grouped_images[filename]
            base_name = os.path.splitext(filename)[0]

            # 清除现有图片显示
            for widget in self.image_frame.winfo_children():
                widget.destroy()

            # 创建2x2网格布局显示4个通道的图片
            for i, path in enumerate(image_paths):
                # 创建通道框架
                channel_frame = ttk.Frame(self.image_frame, borderwidth=2, relief=tk.RAISED)
                channel_frame.grid(row=i//2, column=i%2, padx=5, pady=5, sticky="nsew")
                
                # 通道名称
                channel_name = os.path.basename(os.path.dirname(path))
                ttk.Label(channel_frame, text=f"通道: {channel_name}", font=('SimHei', 10, 'bold')).pack(pady=2)
                
                # 文件名
                ttk.Label(channel_frame, text=f"文件名: {filename}", font=('SimHei', 9)).pack(pady=2)
                
                # NPY文件状态
                npy_path = os.path.join(os.path.dirname(path), f"{base_name}.npy")
                npy_status = "找到NPY文件" if os.path.exists(npy_path) else "未找到NPY文件"
                ttk.Label(channel_frame, text=npy_status, font=('SimHei', 9)).pack(pady=2)
                
                # 图片显示区域
                img_frame = ttk.Frame(channel_frame)
                img_frame.pack(fill=tk.BOTH, expand=True, pady=5)
                
                try:
                    # 打开并调整图片大小
                    img = Image.open(path)
                    img.thumbnail((400, 300))  # 缩小图片以适应网格
                    photo = ImageTk.PhotoImage(img)
                    img_label = ttk.Label(img_frame, image=photo)
                    img_label.image = photo  # 保持引用
                    img_label.pack(fill=tk.BOTH, expand=True)
                except Exception as e:
                    ttk.Label(img_frame, text=f"无法加载图片:\n{str(e)}", anchor=tk.CENTER).pack(fill=tk.BOTH, expand=True)

            # 配置网格权重，使图片区域可调整大小
            self.image_frame.grid_rowconfigure(0, weight=1)
            self.image_frame.grid_rowconfigure(1, weight=1)
            self.image_frame.grid_columnconfigure(0, weight=1)
            self.image_frame.grid_columnconfigure(1, weight=1)

            # 更新状态栏
            self.status_var.set(f"当前图片组: {filename} ({self.current_index + 1}/{len(self.image_names)}) | 共 {len(self.channel_dirs)} 个通道")

    def delete_current_files(self):
        if 0 <= self.current_index < len(self.image_names):
            filename = self.image_names[self.current_index]
            base_name = os.path.splitext(filename)[0]
            deleted_files = []
            errors = []

            try:
                # 删除所有通道中的图片和NPY文件
                for i, channel_dir in enumerate(self.channel_dirs):
                    # 图片文件路径
                    img_path = os.path.join(channel_dir, filename)
                    # NPY文件路径
                    npy_path = os.path.join(channel_dir, f"{base_name}.npy")

                    # 删除图片
                    if os.path.exists(img_path):
                        try:
                            os.remove(img_path)
                            deleted_files.append(f"通道{i+1}: {os.path.basename(img_path)}")
                        except Exception as e:
                            errors.append(f"通道{i+1}图片删除失败: {str(e)}")

                    # 删除NPY文件
                    if os.path.exists(npy_path):
                        try:
                            os.remove(npy_path)
                            deleted_files.append(f"通道{i+1}: {os.path.basename(npy_path)}")
                        except Exception as e:
                            errors.append(f"通道{i+1}NPY删除失败: {str(e)}")

                # 显示结果
                if deleted_files:
                    result_msg = f"成功删除 {len(deleted_files)} 个文件:\n{chr(10).join(deleted_files)}"
                    if errors:
                        result_msg += f"\n\n遇到 {len(errors)} 个错误:\n{chr(10).join(errors)}"
                    messagebox.showinfo("删除结果", result_msg)

                    # 重新加载并更新显示
                    self.load_grouped_images()
                    self.image_names = list(self.grouped_images.keys())
                    
                    # 调整索引
                    if self.current_index >= len(self.image_names) and self.image_names:
                        self.current_index = len(self.image_names) - 1
                    elif not self.image_names:
                        self.current_index = -1
                    
                    self.update_ui_state()
                    if self.image_names:
                        self.display_current_images()
                    else:
                        # 清空显示
                        for widget in self.image_frame.winfo_children():
                            widget.destroy()
                        ttk.Label(self.image_frame, text="所有通道中已没有共有的PNG图片", anchor=tk.CENTER).pack(fill=tk.BOTH, expand=True)
                else:
                    messagebox.showwarning("警告", "没有找到可删除的文件")

            except Exception as e:
                messagebox.showerror("错误", f"删除文件时出错:\n{str(e)}")

    def previous_image(self):
        if self.image_names:
            self.current_index = (self.current_index - 1) % len(self.image_names)
            self.display_current_images()

    def next_image(self):
        if self.image_names:
            self.current_index = (self.current_index + 1) % len(self.image_names)
            self.display_current_images()

    def update_ui_state(self):
        # 更新按钮状态
        has_images = len(self.image_names) > 0
        self.delete_btn.config(state=tk.NORMAL if has_images else tk.DISABLED)
        self.prev_btn.config(state=tk.NORMAL if has_images else tk.DISABLED)
        self.next_btn.config(state=tk.NORMAL if has_images else tk.DISABLED)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageManagerApp(root)
    root.mainloop()