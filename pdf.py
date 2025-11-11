import os
import sys
from tkinter import Tk, Label, Button, Listbox, Checkbutton, IntVar, filedialog, messagebox, ttk
from tkinter import END, LEFT, RIGHT, BOTH, X, Y, W, E, TOP
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from PIL import Image
 
class ImageToPDFConverter:
    def __init__(self, root):
        self.root = root
        self.root.title('图片批量转PDF工具-By：Killerzeno')
        self.root.geometry('800x700')
 
 
        def resource_path(relative_path):
            try:
                base_path = sys._MEIPASS  # 打包后资源文件的临时目录
            except Exception:
                base_path = os.path.abspath(".")  # 开发环境下的当前目录
            return os.path.join(base_path, relative_path)
 
        # 修改设置窗口图标代码
        try:
            self.root.iconbitmap(resource_path('logo.ico'))  # 使用 resource_path 函数获取图标路径
        except Exception as e:
            messagebox.showwarning('警告', f'无法加载图标文件 logo.ico: {str(e)}')
 
 
 
        # 标题
        title_label = Label(self.root, text='图片批量转PDF工具', font=('Arial', 30, 'bold'), fg='green')
        title_label.pack(pady=10)
 
        # 说明标签
        info_label = Label(self.root, text='点击"添加图片"按钮选择图片文件', font=('Arial', 12))
        info_label.pack(pady=5)
 
        # 图片列表
        self.listbox = Listbox(self.root, selectmode='extended', height=20, width=100)
        self.listbox.pack(pady=10, padx=10, fill=BOTH, expand=True)
 
        # 按钮区域
        buttons_frame = ttk.Frame(self.root)
        buttons_frame.pack(fill=X, padx=10, pady=5)
 
        add_button = Button(buttons_frame, text='添加图片', command=self.add_images)
        add_button.pack(side=LEFT, padx=10)
 
        remove_button = Button(buttons_frame, text='移除选中', command=self.remove_selected)
        remove_button.pack(side=LEFT, padx=10)
 
        clear_button = Button(buttons_frame, text='清空列表', command=self.clear_list)
        clear_button.pack(side=LEFT, padx=10)
 
        # 上移和下移按钮
        move_up_button = Button(buttons_frame, text='上移图片', command=self.move_up)
        move_up_button.pack(side=LEFT, padx=10)
 
        move_down_button = Button(buttons_frame, text='下移图片', command=self.move_down)
        move_down_button.pack(side=LEFT, padx=10)
 
        # 选项区域
        options_frame = ttk.Frame(self.root)
        options_frame.pack(fill=X, padx=10, pady=5)
 
        self.single_pdf_var = IntVar(value=1)
        self.multi_pdf_var = IntVar()
        self.keep_original_size_var = IntVar()  # 新增变量，用于是否保留原始尺寸
 
        single_pdf_check = Checkbutton(options_frame, text='所有图片合并为一个PDF', variable=self.single_pdf_var,
                                       command=lambda: self.toggle_checkboxes(self.single_pdf_var, self.multi_pdf_var))
        single_pdf_check.pack(side=LEFT, padx=10)
 
        multi_pdf_check = Checkbutton(options_frame, text='每张图片生成单独PDF', variable=self.multi_pdf_var,
                                      command=lambda: self.toggle_checkboxes(self.multi_pdf_var, self.single_pdf_var))
        multi_pdf_check.pack(side=LEFT, padx=10)
 
        # 新增选项：是否保留原始尺寸
        keep_original_size_check = Checkbutton(options_frame, text='保留原始尺寸', variable=self.keep_original_size_var)
        keep_original_size_check.pack(side=LEFT, padx=10)
 
        # 转换按钮
        convert_button = Button(self.root, text='转换为PDF', bg='#4CAF50', fg='white', command=self.convert_to_pdf)
        convert_button.pack(fill=X, padx=10, pady=10)
 
        # 进度条
        self.progress = ttk.Progressbar(self.root, orient='horizontal', length=200, mode='determinate')
        self.progress.pack(pady=10)
 
    def toggle_checkboxes(self, selected_var, other_var):
        if selected_var.get() == 1:
            other_var.set(0)
 
    def add_images(self):
        files = filedialog.askopenfilenames(filetypes=[('图片文件', '*.jpg *.jpeg *.png *.bmp *.gif *.tiff')])
        for file in files:
            if self.is_image_file(file):
                self.listbox.insert(END, file)
 
    def remove_selected(self):
        selected_items = self.listbox.curselection()
        for item in selected_items[::-1]:  # 从后往前删除，避免索引变化
            self.listbox.delete(item)
 
    def clear_list(self):
        self.listbox.delete(0, END)
 
    def is_image_file(self, file_path):
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
        return os.path.isfile(file_path) and os.path.splitext(file_path)[1].lower() in image_extensions
 
    def move_up(self):
        selected_items = self.listbox.curselection()
        for item in selected_items:
            if item > 0:
                self.listbox.insert(item - 1, self.listbox.get(item))
                self.listbox.delete(item + 1)
                self.listbox.selection_set(item - 1)
 
    def move_down(self):
        selected_items = self.listbox.curselection()
        for item in selected_items[::-1]:  # 从后往前处理
            if item < self.listbox.size() - 1:
                self.listbox.insert(item + 2, self.listbox.get(item))
                self.listbox.delete(item)
                self.listbox.selection_set(item + 1)
 
    def convert_to_pdf(self):
        if self.listbox.size() == 0:
            messagebox.showwarning('警告', '没有可转换的图片文件!')
            return
 
        output_dir = filedialog.askdirectory()
        if not output_dir:
            return
 
        try:
            if self.single_pdf_var.get():
                output_path = os.path.join(output_dir, 'combined.pdf')
                c = canvas.Canvas(output_path, pagesize=letter)
                for i in range(self.listbox.size()):
                    img_path = self.listbox.get(i)
                    self.add_image_to_pdf(c, img_path)
                    self.update_progress(i + 1, self.listbox.size())
                c.save()
                messagebox.showinfo('完成', f'已生成合并PDF文件: {output_path}')
            else:
                for i in range(self.listbox.size()):
                    img_path = self.listbox.get(i)
                    filename = os.path.splitext(os.path.basename(img_path))[0]
                    output_path = os.path.join(output_dir, f'{filename}.pdf')
                    c = canvas.Canvas(output_path, pagesize=letter)
                    self.add_image_to_pdf(c, img_path)
                    c.save()
                    self.update_progress(i + 1, self.listbox.size())
                messagebox.showinfo('完成', f'已生成{self.listbox.size()}个PDF文件到目录: {output_dir}')
        except Exception as e:
            messagebox.showerror('错误', f'转换过程中发生错误: {str(e)}')
 
    def add_image_to_pdf(self, c, img_path):
        try:
            img = Image.open(img_path)
            img_width, img_height = img.size
 
            if self.keep_original_size_var.get():  # 如果选择保留原始尺寸
                # 设置PDF页面大小为图片的原始尺寸
                c.setPageSize((img_width, img_height))
                # 在PDF页面上绘制图片
                c.drawImage(img_path, 0, 0, width=img_width, height=img_height)
            else:  # 否则自动调整大小以适应PDF页面
                pdf_width, pdf_height = letter
                width_ratio = pdf_width / img_width
                height_ratio = pdf_height / img_height
                scale = min(width_ratio, height_ratio)
 
                x = (pdf_width - img_width * scale) / 2
                y = (pdf_height - img_height * scale) / 2
 
                c.drawImage(img_path, x, y, width=img_width * scale, height=img_height * scale)
 
            c.showPage()
        except Exception as e:
            raise Exception(f'处理图片 {os.path.basename(img_path)} 时出错: {str(e)}')
 
    def update_progress(self, value, max_value):
        self.progress['value'] = (value / max_value) * 100
        self.root.update_idletasks()
 
if __name__ == '__main__':
    root = Tk()
    app = ImageToPDFConverter(root)
    root.mainloop()