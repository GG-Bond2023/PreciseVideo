import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
import pylab

class CurveDrawingApp:
    def __init__(self, repoint_num,image_path=None,image=None):
        self.master = tk.Tk()
        self.master.title("Curve Drawing App")
        self.repoint_num = repoint_num
        self.re_point = []
        self.oval_ids = []  # 用于存储椭圆的 ID
        self.inputs = []  # 用于存储输入框的引用
        # 加载图像
        if image_path is not None:
            self.image = Image.open(image_path)
        else:
            self.image = image



        # 创建画布
        self.canvas = tk.Canvas(self.master, width=self.image.width, height=self.image.height)
        self.canvas.pack()

        # 显示图像
        self.tk_image = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

        # 鼠标事件初始化
        self.current_points = []
        self.curves = None
        self.canvas.bind("<Button-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

        # 撤销按钮
        self.undo_button = tk.Button(self.master, text="打印当前坐标", command=self.print_p)
        self.undo_button.pack()

    def on_button_press(self, event):
        self.current_points.clear()
        self.current_points.append((event.x, event.y))

    def on_mouse_drag(self, event):
        self.current_points.append((event.x, event.y))
        self.redraw_current_curve()

    def on_button_release(self, event):
        if len(self.current_points) > 1:
            # 保存当前曲线的坐标
            self.curves = self.current_points.copy()
            self.redraw_all_curves()

    def redraw_current_curve(self):
        self.canvas.delete("current_curve")
        if self.current_points:
            self.canvas.create_line(self.current_points, fill="red", width=2, tags="current_curve")

    def redraw_all_curves(self):
        self.canvas.delete("all_curves")
        self.canvas.create_line(self.curves, fill="red", width=2, tags="all_curves")



    def print_p(self):

        step_lenth = len(self.curves) // (self.repoint_num)
        self.re_point = []
        for i in range(self.repoint_num):
            self.re_point.append(self.curves[i * step_lenth])
            oval_id = self.canvas.create_oval(self.curves[i*step_lenth][0]-3, self.curves[i*step_lenth][1]-3, self.curves[i*step_lenth][0]+3, self.curves[i*step_lenth][1]+3, fill="blue", outline="")
            self.oval_ids.append(oval_id)  # 保存椭圆的 ID


        print("当前坐标序列:",self.curves)
        print("间隔坐标点:",self.re_point)


    def run(self):
        self.master.mainloop()

        return self.re_point


# 使用示例
if __name__ == "__main__":
    image_path = "/media/dlz/Data/workspace/my_work/VideoSD_v1_1/output/Astronaut___Forests, lane,tall_trees, thick_leaves, light_and_shadow, small_animals/2024-08-23T09-10-27/0_0_org_bg.png"  # 替换为你的背景图片路径
    app = CurveDrawingApp(image_path=image_path,repoint_num=8)
    a = app.run()
    print("曲线轨迹坐标:", a)


    cropped_images = []

    image_paths = [
        "/media/dlz/Data/workspace/my_work/VideoSD_v1_1/output/beach_Astronaut_30_1326_no_True_2024-08-09T15-39-44_0.png",
        "/media/dlz/Data/workspace/my_work/VideoSD_v1_1/output/beach_Astronaut_30_1326_no_True_2024-08-09T15-39-44_1.png",
        "/media/dlz/Data/workspace/my_work/VideoSD_v1_1/output/beach_Astronaut_30_1326_no_True_2024-08-09T15-39-44_2.png",
        "/media/dlz/Data/workspace/my_work/VideoSD_v1_1/output/beach_Astronaut_30_1326_no_True_2024-08-09T15-39-44_3.png",
        "/media/dlz/Data/workspace/my_work/VideoSD_v1_1/output/beach_Astronaut_30_1326_no_True_2024-08-09T15-39-44_4.png",
        "/media/dlz/Data/workspace/my_work/VideoSD_v1_1/output/beach_Astronaut_30_1326_no_True_2024-08-09T15-39-44_5.png",
        "/media/dlz/Data/workspace/my_work/VideoSD_v1_1/output/beach_Astronaut_30_1326_no_True_2024-08-09T15-39-44_6.png",
        "/media/dlz/Data/workspace/my_work/VideoSD_v1_1/output/beach_Astronaut_30_1326_no_True_2024-08-09T15-39-44_7.png"
    ]
    for image_path, (x, y) in zip(image_paths, a):
        # 打开图片
        img = Image.open(image_path)

        # 计算截取区域
        width, height = (200,200)
        box = (x-width//2, y-height//2, x + width//2, y + height//2)

        # 截取图片
        cropped_img = img.crop(box)
        cropped_images.append(cropped_img)

    for k in range(8):

        plt.imshow(cropped_images[k])
        pylab.show()

        b = cropped_images[k].resize((512,512))
        plt.imshow(b)
        pylab.show()

        c = b.resize((200,200))
        plt.imshow(c)
        pylab.show()
        c = None