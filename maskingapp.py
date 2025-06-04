import tkinter as tk
from PIL import Image, ImageDraw, ImageTk
import numpy as np
import torch
from tkinter import simpledialog,messagebox
import torchvision.transforms as transforms



tensor = transforms.ToTensor()
class MaskingApp:
    def __init__(self, background_image_path=None,image=None):
        self.background_image_path = background_image_path
        self.master = tk.Tk()
        self.master.title("Mask Drawing App")

        # 加载背景图像
        if background_image_path == None:
            self.background_image = image.convert("RGBA")
        else:
            self.background_image = Image.open(background_image_path).convert("RGBA")

        # 创建画布
        width, height = self.background_image.size
        self.canvas = tk.Canvas(self.master, bg="white", width=width, height=height)
        self.canvas.pack()

        # 初始化掩膜图像和绘图对象
        self.mask_image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        self.draw = ImageDraw.Draw(self.mask_image)

        #用于返回淹没颜色列表
        self.tips = ''

        # 预定义颜色
        self.colors = [
            (255, 0, 0, 128),  # 红色
            (0, 255, 0, 128),  # 绿色
            (0, 0, 255, 128),  # 蓝色
            (255, 255, 0, 128),  # 黄色
            (255, 165, 0, 128),  # 橙色
            (128, 0, 128, 128),  # 紫色
            (0, 255, 255, 128),  # 青色
            (255, 192, 203, 128),  # 粉色
            (128, 128, 128, 128),  # 灰色
            (0, 0, 0, 128)  # 黑色
        ]
        self.colors_name = {
            (255, 0, 0, 128):'红',
            (0, 255, 0, 128):'绿',
            (0, 0, 255, 128):'蓝',
            (255, 255, 0, 128):'黄',
            (255, 165, 0, 128):'橙',
            (128, 0, 128, 128):'紫',
            (0, 255, 255, 128):'青',
            (255, 192, 203, 128):'粉',
            (128, 128, 128, 128):'灰',
            (0, 0, 0, 128):'黑'
        }
        self.current_color = self.colors[0]  # 默认颜色

        # 创建颜色选择按钮
        self.color_buttons = []
        for color in self.colors:
            button = tk.Button(self.master, text=self.colors_name[color],bg=self.rgb_to_hex(color[:3]), command=lambda c=color: self.set_color(c),
                               width=2)
            button.pack(side=tk.LEFT)
            self.color_buttons.append(button)

        # 鼠标事件初始化
        self.points = []
        self.canvas.bind("<Button-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

        # 显示背景图像
        self.tk_background_image = ImageTk.PhotoImage(self.background_image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_background_image)

        # 添加完成选择按钮
        self.complete_button = tk.Button(self.master, text="完成选择", command=self.complete_selection)
        self.complete_button.pack()

        # 添加输入数字的按钮
        # self.input_button = tk.Button(self.master, text="输入数字", command=self.input_numbers)
        # self.input_button.pack()

        # 添加输出结果的按钮
        self.output_button = tk.Button(self.master, text="输出结果", command=self.output_results)
        self.output_button.pack()

        self.mask_tensor = None  # 用于存储掩膜张量
        self.result_image = None  # 用于存储带掩膜的图像
        self.bw_masks = None  # 用于存储独立黑白掩膜
        self.numbers = []  # 用于存储用户输入的数字
        self.weight_tensor = None #用于存储权重淹没

    def rgb_to_hex(self, rgb):
        return f'#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}'

    def set_color(self, color):
        self.current_color = color

    def on_button_press(self, event):
        self.points.clear()
        self.points.append((event.x, event.y))

    def on_mouse_drag(self, event):
        self.points.append((event.x, event.y))
        self.canvas.delete("mask")
        self.canvas.create_polygon(self.points, outline="black", fill="", tags="mask")

    def on_button_release(self, event):
        if len(self.points) > 1:
            self.draw.polygon(self.points, fill=self.current_color)
            self.update_canvas()

    def update_canvas(self):
        combined = Image.alpha_composite(self.background_image, self.mask_image)
        self.tk_combined_image = ImageTk.PhotoImage(combined)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_combined_image)

    def complete_selection(self):
        self.mask_tensor = np.array(self.mask_image)
        self.result_image = Image.alpha_composite(self.background_image, self.mask_image)

        # 生成黑白掩膜
        self.generate_bw_mask()

        # self.master.quit()  # 退出主循环

    def generate_bw_mask(self):
        # 确保使用 GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载图像
        data = np.array(self.mask_image)

        # 将数据转换为 PyTorch 张量，并移动到 GPU
        data_tensor = torch.tensor(data, device=device)

        # 获取唯一颜色
        unique_colors = torch.unique(data_tensor.view(-1, data_tensor.shape[2]), dim=0)


        # 生成掩模图像
        self.bw_masks = None
        color_list = []
        for i, color in enumerate(unique_colors):
            if color[3]!=0:
                # 创建全黑图像
                mask = torch.zeros(512, 512)
                # 创建掩模
                mask[(data_tensor[:, :, :3] == color[:3]).all(dim=2)] = 1  # 只考虑 RGB

                if self.bw_masks is None:
                    self.bw_masks = torch.unsqueeze(mask, dim=0)
                else:
                    self.bw_masks = torch.cat((self.bw_masks, torch.unsqueeze(mask, dim=0)),dim=0)
                color = color.cpu().numpy()
                color = (color[0],color[1],color[2],color[3])
                color_list.append(color)
        a = self.input_numbers(self.bw_masks.shape[0],color_list)
        a = a

    def input_numbers(self,sum,color_list):
        # 弹出输入框，输入 N 个 0 到 1 之间的数字
        self.tips = ''.join(f"{arg}_" for arg in [self.colors_name[key] for key in color_list])
        input_str = simpledialog.askstring("输入数字", f"请输入{len(color_list)}个({self.tips}) 0 到 1 之间的数字，用逗号分隔:")
        if input_str:
            try:
                self.numbers = [float(num) for num in input_str.split(",")]
                if all(0 <= num <= 1 for num in self.numbers):
                    print("输入的数字:", self.numbers)

                    # 创建一个 512x512 的张量用于存储最终结果
                    weight_tensor_temp = torch.zeros((512, 512))

                    # 根据输入数值和对应掩膜生成最终张量
                    for i, value in enumerate(self.numbers):
                        if i < self.bw_masks.shape[0]:  # 确保掩膜索引在范围内
                            weight_tensor_temp += self.bw_masks[i] * value

                    # 将生成的张量存储为类的一个属性，便于后续处理
                    self.weight_tensor = weight_tensor_temp
                    print("生成的最终张量:weight_tensor")


                    return self.numbers
                else:
                    tk.messagebox.showerror("错误", "所有数字必须在 0 到 1 之间!")
            except ValueError:
                tk.messagebox.showerror("错误", "请输入有效的数字!")

    def output_results(self):
        # 输出掩膜和用户输入的数字
        print("掩膜张量:", self.mask_tensor)
        print("带掩膜的图像:", self.result_image)
        print("用户输入的数字:", self.numbers)

    def run(self):
        self.master.mainloop()
        return self.weight_tensor, self.mask_tensor, self.result_image, self.bw_masks, self.numbers, self.tips


# 使用示例
if __name__ == "__main__":
    image_path = "/media/dlz/Data/workspace/my_work/VideoSD_v1_1/output/Astronaut___Forests, lane,tall_trees, thick_leaves, light_and_shadow, small_animals/2024-08-23T09-10-27/0_0_org_bg.png"  # 替换为你的背景图片路径

    app = MaskingApp(image_path)
    mask_data, masked_image, bw_masks, user_numbers = app.run()

    # 保存掩膜和带掩膜的图像
    Image.fromarray(mask_data).save("/media/dlz/Data/workspace/my_work/VideoSD_v1_1/output/mask_image.png")
    masked_image.save("/media/dlz/Data/workspace/my_work/VideoSD_v1_1/output/masked_image.png")

    # 输出用户输入的数字
    print("用户输入的数字:", user_numbers)
