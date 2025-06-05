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

        if background_image_path == None:
            self.background_image = image.convert("RGBA")
        else:
            self.background_image = Image.open(background_image_path).convert("RGBA")


        width, height = self.background_image.size
        self.canvas = tk.Canvas(self.master, bg="white", width=width, height=height)
        self.canvas.pack()

        # 
        self.mask_image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        self.draw = ImageDraw.Draw(self.mask_image)

        self.tips = ''

        self.colors = [
            (255, 0, 0, 128),  # 
            (0, 255, 0, 128),  # 
            (0, 0, 255, 128),  # 
            (255, 255, 0, 128),  # 
            (255, 165, 0, 128),  # 
            (128, 0, 128, 128),  # 
            (0, 255, 255, 128),  # 
            (255, 192, 203, 128),  # 
            (128, 128, 128, 128),  # 
            (0, 0, 0, 128)  # 
        ]
        self.colors_name = {
            (255, 0, 0, 128): 'Red',
            (0, 255, 0, 128): 'Green',
            (0, 0, 255, 128): 'Blue',
            (255, 255, 0, 128): 'Yellow',
            (255, 165, 0, 128): 'Orange',
            (128, 0, 128, 128): 'Purple',
            (0, 255, 255, 128): 'Cyan',
            (255, 192, 203, 128): 'Pink',
            (128, 128, 128, 128): 'Gray',
            (0, 0, 0, 128): 'Black'
        }
        self.current_color = self.colors[0]  # 

        # 
        self.color_buttons = []
        for color in self.colors:
            button = tk.Button(self.master, text=self.colors_name[color],bg=self.rgb_to_hex(color[:3]), command=lambda c=color: self.set_color(c),
                               width=2)
            button.pack(side=tk.LEFT)
            self.color_buttons.append(button)

        # 
        self.points = []
        self.canvas.bind("<Button-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

        # 
        self.tk_background_image = ImageTk.PhotoImage(self.background_image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_background_image)

        # 
        self.complete_button = tk.Button(self.master, text="Confirm selection", command=self.complete_selection)
        self.complete_button.pack()


        self.output_button = tk.Button(self.master, text="output result", command=self.output_results)
        self.output_button.pack()

        self.mask_tensor = None  # 
        self.result_image = None  #
        self.bw_masks = None  # 
        self.numbers = []  # 
        self.weight_tensor = None #

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

        # 
        self.generate_bw_mask()

        # self.master.quit()  # 

    def generate_bw_mask(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        
        data = np.array(self.mask_image)

        data_tensor = torch.tensor(data, device=device)

        
        unique_colors = torch.unique(data_tensor.view(-1, data_tensor.shape[2]), dim=0)



        self.bw_masks = None
        color_list = []
        for i, color in enumerate(unique_colors):
            if color[3]!=0:

                mask = torch.zeros(512, 512)

                mask[(data_tensor[:, :, :3] == color[:3]).all(dim=2)] = 1  

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

        self.tips = ''.join(f"{arg}_" for arg in [self.colors_name[key] for key in color_list])
        input_str = simpledialog.askstring("Input number", f"Please enter {len(color_list)} numbers between 0 and 1 ({self.tips}), separated by commas:")
        if input_str:
            try:
                self.numbers = [float(num) for num in input_str.split(",")]
                if all(0 <= num <= 1 for num in self.numbers):

                    weight_tensor_temp = torch.zeros((512, 512))

                    for i, value in enumerate(self.numbers):
                        if i < self.bw_masks.shape[0]:  
                            weight_tensor_temp += self.bw_masks[i] * value


                    self.weight_tensor = weight_tensor_temp


                    return self.numbers
                else:
                    tk.messagebox.showerror("error number", "All numbers must be between 0 and 1!")
            except ValueError:
                tk.messagebox.showerror("error number")

    def output_results(self):

        print("Number entered:", self.numbers)

    def run(self):
        self.master.mainloop()
        return self.weight_tensor, self.mask_tensor, self.result_image, self.bw_masks, self.numbers, self.tips



if __name__ == "__main__":
    image_path = "/media/dlz/Data/workspace/my_work/VideoSD_v1_1/output/Astronaut___Forests, lane,tall_trees, thick_leaves, light_and_shadow, small_animals/2024-08-23T09-10-27/0_0_org_bg.png" 

    app = MaskingApp(image_path)
    mask_data, masked_image, bw_masks, user_numbers = app.run()

    Image.fromarray(mask_data).save("/media/dlz/Data/workspace/my_work/VideoSD_v1_1/output/mask_image.png")
    masked_image.save("/media/dlz/Data/workspace/my_work/VideoSD_v1_1/output/masked_image.png")


