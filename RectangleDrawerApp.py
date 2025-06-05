import tkinter as tk
from PIL import Image, ImageTk

class RectangleDrawerApp:
    def __init__(self, initial_coords, image_path=None,image=None):
        self.master = tk.Tk()
        self.master.title("Image Display App")

        
        if image_path is not None:
            self.image = Image.open(image_path)
        else:
            self.image = image
        
        self.canvas = tk.Canvas(self.master, width=self.image.width, height=self.image.height)
        self.canvas.pack()

        
        self.tk_image = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

        
        self.initial_coords = initial_coords

        
        self.size_entries = []
        for i, (x, y) in enumerate(self.initial_coords):
            label = tk.Label(self.master, text=f"({x}, {y}):")
            label.pack()
            entry = tk.Entry(self.master)
            entry.pack()
            self.size_entries.append(entry)

        self.draw_button = tk.Button(self.master, text="Draw rectangle", command=self.draw_boxes)
        self.draw_button.pack()

        self.end_button = tk.Button(self.master, text="over", command=self.end_program)
        self.end_button.pack()

        self.final_sizes = [] 

    def draw_boxes(self):
        
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)  

        for (x, y), entry in zip(self.initial_coords, self.size_entries):
            size_input = entry.get()
            try:
                width = int(size_input)
                x1, y1 = x-width//2, y-width
                x2, y2 = x + width//2, y
                self.canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=2)
            except ValueError:
                print(f"Draw rectangle error")

    def end_program(self):

        self.final_sizes.clear()  
        for entry in self.size_entries:
            size_input = entry.get()
            try:
                width = int(size_input)
                self.final_sizes.append(width)
            except ValueError:
                self.final_sizes.append(None)  

        print("最终的各方框尺寸:", self.final_sizes)
        self.master.quit()  

    def get_final_sizes(self):
        return self.final_sizes

    def run(self):
        self.master.mainloop()
        return self.get_final_sizes()  


if __name__ == "__main__":
    image_path = "/media/dlz/Data/workspace/my_work/VideoSD_v1_1/output/Astronaut___Forests, lane,tall_trees, thick_leaves, light_and_shadow, small_animals/2024-08-27T15-04-08/0_0_org_bg.png"
    initial_coords = [
        (256, 256),  
        (256, 400),  
    ]
    app = RectangleDrawerApp(image_path=image_path, initial_coords=initial_coords)
    final_sizes = app.run() 







