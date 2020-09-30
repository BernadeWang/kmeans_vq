import tkinter as tk
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk

import numpy as np
from matplotlib import cm

from src.api_image_vq import api_train_vq


if __name__ == '__main__':
    window = tk.Tk()
    window.title('china')

    file_path = askopenfilename()

    img = Image.open(file_path)
    w1, h1 = img.size
    img1 = ImageTk.PhotoImage(img)

    _, img2 = api_train_vq(np.array(img))
    img2 = cm.ScalarMappable().to_rgba(img2, bytes=True)
    img2 = Image.fromarray(img2).convert('RGB')
    w2, h2 = img2.size
    img2 = ImageTk.PhotoImage(img2)

    canvas = tk.Canvas(window, width=w1+w2, height=h1)
    canvas.create_image(w1/2, h1/2, image=img1)
    canvas.create_image(w1 + w2/2, h2/2, image=img2)
    # img = load_sample_image('china.jpg')
    # canvas = Canvas(window, height=427, width=640)
    # canvas_img = canvas.create_image(image=img)
    canvas.pack()
    window.mainloop()
