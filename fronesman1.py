import cv2

import numpy as np

import tkinter as tk

from matplotlib import pyplot as plt

from tkinter import filedialog

from PIL import ImageTk

# img = cv2.imread('img.png')


HEIGHT = 600

WIDTH = 600

root = tk.Tk()


def open_image(entry):
    img = filedialog.askopenfilename(initialdir="/Pictures", title="Select Image",

                                     filetypes=(("jpeg files", "*.jpg"), ("png files", "*.png"), ("All files", "*.*")))

    img = cv2.imread(img)

    cv2.imshow('opened', img)

    cv2.waitKey(0)

    cv2.destroyAllWindows()

    # cv2.imwrite('img_copy.png', img)

    print(entry)


def transfomation_image(entry):
    img1 = filedialog.askopenfilename(initialdir="/Pictures", title="Select Image",

                                      filetypes=(("jpeg files", "*.jpg"), ("png files", "*.png"), ("All files", "*.*")))

    img1 = cv2.imread(img1)

    rows, cols = img1.shape[:2]

    print("Height: ", rows)

    print("Width: ", cols)

    scaled_img = cv2.resize(img1, None, fx=1 / 2, fy=1 / 2)

    # T = translation matrix

    T = np.float32([[1, 0, 100], [0, 1, 50]])

    img_translation = cv2.warpAffine(img1, T, (rows, cols))

    # R = rotation matrix

    R = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)

    rotated_img = cv2.warpAffine(img1, R, (cols, rows))

    cv2.imshow("Original image", img1)

    cv2.imshow("Scaled image", scaled_img)

    cv2.imshow("Translated image", img_translation)

    cv2.imshow("Rotated image", rotated_img)

    cv2.waitKey(0)

    cv2.destroyAllWindows()

    print(entry)


def prewit(entry):
    img2 = filedialog.askopenfilename(initialdir="/Pictures", title="Select Image",

                                      filetypes=(("jpeg files", "*.jpg"), ("png files", "*.png"), ("All files", "*.*")))

    img = cv2.imread(img2)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_gaussian = cv2.GaussianBlur(gray, (3, 3), 0)

    # prewitt

    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

    img_prewittx = cv2.filter2D(img_gaussian, -1, kernelx)

    img_prewitty = cv2.filter2D(img_gaussian, -1, kernely)

    cv2.imshow("original", img)

    cv2.imshow("Prewitt X", img_prewittx)

    cv2.imshow("Prewitt Y", img_prewitty)

    cv2.waitKey(0)

    cv2.destroyAllWindows()

    print(entry)


def Colorspacing(entry):
    img3 = filedialog.askopenfilename(initialdir="/Pictures", title="Select Image",

                                      filetypes=(("jpeg files", "*.jpg"), ("png files", "*.png"), ("All files", "*.*")))

    img = cv2.imread(img3)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # converting the image to gray color

    red = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # HSV it stands for high saturation value

    cv2.imshow('oridinal', img)  # calling the original pic

    cv2.imshow('gray', gray)  # calling the gray pic

    cv2.imshow('HSV', red)  ##calling the HSV pic

    cv2.waitKey(0)

    cv2.destroyAllWindows()

    print(entry)


def High_pass_filter(entry):
    img4 = filedialog.askopenfilename(initialdir="/Pictures", title="Select Image",

                                      filetypes=(("jpeg files", "*.jpg"), ("png files", "*.png"), ("All files", "*.*")))

    img = cv2.imread(img4)

    blurr = cv2.blur(img, (5, 5))

    plt.subplot(121), plt.imshow(img), plt.title('Original')

    plt.xticks([]), plt.yticks([])

    plt.subplot(122), plt.imshow(blurr), plt.title('Blurred')

    plt.xticks([]), plt.yticks([])

    plt.show()

    print(entry)


canvas = tk.Canvas(root, height=HEIGHT, width=WIDTH)

canvas.pack()

background_image = tk.PhotoImage(file='camel.png')

background_label = tk.Label(root, image=background_image)

background_label.place(relwidth=1, relheight=1)

frame = tk.Frame(root, bg='grey', bd=1)

frame.place(relx=0.27, rely=0.1, relheight=0.4, relwidth=0.5)

entry = tk.Frame(root, bg='gray', bd=5)

entry.place()

label = tk.Label(frame, text="Welcome to Editor", bg='purple', font="Bauhaus 14 bold underline")

label.pack()

button1 = tk.Button(frame, text="Open Image", font=50, bg='yellow', command=lambda: open_image(entry))

button2 = tk.Button(frame, text="Image Transformation", bg='yellow', font=40,
                    command=lambda: transfomation_image(entry))

button3 = tk.Button(frame, text="Image Prewit", font=50, bg='yellow', command=lambda: prewit(entry))

button4 = tk.Button(frame, text="Image Colorspace", font=50, bg='yellow', command=lambda: Colorspacing(entry))

button5 = tk.Button(frame, text="Image HPF", font=50, bg='yellow', command=lambda: High_pass_filter(entry))

button1.place(relx=0, rely=0.15, relheight=0.15, relwidth=1)

button2.place(relx=0, rely=0.31, relheight=0.15, relwidth=1)

button3.place(relx=0, rely=0.47, relheight=0.15, relwidth=1)

button4.place(relx=0, rely=0.64, relheight=0.15, relwidth=1)

button5.place(relx=0, rely=0.81, relheight=0.15, relwidth=1)

root.geometry("600x600")

root.title('frones Image Processing')

root.mainloop()