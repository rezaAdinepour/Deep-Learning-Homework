import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import cv2
import os

def read_img(dir, format):
    images = []
    for img in os.listdir(dir):
        if img.endswith("." + format):
            images.append(cv2.imread(os.path.join(dir, img)))
    return len(images), images



dir = "Dataset-Q5/"
_, images = read_img(dir, "jpg")
print("total images: ", len(images))


plt.figure(figsize=(8, 6))
for i, img in enumerate(images):
    plt.subplot(2, 5, i + 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(f"Image {i + 1}")
    plt.tight_layout()


# half resulution of images
half_res_img = []
for img in images:
    half_res_img.append(cv2.resize(img, (img.shape[0] // 2, img.shape[1] // 2)))

plt.figure(figsize=(8, 6))
for i, img in enumerate(half_res_img):
    plt.subplot(2, 5, i + 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(f"Image {i + 1}")
    plt.tight_layout()
plt.show()