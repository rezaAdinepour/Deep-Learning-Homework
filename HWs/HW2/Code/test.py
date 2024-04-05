import torch
import cv2
import matplotlib.pyplot as plt
import random
from utils import*





# load images
dir = "Dataset-Q5/"
_, images = read_img(dir, "jpg")
print("total images: ", len(images))

# plot full resolution images
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

# plot half resolution of images
plt.figure(figsize=(8, 6))
for i, img in enumerate(half_res_img):
    plt.subplot(2, 5, i + 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(f"Image {i + 1}")
    plt.tight_layout()
# plt.show()



# # Initialize the dataset
# dataset = []

# # Loop over each image and its lower resolution version
# for img, half_res_img in zip(images, half_res_img):
#     # Convert the images to RGB
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     half_res_img = cv2.cvtColor(half_res_img, cv2.COLOR_BGR2RGB)
    
#     # Pad the lower resolution image with zeros
#     half_res_img = cv2.copyMakeBorder(half_res_img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
#     # Initialize the image dataset
#     img_dataset = []
    
#     # Loop over each pixel in the original image
#     for i in range(img.shape[0]):
#         for j in range(img.shape[1]):
#             # Map the pixel to the lower resolution image
#             i_half = i // 2
#             j_half = j // 2
            
#             # Find the eight neighbors
#             neighbors = []
#             for di in [-1, 0, 1]:
#                 for dj in [-1, 0, 1]:
#                     # Append each color channel separately
#                     neighbors.extend(half_res_img[i_half + di + 1, j_half + dj + 1].tolist())
            
#             # Add the neighbors and the original pixel to the image dataset
#             img_dataset.append(neighbors + img[i, j].tolist())
    
#     # Add the image dataset to the overall dataset
#     dataset.extend(img_dataset)

# # Convert the dataset to a PyTorch tensor
# dataset = torch.tensor(dataset, dtype=torch.float)

# # Print the size of the tensor
# print(f"Size of dataset tensor: {dataset.size()}")


# Initialize the dataset
dataset = []



# Loop over each image and its lower resolution version
for img, half_res_img in zip(images, half_res_img):
    # Convert the images to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    half_res_img = cv2.cvtColor(half_res_img, cv2.COLOR_BGR2RGB)
    
    # insert zero pixels into the image
    half_res_img = cv2.copyMakeBorder(half_res_img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
    # Initialize the image dataset
    img_dataset = []
    
    # Loop over each pixel in the original image
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # map the pixel to the lower resolution image
            # lower resolution iamge is the half of the original image
            i_half = i // 2
            j_half = j // 2
            
            # find the eight neighbors
            neighbors = []
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    # Append each color channel separately
                    neighbors.extend(half_res_img[i_half + di + 1, j_half + dj + 1].tolist())
            
            # Add the neighbors and the original pixel to the image dataset
            img_dataset.append((neighbors, img[i, j].tolist()))
    
    # Add the image dataset to the overall dataset
    dataset.append(img_dataset)

# Convert the dataset to a PyTorch tensor
dataset = [[torch.tensor(data[0], dtype=torch.float), torch.tensor(data[1], dtype=torch.float)] for img_data in dataset for data in img_data]

# Stack the features and labels separately
features = torch.stack([img_data[0] for img_data in dataset])
labels = torch.stack([img_data[1] for img_data in dataset])

# Print the sizes of the tensors
print(f"Size of features tensor: {features.size()}")
print(f"Size of labels tensor: {labels.size()}")
print('-'*50)




# Split the dataset into training, testing, and validation sets
train_dataset = dataset[:7*32*32]
test_dataset = dataset[7*32*32:9*32*32]
val_dataset = dataset[9*32*32:]

# Shuffle the training dataset
random.shuffle(train_dataset)

# Stack the features and labels separately for each dataset
train_features = torch.stack([data[0] for data in train_dataset])
train_labels = torch.stack([data[1] for data in train_dataset])
test_features = torch.stack([data[0] for data in test_dataset])
test_labels = torch.stack([data[1] for data in test_dataset])
val_features = torch.stack([data[0] for data in val_dataset])
val_labels = torch.stack([data[1] for data in val_dataset])

# Print the sizes of the tensors
print(f"Size of training features tensor: {train_features.size()}")
print(f"Size of training labels tensor: {train_labels.size()}")
print(f"Size of testing features tensor: {test_features.size()}")
print(f"Size of testing labels tensor: {test_labels.size()}")
print(f"Size of validation features tensor: {val_features.size()}")
print(f"Size of validation labels tensor: {val_labels.size()}")





