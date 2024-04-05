import torch
import cv2
import matplotlib.pyplot as plt
import random
from utils import*
from torch import nn, optim
from sklearn.metrics import mean_absolute_error
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np
from skimage.util import img_as_float




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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






# Initialize the MLP and move it to the device
model = multi_layer_perceptron().to(device)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Lists to store loss and accuracy values for training, testing and validation sets
train_loss_values = []
train_accuracy_values = []
test_loss_values = []
test_accuracy_values = []
val_loss_values = []
val_accuracy_values = []

EPOCH = 5

# Function to calculate loss and accuracy
def calculate_loss_accuracy(dataset):
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for i, data in enumerate(dataset, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # calculate "accuracy"
            total_predictions += labels.size(0)
            correct_predictions += ((outputs - labels).abs() / labels.abs() < 0.1).sum().item()

            # print statistics
            running_loss += loss.item()
    avg_loss = running_loss / len(dataset)
    accuracy = correct_predictions / total_predictions
    return avg_loss, accuracy

# Train the MLP
for epoch in range(EPOCH):  # loop over the dataset multiple times
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    for i, data in enumerate(train_dataset, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # calculate "accuracy"
        total_predictions += labels.size(0)
        correct_predictions += ((outputs - labels).abs() / labels.abs() < 0.1).sum().item()

        # print statistics
        running_loss += loss.item()
    avg_loss = running_loss / len(train_dataset)
    accuracy = correct_predictions / total_predictions
    # print(f"Epoch {epoch + 1}, Train Loss: {avg_loss}, Train Accuracy: {accuracy}")
    train_loss_values.append(avg_loss)
    train_accuracy_values.append(accuracy)

    model.eval()
    test_loss, test_accuracy = calculate_loss_accuracy(test_dataset)
    # print(f"Epoch {epoch + 1}, Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
    test_loss_values.append(test_loss)
    test_accuracy_values.append(test_accuracy)

    val_loss, val_accuracy = calculate_loss_accuracy(val_dataset)
    # print(f"Epoch {epoch + 1}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")
    val_loss_values.append(val_loss)
    val_accuracy_values.append(val_accuracy)

    print ('Epoch [{}/{}] Train Loss: {:.4f} | Train Accuracy: {:.2f} | Test Loss: {:.4f} | Test Accuracy: {:.2f} | Validation Loss: {:.4f} | Validation Accuracy: {:.2f}  '.format(epoch+1,
            EPOCH, avg_loss, accuracy, test_loss, test_accuracy, val_loss, val_accuracy))



print('Finished Training')

# Plot loss and accuracy
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(train_loss_values, label='Train')
plt.plot(test_loss_values, label='Test')
plt.plot(val_loss_values, label='Validation')
plt.title('Loss per epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracy_values, label='Train')
plt.plot(test_accuracy_values, label='Test')
plt.plot(val_accuracy_values, label='Validation')
plt.title('Accuracy per epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()











# Function to generate high-resolution images
def generate_images(model, dataset):
    model.eval()
    with torch.no_grad():
        images = []
        for data in dataset:
            inputs = data[0].to(device)
            outputs = model(inputs)
            images.append(outputs.cpu().numpy())
        return images

# Generate high-resolution images
high_res_images = generate_images(model, test_dataset)


# Calculate SSIM and PSNR
ssim_values = []
psnr_values = []
for i, img in enumerate(high_res_images):
    img_true = img_as_float(test_labels[i].cpu().numpy())
    img_test = img_as_float(img)
    ssim_values.append(ssim(img_true, img_test, win_size=3, multichannel=True, data_range=1.0))
    psnr_values.append(psnr(img_true, img_test, data_range=1.0))

print(f"Average SSIM: {np.mean(ssim_values)}")
print(f"Average PSNR: {np.mean(psnr_values)}")