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
