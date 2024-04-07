# # Initialize a list to store the reconstructed images
# reconstructed_images = []

# # Loop over each lower resolution image
# for half_res_img in half_res_img:
#     # Convert the image to RGB
#     half_res_img = cv2.cvtColor(half_res_img, cv2.COLOR_BGR2RGB)
    
#     # Pad the image with zeros
#     half_res_img = cv2.copyMakeBorder(half_res_img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
#     # Initialize a list to store the pixels of the reconstructed image
#     reconstructed_img = []
    
#     # Loop over each pixel in the lower resolution image
#     for i in range(1, half_res_img.shape[0] - 1):
#         for j in range(1, half_res_img.shape[1] - 1):
#             # Find the eight neighbors
#             neighbors = []
#             for di in [-1, 0, 1]:
#                 for dj in [-1, 0, 1]:
#                     # Append each color channel separately
#                     neighbors.extend(half_res_img[i + di, j + dj].tolist())
            
#             # Convert the neighbors to a PyTorch tensor and move it to the device
#             neighbors = torch.tensor(neighbors, dtype=torch.float).to(device)
            
#             # Run the model on the neighbors
#             output = model(neighbors)
            
#             # Add the output to the reconstructed image
#             reconstructed_img.append(output.tolist())
    
#     # Reshape the reconstructed image to the original image size
#     reconstructed_img = np.array(reconstructed_img, dtype=np.uint8).reshape((half_res_img.shape[0] - 2, half_res_img.shape[1] - 2, 3))
    
#     # Add the reconstructed image to the list
#     reconstructed_images.append(reconstructed_img)

# # Plot the reconstructed images
# plt.figure(figsize=(8, 6))
# for i, img in enumerate(reconstructed_images[:10]):  # Limit to first 10 images
#     plt.subplot(2, 5, i + 1)
#     plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     plt.axis("off")
#     plt.title(f"Reconstructed Image {i + 1}")
#     plt.tight_layout()
# plt.show()