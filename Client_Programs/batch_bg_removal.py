# # Import the required modules
# import os
# from rembg import remove
# from PIL import Image

# input_dir = 'Dataset_alpha'
# output_dir = 'Segmented_Dataset'

# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)


# folders = ["Brightness_Decrease", "Brightness_Increase", "Chrome_Open", "Cursor_Movement", "Double_Click", "Initiation", "Left_Click", "Neutral", "Nothing", "Right_Click", "Screenshot", "Scroll", "Shutdown", "Volume_Decrease", "Volume_Increase"]

# # Process each folder
# for folder in folders:
#     # Define the input and output paths for this folder
#     input_path = os.path.join(input_dir, folder)
#     output_path = os.path.join(output_dir, folder)

#     # Create the output path for this folder if it does not exist
#     if not os.path.exists(output_path):
#         os.makedirs(output_path)

#     # Process each image in this folder
#     for filename in os.listdir(input_path):
#         # Define the input and output paths for this image
#         input_image_path = os.path.join(input_path, filename)
#         output_image_path = os.path.join(output_path, filename)

#         # Open the input image and remove the background
#         input_image = Image.open(input_image_path)
#         output_image = remove(input_image)

#         # Convert the output image to RGB mode
#         output_image = output_image.convert('RGB')

#         # Save the output image with the same name
#         output_image.save(output_image_path)

#         # Print a progress message
#         print(f"Processed image {filename} in folder {folder}.")

# # Print a success message
# print("Images segmented successfully.")

# Import the required modules
import os
from rembg import remove
from PIL import Image

# Define the input and output paths for this image
input_image_path = '/Users/dipashrestha/Downloads/a.jpg'
output_image_path = '/Users/dipashrestha/Downloads/a.png'

# Open the input image and remove the background
input_image = Image.open(input_image_path)
output_image = remove(input_image)

# Convert the output image to RGB mode
output_image = output_image.convert('RGB')

# Save the output image with the same name
output_image.save(output_image_path)

# Print a success message
print("Image segmented successfully.")
