import cv2
import os
import numpy as np
from skimage.util import random_noise
from skimage import img_as_ubyte

# Main directory name
main_directory = "Dataset_alpha_split"

# List of subfolder names
subfolder_names = ["test", "train", "validation"]

# List of class names
class_names = []
for subfolder_name in subfolder_names:
    class_names.extend([d for d in os.listdir(os.path.join(main_directory, subfolder_name)) if os.path.isdir(os.path.join(main_directory, subfolder_name, d))])
class_names = list(set(class_names))

# Desired size for resizing
target_size = (380, 380)  # efficientnetv2 S,M,L models

# Augmentation parameters
num_rotations = 4
num_flips = 4
num_color_transformations = 4
brightness_factors = [0.6,1.3]
contrast_factors = [0.7, 1.3]
noise_levels = [0.02, 0.1]
zoom_factors = [0.7, 1.3, 0.8, 1.2]  # Example zoom factors

# Loop through subfolders
for subfolder_name in subfolder_names:
    if subfolder_name == "train":
        for class_name in class_names:
            input_folder_name = os.path.join(main_directory, subfolder_name, class_name)
            output_folder_name = os.path.join(main_directory, subfolder_name, f"{class_name}_processed")

            # Ensure the output folder exists
            if not os.path.exists(output_folder_name):
                os.makedirs(output_folder_name)

            # Loop through images in the input folder
            for image_file in os.listdir(input_folder_name):
                if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    print(f"Processing image {image_file} in folder {input_folder_name}")

                    # Load image
                    image = cv2.imread(os.path.join(input_folder_name, image_file))

                    # Resize the original image
                    resized_image = cv2.resize(image, target_size)

                    # Save the preprocessed original resized image
                    original_resized_output = os.path.join(output_folder_name, f"original_{image_file}")
                    cv2.imwrite(original_resized_output, resized_image)

                    # Apply zoom augmentation and other transformations
                    for zoom_factor in zoom_factors:
                        # Apply zoom augmentation
                        zoomed_image = cv2.resize(image, (int(target_size[0] * zoom_factor), int(target_size[1] * zoom_factor)))

                        # Unique rotations
                        for i in range(num_rotations):
                            rotated_image = np.rot90(zoomed_image, k=i + 1)
                            rotated_output = os.path.join(output_folder_name, f"rotated_unique_{i}_{image_file}")
                            cv2.imwrite(rotated_output, rotated_image)

                            # Unique flips
                            for j in range(num_flips):
                                flipped_image = np.fliplr(rotated_image)
                                flipped_output = os.path.join(output_folder_name, f"flipped_unique_{i}_{j}_{image_file}")
                                cv2.imwrite(flipped_output, flipped_image)

                        # Unique color space transformations
                        for i in range(num_color_transformations):
                            color_transformed_image = zoomed_image[:, :, ::-1]  # Reverse color channels
                            color_output = os.path.join(output_folder_name, f"color_transformed_unique_{i}_{image_file}")
                            cv2.imwrite(color_output, color_transformed_image)

                        # Apply brightness/contrast adjustments and random noise to the zoomed image
                        for brightness in brightness_factors:
                            for contrast in contrast_factors:
                                for noise_level in noise_levels:
                                    augmented_image = zoomed_image.copy()

                                    # Apply brightness/contrast adjustments
                                    augmented_image = cv2.convertScaleAbs(augmented_image, alpha=contrast, beta=brightness * 255)

                                    # Add random noise
                                    noisy_image = random_noise(augmented_image, var=noise_level)
                                    noisy_image = img_as_ubyte(noisy_image)

                                    # Save the augmented image
                                    augmentation_suffix = f"b{int(brightness*100)}_c{int(contrast*100)}_n{int(noise_level*100)}"
                                    zoom_suffix = f"zoom{int(zoom_factor * 100)}"
                                    augmented_output = os.path.join(output_folder_name, f"augmented_{augmentation_suffix}_{zoom_suffix}_{image_file}")
                                    cv2.imwrite(augmented_output, noisy_image)

print("Augmentation for train data complete.")




#using keras
# import os
# import cv2
# import numpy as np
# from keras.preprocessing.image import ImageDataGenerator

# # Main directory name
# main_directory = "Dataset_alpha_keras"

# # List of subfolder names
# subfolder_names = ["test", "train", "validation"]

# # List of class names
# class_names = ['Cursor_Movement', 'Double_Click', 'Left_Click', 'Right_Click', 'Chrome_Open', 'Shutdown', 'Initiation',
#               'Volume_Increase', 'Volume_Decrease', 'Brightness_Increase', 'Brightness_Decrease', 'Screenshot', 'Scroll',
#               'Neutral', 'Nothing']

# # Desired size for resizing
# target_size = (600, 600)

# # Augmentation parameters
# batch_size = 32

# # Create an ImageDataGenerator with reduced augmentation settings
# datagen = ImageDataGenerator(
#     rotation_range=360,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     shear_range=0.2,
#     zoom_range=0.3,
#     horizontal_flip=True,
#     brightness_range=[0.2, 1.5],
#     fill_mode='nearest'
# )

# # Loop through subfolders
# for subfolder_name in subfolder_names:
#     if subfolder_name == "train":
#         for class_name in class_names:
#             input_folder_name = os.path.join(main_directory, subfolder_name, class_name)
#             output_folder_name = os.path.join(main_directory, subfolder_name, f"{class_name}_processed")

#             # Ensure the output folder exists
#             if not os.path.exists(output_folder_name):
#                 os.makedirs(output_folder_name)

#             # List all image files in the input folder
#             image_files = [f for f in os.listdir(input_folder_name) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

#             # Loop through images in the input folder
#             for image_file in image_files:
#                 image_path = os.path.join(input_folder_name, image_file)
#                 print(f"Processing image {image_file} in folder {input_folder_name}")

#                 # Load image
#                 image = cv2.imread(image_path)
#                 image = cv2.resize(image, target_size)

#                 # Expand the dimensions to fit the datagen requirement
#                 image = np.expand_dims(image, axis=0)

#                 # Generate augmented images and save them
#                 generator = datagen.flow(image, batch_size=1, save_to_dir=output_folder_name,
#                                          save_prefix='augmented_', save_format='jpeg')
                
#                 for _ in range(batch_size):
#                     batch = next(generator)
#                     augmented_image = batch[0].astype('uint8')

#                     augmented_output = os.path.join(output_folder_name, f"augmented_{image_file}")
#                     cv2.imwrite(augmented_output, augmented_image)

# print("Augmentation for train data complete.")
