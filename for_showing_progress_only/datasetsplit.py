import os
import random
import shutil

# Paths
input_folder = "Dataset_alpha"
output_folder = "Dataset_alpha_split"

# Get list of new class names from input folder
classes = [d for d in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, d))]

num_images_per_class = 100
train_ratio = 0.7
validation_ratio = 0.15
test_ratio = 0.15

# Create output directories
for split in ['train', 'validation', 'test']:
    split_folder = os.path.join(output_folder, split)
    os.makedirs(split_folder, exist_ok=True)
    for class_name in classes:
        class_folder = os.path.join(split_folder, class_name)
        os.makedirs(class_folder, exist_ok=True)

# Shuffle and distribute images
for class_name in classes:
    class_images = [f for f in os.listdir(os.path.join(input_folder, class_name)) if f.endswith('.jpg')]
    random.shuffle(class_images)

    num_total = len(class_images)
    num_train = int(num_total * train_ratio)
    num_validation = int(num_total * validation_ratio)
    num_test = num_total - num_train - num_validation

    train_images = class_images[:num_train]
    validation_images = class_images[num_train:num_train + num_validation]
    test_images = class_images[num_train + num_validation:num_train + num_validation + num_test]

    for image in train_images:
        src = os.path.join(input_folder, class_name, image)
        dst = os.path.join(output_folder, 'train', class_name, image)
        shutil.copy(src, dst)

    for image in validation_images:
        src = os.path.join(input_folder, class_name, image)
        dst = os.path.join(output_folder, 'validation', class_name, image)
        shutil.copy(src, dst)

    for image in test_images:
        src = os.path.join(input_folder, class_name, image)
        dst = os.path.join(output_folder, 'test', class_name, image)
        shutil.copy(src, dst)

print("Dataset split and organized successfully.")
