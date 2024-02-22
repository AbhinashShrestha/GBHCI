# import os
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg

# # Path to the Dataset directory
# dataset_dir = 'Data/Dataset'

# # Get a list of all directories in the Dataset directory
# dirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]

# # For each directory
# for d in dirs:
#     # Get a list of all files in the directory
#     files = os.listdir(os.path.join(dataset_dir, d))
    
#     # If there are any files in the directory
#     if files:
#         # Get the first file
#         file = files[0]
        
#         # Construct the full path to the file
#         file_path = os.path.join(dataset_dir, d, file)
        
#         # Read the image file
#         img = mpimg.imread(file_path)
        
#         # Create a new figure
#         plt.figure()
        
#         # Display the image
#         plt.imshow(img)
        
#         # Set the title of the figure to the directory name
#         plt.title(d)
        
#         # Show the figure
#         plt.show()


# import os
# from PIL import Image, ImageDraw, ImageFont

# def create_collage(directory, output_path, images_per_row):
#     # Get all subdirectories
#     subdirectories = sorted([os.path.join(directory, d) for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))])

#     images = []
#     labels = []

#     for subdir in subdirectories:
#         # Get all files in subdirectory
#         files = [os.path.join(subdir, f) for f in os.listdir(subdir) if os.path.isfile(os.path.join(subdir, f))]

#         # Filter out non-image files
#         image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

#         if image_files:
#             # Open the first image and append to list
#             images.append(Image.open(image_files[0]))
#             # Append the label (subdirectory name)
#             labels.append(os.path.basename(subdir))

#     # Determine the size of the collage
#     max_width = max(image.width for image in images)
#     max_height = max(image.height for image in images)

#     # Create a new image for the collage
#     collage = Image.new('RGB', (max_width * images_per_row, max_height * ((len(images) + images_per_row - 1) // images_per_row)))

#     # Draw object for adding labels
#     draw = ImageDraw.Draw(collage)
#     # Font for labels (you may need to specify the full path to the font file)
#     # font = ImageFont.truetype("arial", 15)
#     font = ImageFont.truetype("/usr/share/fonts/truetype/msttcorefonts/Arial.ttf", 25)


#     for i, image in enumerate(images):
#         # Calculate the position of the image in the collage
#         x = (i % images_per_row) * max_width
#         y = (i // images_per_row) * max_height

#         # Paste the image into the collage
#         collage.paste(image, (x, y))

#         # Draw the label
#         draw.text((x + 10, y), labels[i], fill='Black', font=font)

#     # Save the collage
#     collage.save(output_path)

# # Call the function with your directory and output path
# create_collage('Data/Dataset_alpha', 'gesture.tiff', 4)


from PIL import Image, ImageDraw, ImageFont
import os

# Define the mapping between folder names and labels
folder_label_mapping = {
    'A': 'Shutdown',
    'B': 'Scroll_Up',
    'Blank': 'Anomaly',
    'C': 'Chrome_Open',
    'E': 'Screenshot',
    'F': 'Scroll_Down',
    'G': 'Double Click',
    'H': 'H',
    'K': 'VSCode_Open',
    'L': 'Left_Click',
    'N': 'Right_Click',
    'O': 'PowerPoint_Open',
    'P': 'Brightness_Increase',
    'Q': 'Brightness_Decrease',
    'S': 'Neutral',
    'V': 'VSCode_Open',
    'X': 'Volume_Down',
    'Y': 'Volume_Increase',
    'space':'Restart'
}


# Specify the path of the main folder
main_folder_path = r'E:\MajorProject\Gesture based HCI\GBHCI\Data\asl_dataset'

# Create a new image for the collage
collage = Image.new('RGB', (900, 2100))  # Adjust the size of the collage to fit the images

# Create a draw object
draw = ImageDraw.Draw(collage)

# Load a font (this will depend on your system)
font = ImageFont.truetype("arial.ttf", 15)

x_offset = 0
y_offset = 0

# Get the list of all subfolders in the main folder
subfolders = [f.name for f in os.scandir(main_folder_path) if f.is_dir()]

# Iterate over each folder and add an image to the collage
for folder in subfolders:
    # Convert the folder name to uppercase for comparison
    folder_upper = folder.upper()

    # Check if the folder is in the list of classes
    if folder_upper in folder_label_mapping:
        # Get the list of images in the folder
        images = os.listdir(os.path.join(main_folder_path, folder))
        
        # Open the first image in the folder
        img = Image.open(os.path.join(main_folder_path, folder, images[0]))
        
        # Resize the image to fit in the collage
        img = img.resize((300, 300))  # Adjust the size of the images to fit in the collage
        
        # Add the image to the collage
        collage.paste(img, (x_offset, y_offset))
        
        # Draw the label on the image
        draw.text((x_offset, y_offset), folder_label_mapping[folder_upper], fill="white", font=font)
        
        # Update the offsets
        x_offset += 300
        if x_offset >= collage.width:
            x_offset = 0
            y_offset += 300

# Save the collage
collage.save('E:\MajorProject\Gesture based HCI\GBHCI\Visuals/Gestures.png')
