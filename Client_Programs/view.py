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


import os
from PIL import Image, ImageDraw, ImageFont

def create_collage(directory, output_path, images_per_row):
    # Get all subdirectories
    subdirectories = sorted([os.path.join(directory, d) for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))])

    images = []
    labels = []

    for subdir in subdirectories:
        # Get all files in subdirectory
        files = [os.path.join(subdir, f) for f in os.listdir(subdir) if os.path.isfile(os.path.join(subdir, f))]

        # Filter out non-image files
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if image_files:
            # Open the first image and append to list
            images.append(Image.open(image_files[0]))
            # Append the label (subdirectory name)
            labels.append(os.path.basename(subdir))

    # Determine the size of the collage
    max_width = max(image.width for image in images)
    max_height = max(image.height for image in images)

    # Create a new image for the collage
    collage = Image.new('RGB', (max_width * images_per_row, max_height * ((len(images) + images_per_row - 1) // images_per_row)))

    # Draw object for adding labels
    draw = ImageDraw.Draw(collage)
    # Font for labels (you may need to specify the full path to the font file)
    # font = ImageFont.truetype("arial", 15)
    font = ImageFont.truetype("/usr/share/fonts/truetype/msttcorefonts/Arial.ttf", 25)


    for i, image in enumerate(images):
        # Calculate the position of the image in the collage
        x = (i % images_per_row) * max_width
        y = (i // images_per_row) * max_height

        # Paste the image into the collage
        collage.paste(image, (x, y))

        # Draw the label
        draw.text((x + 10, y), labels[i], fill='Black', font=font)

    # Save the collage
    collage.save(output_path)

# Call the function with your directory and output path
create_collage('Data/Dataset_alpha', 'gesture.tiff', 4)
