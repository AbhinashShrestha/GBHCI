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
    'H': 'Right_Click',
    'L': 'Left_Click',
    'N': 'N',
    'O': 'PowerPoint_Open',
    'P': 'Brightness_Increase',
    'Q': 'Brightness_Decrease',
    'S': 'Neutral',
    'V': 'VSCode_Open',
    'X': 'Volume_Down',
    'Y': 'Volume_Increase',
    'SPACE':'Restart'
}

# Specify the path of the main folder
main_folder_path = r'E:\MajorProject\Gesture based HCI\GBHCI\Data\asl_dataset'

# Get the list of all subfolders in the main folder
subfolders = [f.name for f in os.scandir(main_folder_path) if f.is_dir() and f.name.upper() in (key.upper() for key in folder_label_mapping.keys())]

# Count the total number of images
total_images = len(subfolders)

# Calculate the number of rows and columns for the collage
num_rows = total_images // 3 if total_images % 3 == 0 else total_images // 3 + 1
num_cols = min(total_images, 3)

# Create a new image for the collage with the adjusted size
collage = Image.new('RGB', (num_cols * 300, num_rows * 300))

# Create a draw object
draw = ImageDraw.Draw(collage)

# Load a font (this will depend on your system)
font = ImageFont.truetype("arial.ttf", 15)

x_offset = 0
y_offset = 0

# Iterate over each folder and add an image to the collage
for folder in subfolders:
    # Convert the folder name to uppercase for comparison
    folder_upper = folder.upper()

    # Check if the folder is in the list of classes
    if folder_upper in (key.upper() for key in folder_label_mapping.keys()):
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
collage.save('E:\MajorProject\Gesture based HCI\GBHCI\Gestures.png')
