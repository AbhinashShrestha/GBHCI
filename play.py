# Import the required modules
from rembg import remove
from PIL import Image

# Define the input and output paths
input_path = 'classmates_test_images/yo.jpg'
output_path = 'your_image_segmented.jpg'

# Open the input image and remove the background
input = Image.open(input_path)
output = remove(input)

# Convert the output image to RGB mode
output = output.convert('RGB')

# Save the output image with a new name
output.save(output_path)

# Print a success message
print("Background removed successfully.")
