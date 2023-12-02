import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
img_path = 'Dataset_alpha/Brightness_Increase/image_1.jpg'  # replace with the path to your image
img = cv2.imread(img_path)

# Create the background subtractor
backSub = cv2.createBackgroundSubtractorMOG2()

# Apply background subtraction
fgMask = backSub.apply(img)

# Apply a binary threshold to the foreground mask
_, fgMask = cv2.threshold(fgMask, 244, 255, cv2.THRESH_BINARY)

# Bitwise-and the mask with the original image
img_segmented = cv2.bitwise_and(img, img, mask=fgMask)

# Convert the images from BGR to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_segmented = cv2.cvtColor(img_segmented, cv2.COLOR_BGR2RGB)

# Display the images side by side
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_segmented)
plt.title('Segmented Image')
plt.axis('off')

plt.show()

