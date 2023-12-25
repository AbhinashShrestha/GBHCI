from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Assuming your data directory is organized with one subdirectory per class
class_names = ["Brightness_Decrease", "Brightness_Increase", "Chrome_Open", "Cursor_Movement", "Double_Click", "Initiation", "Left_Click", "Neutral", "Nothing", "Right_Click", "Screenshot", "Scroll", "Shutdown", "Volume_Decrease", "Volume_Increase"]


# Load the trained model
model = load_model('../Models/EfficientNetB6_without_onehot.keras')
img_height = 528
img_width= 528
# Load the image
img_path = '../classmates_test_images/fista.jpg'  # replace with the path to your image
img = image.load_img(img_path, target_size=(img_height, img_width))

# Preprocess the image
img_array = image.img_to_array(img)
img_batch = np.expand_dims(img_array, axis=0)

# Use the model to predict the class
predictions = model.predict(img_batch)

# The predictions are softmax probabilities, to get the class we find the index of the highest probability
predicted_class = np.argmax(predictions[0])
confidence = np.max(predictions[0])

# Print the class name and confidence
print('The predicted class is:', class_names[predicted_class])
print('Confidence:', confidence)

