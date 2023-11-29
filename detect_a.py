from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import keras 
import tensorflow as tf
# dimensions of our images
img_width, img_height = 380, 380

# load the model we saved
model = load_model('basic.h5')
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


class_names=['Brightness_Decrease', 'Brightness_Increase', 'Chrome_Open', 'Cursor_Movement', 'Double_Click', 'Initiation', 'Left_Click', 'Neutral', 'Nothing', 'Right_Click', 'Screenshot', 'Scroll', 'Shutdown', 'Volume_Decrease', 'Volume_Increase']

img_height=180
img_width=180
# img = keras.preprocessing.image.load_img(
#     'shashin.jpg', target_size=(img_height, img_width)
# )
img = keras.preprocessing.image.load_img(
    'as.jpg', target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)