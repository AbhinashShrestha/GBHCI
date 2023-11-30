import tensorflow as tf
from tensorflow import keras
from keras.applications import EfficientNetV2M

model = EfficientNetV2M(include_top=False, weights='imagenet')

batch_size = 32
img_height = 380
img_width = 380
data_dir = "/content/drive/MyDrive/Colab Notebooks/Dataset_alpha"

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names
NUM_CLASSES = len(class_names)

from tensorflow.keras import layers

data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal_and_vertical",
                      input_shape=(img_height,
                                  img_width,
                                  3)),
    layers.RandomRotation(0.1,
                          fill_mode="reflect",
                          interpolation="bilinear",
                          fill_value=0.0),
    layers.RandomZoom(0.1,
                      fill_mode="reflect",
                      interpolation="bilinear",
                      fill_value=0.0),
    layers.RandomContrast(0.2),
    layers.RandomBrightness(0.2),
    layers.RandomTranslation(0.1, 0.1,
                             fill_mode="reflect",
                             interpolation="bilinear",
                             fill_value=0.0),
    layers.RandomCrop(img_height, img_width)
  ]
)

def preprocess(image, label):
    image = tf.map_fn(lambda img: tf.squeeze(data_augmentation(tf.expand_dims(img, 0)), axis=0), image)
    label = tf.one_hot(label, NUM_CLASSES)
    return image, label



train_ds = train_ds.map(preprocess)
val_ds = val_ds.map(preprocess)

def build_model(num_classes):
    inputs = layers.Input(shape=(img_width, img_height, 3))
    model = EfficientNetV2M(include_top=False, input_tensor=inputs, weights="imagenet")

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)

    # Compile
    model = keras.Model(inputs, outputs, name="EfficientNet")
    optimizer = keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model

model = build_model(num_classes=NUM_CLASSES)

epochs = 40
hist = model.fit(train_ds, epochs=epochs, validation_data=val_ds)
model.save('/content/drive/MyDrive/Colab Notebooks/V2M.h5')
