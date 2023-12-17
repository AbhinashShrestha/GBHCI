import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
from keras.applications import EfficientNetV2L

# Constants
BATCH_SIZE = 16
IMG_HEIGHT = 380
IMG_WIDTH = 380
DATA_DIR = "/content/drive/MyDrive/Colab Notebooks/Dataset"

# Data augmentation
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        layers.RandomRotation(0.1, fill_mode="reflect", interpolation="bilinear", fill_value=0.0),
        layers.RandomZoom(0.1, fill_mode="reflect", interpolation="bilinear", fill_value=0.0),
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.2),
        layers.RandomTranslation(0.1, 0.1, fill_mode="reflect", interpolation="bilinear", fill_value=0.0),
        layers.RandomCrop(IMG_HEIGHT, IMG_WIDTH)
    ]
)

# Preprocessing
def preprocess(image, label, num_classes):
    image = tf.map_fn(lambda img: tf.squeeze(data_augmentation(tf.expand_dims(img, 0)), axis=0), image)
    label = tf.one_hot(label, num_classes)
    return image, label

# Model building
def build_model(num_classes):
    inputs = layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3))
    model = EfficientNetV2L(include_top=False, input_tensor=inputs, weights="imagenet")

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="pred", kernel_regularizer=regularizers.l2(0.01))(x)  # L2 regularization

    # Compile
    model = keras.Model(inputs, outputs, name="EfficientNet")
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-3, decay_steps=10000, decay_rate=0.9)
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# Unfreeze model
def unfreeze_model(model):
    # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
    for layer in model.layers[-20:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    optimizer = keras.optimizers.Adam(learning_rate=1e-5)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

# Load data
def load_data():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE)

    return train_ds, val_ds

def main():
    train_ds, val_ds = load_data()
    class_names = train_ds.class_names
    num_classes = len(class_names)

    train_ds = train_ds.map(lambda x, y: preprocess(x, y, num_classes))
    val_ds = val_ds.map(lambda x, y: preprocess(x, y, num_classes))

    model = build_model(num_classes=num_classes)

    # Create a callback for early stopping
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    epochs = 15

    # Continue training
    hist = model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=[early_stopping_callback])

    unfreeze_model(model)

    epochs = 5
    hist = model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=[early_stopping_callback])

    # Save the entire model as a single file
    model.save('/content/drive/MyDrive/Colab Notebooks/EV2L_asl_ko.h5')

    # Plot training & validation accuracy values
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.show()

if __name__ == "__main__":
    main()
