import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import datetime


def main():
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        r'M:\Gelbooru Data\Working_Set',
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(256, 256),
        batch_size=32)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        r'M:\Gelbooru Data\Working_Set',
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(256, 256),
        batch_size=32)
    train_ds = train_ds.cache().prefetch(buffer_size=-1)
    val_ds = val_ds.cache().prefetch(buffer_size=-1)


    # create model
    model = tf.keras.models.Sequential([
        tf.keras.layers.experimental.preprocessing.Rescaling(scale=1./127.5, offset=-1),
        tf.keras.layers.Conv2D(32, 3, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(256, 3, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.Dense(1)
    ])
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    # Compile model
    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])
    model.fit(train_ds, validation_data=val_ds, epochs=5, callbacks=[tensorboard_callback], shuffle=True)


if __name__ == '__main__':
    main()
