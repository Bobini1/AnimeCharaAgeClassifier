import os
import numpy as np
import matplotlib.pyplot as plt

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
        r'M:\Gelbooru Data\WorkingBMPColor',
        validation_split=0.2,
        subset="training",
        color_mode='rgb',
        seed=123,
        image_size=(224, 224),
        batch_size=32)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        r'M:\Gelbooru Data\WorkingBMPColor',
        validation_split=0.2,
        subset="validation",
        color_mode='rgb',
        seed=123,
        image_size=(224, 224),
        batch_size=32)
    train_ds = train_ds.cache().prefetch(buffer_size=-1)
    val_ds = val_ds.cache().prefetch(buffer_size=-1)

    base = tf.keras.applications.VGG19(
        include_top=False,
        weights="imagenet",
        input_shape=None,
    )

    base.trainable = False

    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
        ]
    )
    inputs = tf.keras.Input(shape=(224, 224, 3))
    # x = inputs
    x = data_augmentation(inputs)

    # inputs = tf.keras.layers.experimental.preprocessing.RandomFlip(mode="horizontal", seed=None)

    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs, outputs)

    print(model.summary())

    # create model
    # model = tf.keras.models.Sequential([
    #     tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=5e-4)),
    #     tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=5e-4)),
    #     tf.keras.layers.Dense(1)
    # ])

    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    # Compile model
    model.compile(optimizer=opt,
                  loss=loss_fn,
                  metrics=['accuracy'])
    model.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=[tensorboard_callback], shuffle=True)
    predictions = np.array([])
    labels = np.array([])
    xs = np.empty((0, 224, 224, 3))
    for x, y in val_ds:
        xs = np.concatenate([xs, x.numpy() / 255])
        predictions = np.concatenate([predictions, np.round(np.clip((model.predict(x)).flatten(), 0, 1))])
        labels = np.concatenate([labels, y])

    print(tf.math.confusion_matrix(labels=labels, predictions=predictions).numpy())
    for x in xs[predictions != labels]:
        plt.imshow(x)
        plt.show()


if __name__ == '__main__':
    main()
