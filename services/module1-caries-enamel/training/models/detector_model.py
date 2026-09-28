import keras


def build_caries_detector(input_shape=(640, 640, 3), num_classes: int = 6):
    inputs = keras.Input(shape=input_shape)

    base = keras.applications.MobileNetV3Small(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
        pooling="avg",
    )

    x = base(inputs)
    box = keras.layers.Dense(4, name="box")(x)
    cls = keras.layers.Dense(num_classes, activation="softmax", name="class")(x)

    model = keras.Model(inputs, [box, cls])
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss={"box": "mse", "class": "sparse_categorical_crossentropy"},
        metrics={"class": "accuracy"},
    )
    return model
