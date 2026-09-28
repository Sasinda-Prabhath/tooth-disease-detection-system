import keras


def build_caries_classifier(input_shape=(224, 224, 3), num_classes: int = 4):
    base = keras.applications.EfficientNetV2B0(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
        pooling="avg",
    )
    base.trainable = False

    inputs = keras.Input(shape=input_shape)
    x = keras.applications.efficientnet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="caries")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
