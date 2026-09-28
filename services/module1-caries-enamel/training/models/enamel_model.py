import keras
import keras.layers as layers


def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    return layers.BatchNormalization()(x)


def build_enamel_unet(input_shape=(512, 512, 1), num_classes: int = 1):
    inputs = keras.Input(shape=input_shape)

    c1 = conv_block(inputs, 32)
    p1 = layers.MaxPooling2D()(c1)
    c2 = conv_block(p1, 64)
    p2 = layers.MaxPooling2D()(c2)
    c3 = conv_block(p2, 128)
    p3 = layers.MaxPooling2D()(c3)
    c4 = conv_block(p3, 256)
    p4 = layers.MaxPooling2D()(c4)
    bn = conv_block(p4, 512)

    u4 = layers.Conv2DTranspose(256, 2, strides=2, padding="same")(bn)
    u4 = conv_block(layers.concatenate([u4, c4]), 256)
    u3 = layers.Conv2DTranspose(128, 2, strides=2, padding="same")(u4)
    u3 = conv_block(layers.concatenate([u3, c3]), 128)
    u2 = layers.Conv2DTranspose(64, 2, strides=2, padding="same")(u3)
    u2 = conv_block(layers.concatenate([u2, c2]), 64)
    u1 = layers.Conv2DTranspose(32, 2, strides=2, padding="same")(u2)
    u1 = conv_block(layers.concatenate([u1, c1]), 32)

    outputs = layers.Conv2D(num_classes, 1, activation="sigmoid")(u1)
    model = keras.Model(inputs, outputs)

    def dice_loss(y_true, y_pred, smooth=1.0):
        import tensorflow as tf

        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        return 1 - (2.0 * intersection + smooth) / (
            tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
        )

    model.compile(optimizer=keras.optimizers.AdamW(learning_rate=1e-4), loss=dice_loss, metrics=["accuracy"])
    return model
