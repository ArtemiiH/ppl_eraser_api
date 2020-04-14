import tensorflow as tf
from tensorflow import keras

m = tf.keras.metrics.MeanIoU(num_classes=2)


def mean_iou(y_true, y_pred):
    metric = m(y_true, tf.round(y_pred))
    m.reset_states()
    return metric


def bce_dc_loss(y_true, y_pred):
    """ Binary crossentropy + Dice Coef loss. """

    def dice_coef(y_true, y_pred, smooth=1):
        """ Dice coef implementation. """
        numerator = 2 * \
            keras.backend.sum(keras.backend.abs(y_true * y_pred), axis=-1)
        denominator = (keras.backend.sum(keras.backend.square(
            y_true), -1) + keras.backend.sum(keras.backend.square(y_pred), -1))
        return (numerator + smooth) / (denominator + smooth)

    def dice_coef_loss(y_true, y_pred):
        return 1 - dice_coef(y_true, y_pred)

    return (keras.losses.binary_crossentropy(y_true, y_pred) +
            dice_coef_loss(y_true, y_pred))
