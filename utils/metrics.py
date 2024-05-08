
import tensorflow as tf


class MeanIoU(tf.keras.metrics.MeanIoU):
    def __init__(self, num_classes, name='mean_io_u', dtype=None):
        super(MeanIoU, self).__init__(num_classes, name, dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # sparse code
        y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)
        return super(MeanIoU, self).update_state(y_true, y_pred, sample_weight)

