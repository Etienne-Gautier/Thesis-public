import tensorflow as tf

class SparseRMSE(tf.losses.Loss):
    def __init__(self):
        super().__init__(name="SparseRMSE")
    def call(self, y_true, y_pred, sample_Weight=None):
        assert y_true.shape == y_pred.shape, "Shape of prediction and groundtruth must be equal"
        max_depth = 655.35
        RMSE = 0
        batch_size = y_true.shape[0]
        for i in range(batch_size):
            observable_matrix = SparseRMSE.sky_observation_matrix_sequence(y_true[i])
            loss_matrix = tf.where(tf.equal(observable_matrix, 1.), x=tf.math.squared_difference(y_true[i],y_pred[i]), y=tf.zeros_like(y_true[i]))
            RMSE += tf.sqrt(tf.reduce_sum(loss_matrix)/tf.reduce_sum(observable_matrix))
        return RMSE
    
    @staticmethod
    def sky_observation_matrix_sequence(image, sky_value=655.35):
        return tf.where(tf.equal(image, sky_value), tf.zeros_like(image), tf.ones_like(image))

class SparseMAE(tf.losses.Loss):
    def __init__(self):
        super().__init__(name="SparseMAE")
    def call(self, y_true, y_pred, sample_Weight=None):
        assert y_true.shape == y_pred.shape, "Shape of prediction and groundtruth must be equal"
        max_depth = 655.35
        loss = 0
        batch_size = y_true.shape[0]
        for i in range(batch_size):
            observable_matrix = SparseMAE.sky_observation_matrix_sequence(y_true[i])
            loss_matrix = tf.where(tf.equal(observable_matrix, 1.), x=tf.math.abs(y_true[i] - y_pred[i]), y=tf.zeros_like(y_true[i]))
            loss += tf.reduce_sum(loss_matrix)/tf.reduce_sum(observable_matrix)
        return loss
    
    @staticmethod
    def sky_observation_matrix_sequence(image, sky_value=655.35):
        return tf.where(tf.equal(image, sky_value), tf.zeros_like(image), tf.ones_like(image))