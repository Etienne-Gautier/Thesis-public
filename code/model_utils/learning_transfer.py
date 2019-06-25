import tensorflow as tf

from layers.recurrent_sparse_conv import RecurrentSparseConv2D
from layers.sparse_conv import SparseConv2D


class LearningTransfer:
    @staticmethod
    def transfer_learning(model_one: tf.keras.Sequential, model_two, alter_recurrent):
        """
        Transfers weights from model_one to model_two
        This implementation transfers only SparseConv2D weights,
        other layers are skipped.
        The implementation is highly reliant on the RecurrentSparseConvNet implementation
        The two models must have the same number of SparseConv2D layers
        """

        for i in range(len(model_one.layers)):
            if isinstance(model_one.layers[i], SparseConv2D):
                if isinstance(model_two.layers[i], RecurrentSparseConv2D) and alter_recurrent:
                    LearningTransfer.transfer_expand_layer(model_one.layers[i], model_two.layers[i])
                if isinstance(model_two.layers[i], SparseConv2D):
                    LearningTransfer.transfer_layer_learning(model_one.layers[i], model_two.layers[i])
                post_stacking_layer = False
    @staticmethod
    def transfer_layer_learning(layer_one: SparseConv2D, layer_two: SparseConv2D):
        layer_two.feature_Conv2D.kernel = tf.identity(layer_one.feature_Conv2D.kernel)
        layer_two.set_bias(tf.identity(layer_one.get_bias()))
    
    @staticmethod
    def transfer_expand_layer(layer_one: SparseConv2D, layer_two: RecurrentSparseConv2D):
        layer_two.set_kernel(tf.concat([tf.identity(layer_one.get_kernel()) for i in range(layer_two.memory_depth + 1)], 2))
        layer_two.set_bias(tf.identity(layer_one.get_bias()))
