import tensorflow as tf

from .sparse_conv import SparseConv2D
from .stacking_layer import StackingLayer


class RecurrentSparseConv2D(tf.keras.layers.Layer):
    """ A recurrentSparseConv2D is a layer that applies sparse convolution to the current input as well as a memory of past inputs
    """

    def __init__(self, memory_depth:int=2, is_last_layer: bool = False, filters=32, kernel_size=3, strides=1, l2_scale=0.0, trainable=True, name:str="RecurrentSparseConv2D", dtype=None, **kwargs):
        super().__init__(trainable=trainable, name=name, dtype=dtype, dynamic=True, **kwargs)
        self.stacking_layer = StackingLayer(memory_depth=memory_depth, name=name + "-Memory-")
        self.sparse_conv = SparseConv2D(is_last_layer=is_last_layer, filters=filters, kernel_size=kernel_size, l2_scale=l2_scale, mask_depth=memory_depth + 1, trainable=trainable, name=name + "-SparseConv2D-")
        self.memory_depth = memory_depth
    
    def build(self, input_shape):
        super().build(input_shape)
        self.stacking_layer.build(input_shape)
        self.sparse_conv.build(self.stacking_layer.compute_output_shape(input_shape))
    
    def call(self, tensor):
        return self.sparse_conv(self.stacking_layer(tensor))

    def set_kernel(self, value: tf.Tensor):
        self.sparse_conv.set_kernel(value)
    
    def set_bias(self, value: tf.Tensor):
        self.sparse_conv.set_bias(value)
    
    def compute_output_shape(self, input_shape):
        return self.sparse_conv.compute_output_shape(self.stacking_layer.compute_output_shape(input_shape))
