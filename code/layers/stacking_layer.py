import tensorflow as tf


class StackingLayer(tf.keras.layers.Layer):
    """
    A layer using memory to return a stack (by channels) of the current and previous inputs
    """

    def __init__(self, memory_depth:int=2, name: str=None, dtype=None, *args, **kwargs):
        super().__init__(trainable=False, name=name, dtype=dtype, dynamic=True, **kwargs)
        self.memory_depth: int = memory_depth

    def build(self, input_shape):
        super().build(input_shape)
        self.tensor_memory: [tf.Tensor] = [None] * self.memory_depth
        self.mask_memory: [tf.Tensor] = [None] * self.memory_depth

    def call(self, tensor):
        b, h, w, c = tensor.get_shape()
        assert b == 1, "Can only support batches of size one"
        # Step 1: init when using a new sequence
        tensor_memory_channels, mask_memory_channels = tf.split(tensor, [c-1, 1], axis=3)
        if self.tensor_memory[0] is None:
            for i in range(self.memory_depth):
                self.tensor_memory[i] = tf.constant(tensor_memory_channels, name="initial-memory" + str(i))
                self.mask_memory[i] = tf.constant(mask_memory_channels, name="mask-initial-memory" + str(i))
        #  Step 2: Create output
        out = tf.concat(self.tensor_memory + [tensor_memory_channels] + self.mask_memory + [mask_memory_channels], -1, name="concatenator")

        # Step3: Shift memory by one
        for i in range(self.memory_depth - 2, -1, -1):
            self.tensor_memory[i] = self.tensor_memory[i+1]
            self.mask_memory[i] = self.mask_memory[i+1]
        # Step 4: Save current step in memory
        self.tensor_memory[-1] = tensor_memory_channels
        self.mask_memory[-1] = mask_memory_channels
        return out

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[0], input_shape[1], input_shape[2], (self.memory_depth + 1) * input_shape[3]])
