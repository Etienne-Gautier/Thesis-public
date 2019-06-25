import tensorflow as tf

class SparseConv2D(tf.keras.layers.Layer):
    """
    A layer performing 2D convolution on a sparse input given a sparsity map
    """
    
    def __init__(self, is_last_layer: bool = False,filters=32,kernel_size=3,strides=1,l2_scale=0.0, mask_depth=1, trainable=True, name=None, dtype=None, **kwargs):
        """
        Declares the (input size independant) objects of the Sparse conv layer
        filters: Integer, the dimensionality of the output space (i.e. the number
            of filters in the convolution).
        kernel_size: An integer or tuple/list of 2 integers, specifying the
            height and width of the 2D convolution window.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the height and width.
        l2_scale: float, A scalar multiplier Tensor. 0.0 disables the regularizer.
        """
        super().__init__(trainable=trainable, name=name, dtype=dtype, dynamic=True, **kwargs)
        # Register parameters
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.l2_scale = l2_scale
        self.is_last_layer = is_last_layer
        self.mask_depth = mask_depth

        # Create Operation objects
        self.regularizer = tf.keras.regularizers.l2(self.l2_scale)
        self.feature_Conv2D = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=self.kernel_size, strides=(self.strides, self.strides), trainable=trainable, use_bias=False, padding="same",kernel_regularizer=self.regularizer, name=self.name + "feature_conv")
        self.mask_Conv2D = tf.keras.layers.Conv2D(filters=1, kernel_size=self.kernel_size, strides=(self.strides, self.strides), kernel_initializer=tf.ones_initializer(), trainable=False, use_bias=False, padding="same", name=self.name + "mask_conv")
        self.mask_MaxPooling2D = tf.keras.layers.MaxPooling2D(strides = self.strides, pool_size=self.kernel_size, padding="same", name=self.name + "mask_max_pooling")
        self.b = tf.Variable(lambda: tf.constant(0.01, shape=[self.filters]),trainable=trainable, name=self.name + "bias")


    def build(self, input_shape):
        """ Build the shape dependant objects
        """
        assert len(input_shape) == 4, "input shape is not supported"
        super().build(input_shape)
        # Build the keras layer
        self.feature_Conv2D.build((input_shape[0], input_shape[1], input_shape[2], input_shape[3] - self.mask_depth))
        self.mask_Conv2D.build((input_shape[0], input_shape[1], input_shape[2], self.mask_depth))
        self.mask_MaxPooling2D.build((input_shape[0], input_shape[1], input_shape[2], self.mask_depth))
    
    def call(self, tensor):
        """Arguments
        tensor: Tensor input. Must contain the binary mask in its last channel
        binary_mask: Tensor, a mask with the same size as tensor, channel size = 1
        
        
        Returns:
        Output tensor, binary mask.
        """
        # Step 1: Break down the tensor in features and mask
        CHANNEL_AXIS = 3
        b,h,w,c = tensor.get_shape()
        assert (c-self.mask_depth) % self.mask_depth == 0, "The number of effective input channels must be a multiple of the mask depth"
        tensor, binary_mask = tf.split(tensor, [c- self.mask_depth, self.mask_depth], axis=CHANNEL_AXIS)

        stacks_size = (c-self.mask_depth) / self.mask_depth

        # Step 2: broadcast the mask to the required shape then multiply
        if self.mask_depth>1:
            mask_scaled = tf.concat([tf.broadcast_to(sub_mask, tf.TensorShape([b, h, w, stacks_size])) for sub_mask in tf.split(binary_mask, self.mask_depth, axis=CHANNEL_AXIS)], axis=CHANNEL_AXIS)
        else:
            mask_scaled = binary_mask
        features = tf.multiply(tensor, mask_scaled)
        
        features = self.feature_Conv2D(features)
        
        norm = self.mask_Conv2D(binary_mask)
        norm = tf.where(tf.equal(norm,0),tf.zeros_like(norm),tf.math.reciprocal(norm))

        feature = tf.keras.activations.relu(tf.multiply(features,norm) + self.b)
        mask = self.mask_MaxPooling2D(binary_mask)
        if self.mask_depth>1:
            # mask = tf.reduce_mean(mask, axis=CHANNEL_AXIS, keepdims=True)# is that what was really intended, see next line as an alternative
            mask = tf.reduce_max(mask, axis=CHANNEL_AXIS, keepdims=True)
        if self.is_last_layer :
            return feature
        else:
            return tf.concat([feature, mask], -1, name="layer_output") # Add the mask as a channel on the last dimension

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[0], input_shape[1], input_shape[2], 1 if(self.is_last_layer) else (self.filters + 1)])

    def set_kernel(self, value: tf.Tensor) -> None:
        self.feature_Conv2D.kernel.assign(value)
    
    def set_bias(self, value: tf.Tensor)-> None:
        self.b.assign(value)
    
    def get_kernel(self) -> tf.Tensor:
        return self.feature_Conv2D.kernel
    
    def get_bias(self) -> tf.Tensor:
        return self.b
