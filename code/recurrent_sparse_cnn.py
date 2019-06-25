from typing import List

import tensorflow as tf

from layers.recurrent_sparse_conv import RecurrentSparseConv2D
from layers.sparse_conv import SparseConv2D
from model_utils.learning_transfer import LearningTransfer


class RecurrentSparseConvNet(tf.keras.Sequential):
    """
    A Model performing sparse convolution and using a memory layer to process video inputs
    """
    
    def __init__(self, filters: List[int], kernel_sizes: List[int], l2_scales: List[float], memory_depths: List[int], fully_trainable:bool, name="RecurrentSparseConvNet", dtype=None, **kwargs):
        """
        Declares the layers of the model
        filters: List[int], the number of filters for each sparse conv layer
        kernel_sizes: List[int], specifying the height (and width) of square convolution window for each sparse conv layer
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the height and width.
        l2_scale: List[float], The scalar multiplier Tensor for each layer. 0.0 disables the regularizer.
        """
        assert len(filters) == len(kernel_sizes) and len(filters)==len(l2_scales) and len(filters) == len(memory_depths)

        self.max_recurrence_depth = max(memory_depths)
        depth = len(filters) #to account for the stacking layer
        # Create Layers
        # Register parameters
        layers: List[tf.keras.layers.Layer] = []

        #post recurrence layers
        for i in range(depth):
            is_last_layer: bool = i+1 == depth
            if memory_depths[i] == 0: #SparseConv2D layer
                layers.append(SparseConv2D(is_last_layer, filters=filters[i], kernel_size=kernel_sizes[i], l2_scale=l2_scales[i], trainable=fully_trainable, mask_depth=1, name=name + "-SparseConv2D_layer" + str(i + 1)))
            else: # recurrent layer
                layers.append(RecurrentSparseConv2D(memory_depth=memory_depths[i], is_last_layer=is_last_layer, filters=filters[i], kernel_size=kernel_sizes[i], l2_scale=l2_scales[i], trainable=True, name=name + "-RecurrentSparseConv2D_layer" + str(i + 1)))
            
        
        super().__init__(layers, name=name, **kwargs)

    def fit_sequence(self, train_sequence, epochs:int, callbacks:List, is_birecurrent: bool):
        assert len(train_sequence)> self.max_recurrence_depth, "The sequence cannot output enough data to initialize the model"
        # init the stacking layer
        self._init_recurrence(train_sequence)
        # Remove the first images from the sequence
        train_sequence.prepare_for_recurrent_training(self.max_recurrence_depth, is_birecurrent)
        # Training
        super().fit_generator(train_sequence, epochs=epochs, callbacks=callbacks, shuffle=False)


    def evaluate_sequence(self, eval_sequence, callbacks:List, is_birecurrent: bool, verbose=0):
        assert len(eval_sequence)> self.max_recurrence_depth, "The sequence cannot output enough data to initialize the model"
        
        # init the stacking layer
        self._init_recurrence(eval_sequence)
        # Remove the first images from the sequence
        eval_sequence.prepare_for_recurrent_training(self.max_recurrence_depth, is_birecurrent)
        # Training
        return super().evaluate_generator(eval_sequence, callbacks=callbacks, verbose=verbose)

    def load_weights(self, image_model: tf.keras.Model=None, weight_dir: str=None):
        assert not (image_model is None and weight_dir is None), "the weights must be in the image_model, the weights dir or distributed on both"
        if weight_dir is not None:
            super().load_weights(weight_dir)
        if image_model is not None:
            LearningTransfer.transfer_learning(image_model, self, alter_recurrent=weight_dir is None)

    def _init_recurrence(self, sequence, start: int=0):
        for i in range(self.max_recurrence_depth):
            super().predict(sequence.__getitem__(start + i)[0])

    def predict(self, sequence: tf.keras.utils.Sequence, is_birecurrent:bool, item:int=None):
        "Initialize the model with previous inputs in order to predict the requied item"
        item = self.max_recurrence_depth if item is None else item
        start_offset = (self.max_recurrence_depth//2) if is_birecurrent else self.max_recurrence_depth
        self._init_recurrence(sequence, item - start_offset)
        input_data, _ = sequence.__getitem__(item - start_offset + self.max_recurrence_depth)
        return super().predict_on_batch(input_data)

