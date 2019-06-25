from typing import List

import tensorflow as tf
from matplotlib import pyplot
import numpy as np

from visualization_utils.color_conversion import Colorizer
from DataSequence.synthia_sequence import SynthiaSequence
from recurrent_sparse_cnn import RecurrentSparseConvNet

class ModelVisualizer:
    def __init__(self, models: List[tf.keras.Model], sequence: tf.keras.utils.Sequence):
        self.models = models
        self.sequence = sequence

    def visualize_models(self, item: int):
        input_data, groundtruth = self.sequence.__getitem__(item)
        # print(np.min(groundtruth), np.max(groundtruth))
        observability_mask = SynthiaSequence.sky_observation_matrix_sequence(groundtruth)
        predictions = [model.predict(self.sequence, is_birecurrent=True, item=item) for model in self.models]
        predictions = [tf.multiply(raw_prediction, observability_mask) for raw_prediction in predictions]
        f, axarr = pyplot.subplots(1, 2 + len(self.models))
        axarr[0].imshow(Colorizer.colorise_depth_map(groundtruth.reshape(
            (760, 1280)), is_sparse=True, min_val=0.0, max_val=655.35))
        axarr[1].imshow(Colorizer.colorise_depth_map(
            input_data[0, :, :, 0], is_sparse=True, min_val=0.0, max_val=655.35))
        for i in range(len(self.models)):
            axarr[i + 2].imshow(Colorizer.colorise_depth_map(tf.reshape(predictions[i], (760, 1280)), is_sparse=True, min_val=0.0, max_val=655.35))
        pyplot.show()
    def visualize_sparse_models(self, item:int):
        input_data, groundtruth = self.sequence.__getitem__(item)
        # print(np.min(groundtruth), np.max(groundtruth))
        observability_mask = SynthiaSequence.sky_observation_matrix_sequence(groundtruth)
        predictions = []
        sparsity_masks = []
        for model in self.models:
            model.layers[-1].is_last_layer = False
            if isinstance(model, RecurrentSparseConvNet):
                tensor = model.predict(self.sequence, is_birecurrent=True, item=item)[0]
            else:
                batch, _ = self.sequence.__getitem__(item)
                tensor = model.predict(batch)[0]
            features, sparsity_mask = tf.split(tensor, [1, 1], axis=2)
            predictions.append(tf.reshape(features, (760, 1280)))
            # sparsity_mask = tf.multiply(sparsity_mask, observability_mask)
            sparsity_masks.append(tf.reshape(sparsity_mask, (760, 1280)))
            model.layers[-1].is_last_layer = True
        # f, axarr = pyplot.subplots(1, 2 + len(self.models))
        f, axarr = pyplot.subplots(2, 2)
        axarr[0, 0].imshow(Colorizer.colorise_depth_map(groundtruth.reshape(
            (760, 1280)), is_sparse=True, min_val=0.0, max_val=655.35))
        axarr[0, 1].imshow(Colorizer.colorise_depth_map(
            input_data[0, :, :, 0], is_sparse=True, min_val=0.0, max_val=655.35))
        for i in range(len(self.models)):
            axarr[1, i].imshow(Colorizer.colorise_sparse_depth_map(predictions[i], sparsity_masks[i], min_val=0.0, max_val=655.35))
        pyplot.show()


