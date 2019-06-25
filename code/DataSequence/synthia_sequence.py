import math
import random
from typing import List

import numpy as np
import tensorflow as tf

from DataSequence.sequence_utils import SequenceUtils
from DataSequence.sampling_mode import SamplingMode
from DbAccess.synthia_dao import SynthiaDao
from sdks.depth_devkit.read_depth import IOreader
from synthia_example import SynthiaExample


class SynthiaSequence(tf.keras.utils.Sequence):
    """
    Gathers sequences of train and eval examples
    """

    Synthia_height = 760
    Synthia_width = 1280
    Synthia_input_channels = 2

    def __init__(self, sampling_mode: SamplingMode, sparsity_rate: float, is_train_sequence: bool=True, file_path: str=None, max_batch_size: int=5):
        self.connector = SynthiaDao()
        self.is_train = is_train_sequence
        self.max_batch_size = max_batch_size
        self.image_paths : List[str]
        self.sampling_mode = sampling_mode
        if file_path is not None:
            with open(file_path, 'r') as f:
                self.image_paths = [path for path in map(lambda x: x[:-1], f.readlines())]
        else:
            self.sequences: List[str]
            if(self.is_train):
                self.sequences = self.connector.get_Synthia_train_sequences()
            else:
                self.sequences = self.connector.get_Synthia_eval_sequences()
            
            self.image_paths = []

            for seq in self.sequences:
                self.image_paths.extend(map(lambda x: x.path, self.connector.get_all_Synthia_images_from_sequence(seq)))
        
        self.base_observation_matrix = self.get_sampling_matrix(sparsity_rate * 0.9)
        self.start_offset = 0
        self.groundtruth_offset = 0 #can be negative
    
        

    def __len__(self):
        return math.ceil((len(self.image_paths) - self.start_offset) / self.max_batch_size)

    def __getitem__(self, idx):
        # Step 1: Find the files records in the sequence
        start_index: int = idx * self.max_batch_size + self.start_offset
        length = min(self.max_batch_size, len(self.image_paths) - start_index)
        # Step 2: Initialize the batch variable
        input_batch : np.ndarray = np.zeros((length,
            SynthiaSequence.Synthia_height,
            SynthiaSequence.Synthia_width,
            SynthiaSequence.Synthia_input_channels)) 
        output_batch : np.ndarray = np.zeros((length, SynthiaSequence.Synthia_height, SynthiaSequence.Synthia_width, 1))
        # Step 3: Load values
        for index in range(length):
            input_image = IOreader.depth_read_synthia(SequenceUtils.DATASTSET_LOCATION + self.image_paths[start_index + index])
            # The observation matrix is the combination of 3 things: the sensor's sampling strategy, presence of an obstacle (the sky doesn't return a signal), the reflectivity of the material
            observation_matrix_at_frame = np.multiply(SynthiaSequence.sky_observation_matrix_sequence(input_image), np.multiply(self.base_observation_matrix, SynthiaSequence.get_random_matrix(1))) #warning
            input_batch[index, :, :, 0] = np.multiply(input_image, observation_matrix_at_frame)
            input_batch[index, :, :, 1] = observation_matrix_at_frame
            output_batch[index,:, :, 0] = IOreader.depth_read_synthia(SequenceUtils.DATASTSET_LOCATION + self.image_paths[start_index + index + self.groundtruth_offset])
            
        return (input_batch, output_batch)


    def prepare_for_recurrent_training(self, memory_depth: int, is_birecurrent: bool):
        if is_birecurrent:
            assert memory_depth % 2 == 0, "The memory depth must be symetric"
            self.groundtruth_offset = -1 * (memory_depth//2)
        self.start_offset = memory_depth
    
    def restore_sequence(self):
        self.groundtruth_offset = 0
        self.start_offset = 0
    
    @staticmethod
    def get_random_matrix(sparsity_rate):
        return SequenceUtils.generate_observation_matrix(SynthiaSequence.Synthia_height, SynthiaSequence.Synthia_width, sparsity_rate)

    def get_sampling_matrix(self, sparsity_rate):
        if self.sampling_mode == SamplingMode.RANDOM:
            return SynthiaSequence.get_random_matrix(sparsity_rate)
        elif self.sampling_mode == SamplingMode.LIDAR:
            step = round(1/sparsity_rate)
            return SequenceUtils.generate_lidar_observation_matrix(SynthiaSequence.Synthia_height, SynthiaSequence.Synthia_width, step)
        else:
            raise Exception("Unsupported sampling mode")
    
    @staticmethod
    def sky_observation_matrix_sequence(image, sky_value=655.3):
        return np.where(image>sky_value, 0, 1)
