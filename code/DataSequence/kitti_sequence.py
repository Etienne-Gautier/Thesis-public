import random
from typing import List

import numpy as np

from DbAccess.kitti_dao import KITTIDao
from KITTIExample import KITTIExample
from sdks.depth_devkit.read_depth import IOreader


class KITTISequence:
    """
    Gathers sequences of train and eval examples
    """
    height = 375
    width = 1242
    input_channels = 2

    def __init__(self, connector: KITTIDao):
        self.connector = connector
    
    def generate_train_data(self):
        """
        Sequence of tuples (train_input, train_expected_output) for the KITTI dataset
        """
        
        # Step 1: Get the list of sequences
        sequences: List[str] = self.connector.get_KITTI_train_sequences()
        # Step 2: Yield a batch for each sequence
        yield from generate_batch_for_sequences(sequences)

    def generate_eval_data(self):
        # Step 1: Get the list of sequences
        sequences: List[str] = self.connector.get_KITTI_eval_sequences()
        # Step 2: Yield a batch for each sequence
        yield from generate_batch_for_sequences(sequences)

    def generate_batch_for_sequences(self, sequences: List[str]):
        sequence: str
        # Step 1: Open the sequences 1 by 1 in batch
        for sequence in sequences:
            # Step 1.1: Find the files records in the sequence
            input_files: List[KITTIExample] = self.connector.get_KITTI_images_from_sequence(sequence, True)
            output_files: List[KITTIExample] = self.connector.get_KITTI_images_from_sequence(sequence, False)
            assert len(input_files) == len(output_files), "There must be the same number of inputs and outputs"
            input_batch : np.ndarray = np.zeros((len(input_files), KITTISequence.height, KITTISequence.width, KITTISequence.input_channels)) # clear the batch
            output_batch : np.ndarray = np.zeros((len(input_files), KITTISequence.height, KITTISequence.width, 1)) # clear the batch
            # Step 1.2: Open the files
            for index in range(len(input_files)):
                input_batch[index, :, :, 0] = IOreader.depth_read(input_files[index].path)
                # TODO(etienne): add the observation channel
                output_batch[index, :, :, 0] = IOreader.depth_read(output_files[index].path)
            yield input_batch, output_batch
        
        raise Exception("Sequence sequence has ended")
