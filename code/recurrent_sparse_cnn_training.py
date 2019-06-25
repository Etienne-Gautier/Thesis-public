import os
from datetime import datetime
from typing import List

import matplotlib.pyplot as pyplot
import numpy as np
import tensorflow as tf

from DataSequence.sampling_mode import SamplingMode
from DataSequence.synthia_sequence import SynthiaSequence
from layers.sparse_conv import SparseConv2D
from model_utils.execution_type import EExecutionType
from model_utils.input_completion_loss import SparseMAE, SparseRMSE
from model_utils.learning_transfer import LearningTransfer
from model_utils.model_saver import ModelSaver
from recurrent_sparse_cnn import RecurrentSparseConvNet
from visualization_utils.color_conversion import Colorizer
from visualization_utils.model_visualizer import ModelVisualizer

# Parameters
MIN_LAYER_POSITION: int = 4
MAX_LAYER_POSITION: int = 5
LAYER_NUMBER: int = 6
EXECUTION_TYPE: EExecutionType = EExecutionType.VIZ
SAMPLING_MODE: SamplingMode = SamplingMode.RANDOM
BIRECURRENT: bool = True
RETRAIN: bool = True
INDEPENDENT_MODEL: bool = False
# SPARSITY_RATE: float = 0.02
print("is birecurrent: ", BIRECURRENT)
SPARSITY_RATES: List[float] = [0.01, 0.02, 0.05, 0.1, 0.2]

# SPARSITY_RATES: List[float] = [1/2, 1/4, 1/8, 1/15, 1/25] 


VIZ_SEQ: int = 1
VIZ_IMG: int = 100


for SPARSITY_RATE in SPARSITY_RATES:
    # Step 1: create image model
    dir_name: str
    if SAMPLING_MODE == SamplingMode.RANDOM:
        dir_name = "/model-01-06-2019-new-loss-sparsity-rate-" + str(SPARSITY_RATE)
    else:
        dir_name = "/model-09-06-2019-lidar-sampling-sparsity-rate-" + str(SPARSITY_RATE)
    model_weight_dir: str = "../model_checkpoints" + dir_name + dir_name


    image_model: tf.keras.models.Sequential = tf.keras.models.Sequential([
    SparseConv2D(filters=16, kernel_size=11, name="cnn1"),
    SparseConv2D(filters=16, kernel_size=7, name="cnn2"),
    SparseConv2D(filters=16, kernel_size=5, name="cnn3"),
    SparseConv2D(filters=16, kernel_size=3, name="cnn4"),
    SparseConv2D(filters=16, kernel_size=3, name="cnn5"),
    SparseConv2D(is_last_layer=True, filters=1, kernel_size=1, name="cnn6")], name="SparseConvNet")

    image_model.build((5, 760, 1280, 2))
    image_model.load_weights(model_weight_dir)
    image_model.compile(optimizer=tf.optimizers.Adam(
        learning_rate=0.01), loss=SparseRMSE(), metrics=[SparseMAE()])

    sequences_path_root = "../train_path/"
    train_sequences_files = os.listdir(sequences_path_root)

    eval_sequences_path_root = "../eval_path/"
    eval_sequences_files: List[str] = os.listdir(eval_sequences_path_root)
    for rec_position in range(MIN_LAYER_POSITION, MAX_LAYER_POSITION):
        # Weights and logs savers
        logdir = "..\\logs-recurrent\\scalars\\" + \
            str(datetime.utcnow().timestamp())
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=logdir, write_graph=True)
        
        rec_dir_name: str
        if SAMPLING_MODE == SamplingMode.RANDOM:
            rec_dir_name = "/rec-model-01-06-2019-new-loss-birec-at-depth" +str(rec_position) + "-sparsity-rate-" + str(SPARSITY_RATE)
        else:
            str_sparsity_rate = str(SPARSITY_RATE) if SPARSITY_RATE>0.1 else "{:.4f}".format(SPARSITY_RATE)
            rec_dir_name = "/rec-model-09-06-2019-lidar-rec-at-depth" +str(rec_position) + "-sparsity-rate-" + str_sparsity_rate
        rec_model_weight_dir: str = "../model_checkpoints" + rec_dir_name + rec_dir_name
        model_saver = ModelSaver(rec_model_weight_dir)
        # Recurrent Model initialization
        model = RecurrentSparseConvNet(filters=[16,16,16,16,16,1], kernel_sizes=[11,7,5,3,3,1], l2_scales=[0.0] * LAYER_NUMBER, memory_depths=[2*(i==rec_position) for i in range(LAYER_NUMBER)], fully_trainable=RETRAIN)
        model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001), loss=SparseRMSE(), metrics=[SparseMAE()])
        model.build((1, 760, 1280, 2))
        if EXECUTION_TYPE == EExecutionType.TRAIN or EXECUTION_TYPE == EExecutionType.BOTH:
            print("-----------Training Start------------")
            model.load_weights(image_model if not INDEPENDENT_MODEL else None, weight_dir= rec_model_weight_dir if RETRAIN else None) # Weights must be init from the image model
            model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001), loss=SparseRMSE(), metrics=[SparseMAE()])
            for train_sequences_file in train_sequences_files:
                train_sequence = SynthiaSequence(SAMPLING_MODE, sparsity_rate=SPARSITY_RATE, file_path=sequences_path_root + train_sequences_file, max_batch_size=1)
                print("Recurrent layer at depth", rec_position, "sparsity rate", SPARSITY_RATE)
                print("Sequence", train_sequences_file, "length:", len(train_sequence))
                # fit_sequence is overridden to populate the recurrence weights
                model.fit_sequence(train_sequence, epochs=1, callbacks=[model_saver], is_birecurrent=BIRECURRENT)
            print("-----------Training: End------------")

        if EXECUTION_TYPE == EExecutionType.TEST or EXECUTION_TYPE == EExecutionType.VIZ:
            # weights are loaded from disk and transferred from image_model
            model.load_weights(image_model if not INDEPENDENT_MODEL else None, rec_model_weight_dir)
            model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001), loss=SparseRMSE(), metrics=[SparseMAE()])
        if EXECUTION_TYPE == EExecutionType.TEST or EXECUTION_TYPE == EExecutionType.BOTH:
            print("-----------Testing: Start------------")
            for eval_sequences_file in eval_sequences_files:
                eval_sequence = SynthiaSequence(SAMPLING_MODE, sparsity_rate=SPARSITY_RATE, file_path=eval_sequences_path_root +
                                                eval_sequences_file, max_batch_size=1)  # when using file based paths
                # evaluate_sequence is overridden to populate the recurrence weights
                print("Recurrent layer at depth ", rec_position, "sparsity rate", SPARSITY_RATE)
                print("Sequence: ", eval_sequences_file,
                    "length: ", len(eval_sequence))
                print(model.evaluate_sequence(eval_sequence,
                                            callbacks=[], is_birecurrent=BIRECURRENT, verbose=1))
            print("-----------Testing: End------------")
        if EXECUTION_TYPE == EExecutionType.VIZ:
            eval_sequences_file = eval_sequences_files[VIZ_SEQ]
            eval_sequence = SynthiaSequence(SAMPLING_MODE, sparsity_rate=SPARSITY_RATE, file_path=eval_sequences_path_root +
                                            eval_sequences_file, max_batch_size=1)  # when using file based paths
            visualizer = ModelVisualizer([image_model, model], eval_sequence)
            visualizer.visualize_sparse_models(VIZ_IMG)

