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
from model_utils.model_saver import ModelSaver
from visualization_utils.color_conversion import Colorizer
from visualization_utils.model_visualizer import ModelVisualizer


SAMPLING_MODE: SamplingMode = SamplingMode.LIDAR
execution_type = EExecutionType.TEST
SPARSITY_RATES: List[float] = [1/2, 1/4, 1/8, 1/15, 1/25]

VIZ_SEQ: int = 1

VIZ_IMG: int = 100


logdir="..\\logs\\scalars\\" + str(datetime.utcnow().timestamp())
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)


for SPARSITY_RATE in SPARSITY_RATES:
  dir_name: str = "/model-09-06-2019-lidar-sampling-sparsity-rate-" + str(SPARSITY_RATE)
  model_weight_dir: str = "../model_checkpoints" + dir_name + dir_name

  model_saver = ModelSaver(model_weight_dir)

  model = tf.keras.models.Sequential([
    SparseConv2D(filters=16, kernel_size=11, name="cnn1"),
    SparseConv2D(filters=16, kernel_size=7, name="cnn2"),
    SparseConv2D(filters=16, kernel_size=5, name="cnn3"),
    SparseConv2D(filters=16, kernel_size=3, name="cnn4"),
    SparseConv2D(filters=16, kernel_size=3, name="cnn5"),
    SparseConv2D(is_last_layer = True, filters=1, kernel_size=1, name="cnn6")
  ], name="SparseConvNet")


  model.build((5, 760, 1280, 2))
  if execution_type == EExecutionType.TEST or execution_type == EExecutionType.VIZ:
    model.load_weights(model_weight_dir)
  model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001), loss=SparseRMSE(), metrics=[SparseMAE()])

  # model.summary()
  print("-------------------------------Sparsity rate: " + str(SPARSITY_RATE) + "----------------------------------------")
  if execution_type == EExecutionType.TRAIN or execution_type == EExecutionType.BOTH:
    # train_sequence = SynthiaSequence(is_train_sequence=True, max_batch_size=2) # when a database is available
    train_sequence = SynthiaSequence(SAMPLING_MODE, sparsity_rate=SPARSITY_RATE, file_path="f:\Etienne/datasets/SYNTHIA/train_paths.txt", max_batch_size=2) # when using file based paths
    model.fit_generator(train_sequence, epochs=2, callbacks=[model_saver])
  if execution_type == EExecutionType.TEST or execution_type == EExecutionType.BOTH:
    sequences_path_root = "../eval_path/"
    eval_sequences_files = os.listdir(sequences_path_root)
    for eval_sequences_file in eval_sequences_files:
        eval_sequence = SynthiaSequence(SAMPLING_MODE, sparsity_rate=SPARSITY_RATE, file_path=sequences_path_root + eval_sequences_file, max_batch_size=1) # when using file based paths
        # evaluate_generator is overriden to populate the recurrence weights
        print("Sparse CNN eval on " + eval_sequences_file + " length: " + str(len(eval_sequence)))
        print(model.evaluate_generator(eval_sequence, callbacks=[], verbose=1))
  if execution_type == EExecutionType.VIZ:
    sequences_path_root = "../eval_path/"
    eval_sequences_files = os.listdir(sequences_path_root)
    eval_sequences_file = eval_sequences_files[VIZ_SEQ]
    eval_sequence = SynthiaSequence(SAMPLING_MODE, sparsity_rate=SPARSITY_RATE, file_path=sequences_path_root + eval_sequences_file, max_batch_size=1)
    visualizer = ModelVisualizer([model], eval_sequence)
    visualizer.visualize_models(VIZ_IMG)
