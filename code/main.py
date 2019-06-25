from typing import List

import numpy as np
import rasterio as rio
import tensorflow as tf
from matplotlib import pyplot

from DataSequence.sampling_mode import SamplingMode
from DataSequence.synthia_sequence import SynthiaSequence
from visualization_utils.color_conversion import Colorizer

# Cloud compare: image viewer for point clouds
# PPTK python library for point cloud display


# DatasetDiscovery.update_synthia_training_flag("SYNTHIA-SEQS-04-FALL")


# Step 1: get the file path

# connector = KITTIDao()
# sequences: List[str] = connector.get_KITTI_train_sequences()
# records: List[KITTIExample] = connector.get_KITTI_images_from_sequence(
#     sequences[0], is_raw_data=True)
# # records: List[SynthiaExample] = SynthiaDao().get_all_Synthia_images_from_sequence("SYNTHIA-SEQS-04-FALL")
# record = records[0]
# for i in range(len(records)-1):
#     correl = np.sum(np.multiply(IOreader.depth_read(records[i].path), IOreader.depth_read(records[i + 1].path)))
#     print("correl: %f "%(correl))

# Step 2: display raw image

# depth_array = IOreader.depth_read_synthia('f:\\Etienne/datasets/SYNTHIA/SYNTHIA-SEQS-04-FOG/Depth/Stereo_Left/Omni_B/000604.png')
sequence = SynthiaSequence(SamplingMode.RANDOM, sparsity_rate=0.1 , file_path="../viz_path.txt", max_batch_size=1)

print(sequence.__getitem__(0)[0][0,:,:0].shape)
depth_array = sequence.__getitem__(0)[0][0,:,:,0]
pyplot.imshow(Colorizer.colorise_depth_map(depth_array, is_sparse=True, min_val=1.5, max_val=655.35))
pyplot.show()
# # Step 3: interpolate
# image_array: np.ndarray = IOreader.depth_read(record.path)
# # pyplot.imshow(image_array, cmap='pink')
# # pyplot.show()
# image_array_interpolated = Interpolation.LIDAR_bicubic_interpolate(image_array, 3)


# # Step 4: print interpolated image
# pyplot.imshow(image_array_interpolated, cmap='pink')
# pyplot.show()
