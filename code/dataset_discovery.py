import os
from decimal import Decimal
from typing import Callable, List

import numpy as np
import rasterio as rio
from matplotlib import pyplot
from PIL import Image

from DbAccess.kitti_dao import KITTIDao
from DbAccess.synthia_dao import SynthiaDao
from KITTIExample import KITTIExample
from sdks.depth_devkit.read_depth import IOreader
from synthia_example import SynthiaExample


class DatasetDiscovery:
    @staticmethod
    def file_loop():
        """Opens depth files in the KITTI dataset and opens them in a different window
        """

        TRAIN_PATH : str = "datasets/KITTI/train/"
        VELODYNE_PATH : str = "/proj_depth/velodyne_raw/"
        train_sequences : List[str] = os.listdir(TRAIN_PATH)
        for seqence_name in train_sequences:
            print(seqence_name)
            current_dir : str = TRAIN_PATH + seqence_name + VELODYNE_PATH
            images : List[str] = os.listdir(current_dir)
            for image in images:
                print("    " + image)
                image_files : List[str] = os.listdir(current_dir + image)
                for data_file in image_files:
                    print("       " + data_file)
                    with rio.open(current_dir + image + "/" + data_file) as raster_data:
                        pyplot.imshow(raster_data.read(1), cmap='pink')
                        pyplot.show()


    @staticmethod
    def file_system_to_mongo_db():
        """One time script to ad data to DB"""
        TRAIN_PATH : str = "datasets/KITTI/val/"
        VELODYNE_PATH : str = "/proj_depth/velodyne_raw/image_02"
        train_sequences : List[str] = os.listdir(TRAIN_PATH)
        kitti_dao: KITTIDao = KITTIDao()
        for seqence_name in train_sequences:
            print(seqence_name)
            current_dir : str = TRAIN_PATH + seqence_name + VELODYNE_PATH
            image_files : List[str] = os.listdir(current_dir)
            for order, data_file in enumerate(image_files):
                imageObj : KITTIExample = KITTIExample("C:/Workspace/Thesis/" + current_dir + "/" + data_file)
                imageObj.is_training = False
                imageObj.is_raw_data = True
                imageObj.sequence = seqence_name
                imageObj.is_right_camera = False
                imageObj.order_in_sequence = order
                imageObj.timestamp = os.path.getctime(imageObj.path)
                kitti_dao.insert_KITTI_example(imageObj)
                print("image inserted: " + imageObj.path)

    @staticmethod
    def update_KITTI_path():
        kitti_dao: KITTIDao = KITTIDao()
        for record in kitti_dao.get_KITTI_all():
            record.path = record.path[20:]
            kitti_dao.update_KITTI_example(record)
    

    @staticmethod
    def synthia_file_system_to_mongo_db(sequence_name: str):
        """
        One time script to add dataset to DB
        """

        synthia_dao: SynthiaDao = SynthiaDao()
        BASE_PATH: str = "datasets/SYNTHIA/" + sequence_name + "/Depth/Stereo_Right/"
        views : List[str] = ["Omni_B", "Omni_F", "Omni_L", "Omni_R"]
        for i, view in enumerate(views):
            images_paths = os.listdir(BASE_PATH + view)
            for j, image_path in enumerate(images_paths):
                imageObj = SynthiaExample(BASE_PATH + view + "/" + image_path)
                imageObj.is_training = True
                imageObj.is_raw_data = True
                imageObj.sparsity_level = Decimal(1.0)
                imageObj.sequence = sequence_name
                imageObj.is_right_camera = True
                imageObj.camera_index = i
                imageObj.order_in_sequence = j
                imageObj.timestamp = os.path.getctime(imageObj.path)
                synthia_dao.insert_synthia_example(imageObj)
                print("image inserted: " + imageObj.path)
    
    @staticmethod
    def update_synthia_training_flag(sequence: str):
        synthia_dao : SynthiaDao = SynthiaDao()
        for camera_index in range(4):
            images = synthia_dao.get_Synthia_images_from_sequence(sequence, is_right_camera=True ,camera_index=camera_index)
            for image_obj in images:
                image_obj.is_training = False
                synthia_dao.update_Synthia_example(image_obj)
                print(image_obj.path)
