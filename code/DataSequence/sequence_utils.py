import numpy as np
import random

class SequenceUtils:
    DATASTSET_LOCATION = "../"
    @staticmethod
    def generate_observation_matrix(height: int, width: int, sparsity_rate: float):
        """
        Returns a matrix of 0 and 1 where 
        """
        mat : np.ndarray = np.zeros(shape=(height, width))
        num_points : int = int(height * width * sparsity_rate)
        for index in random.sample(range(height * width), num_points):
            mat[index // width][index % width] = 1
        return mat
    
    @staticmethod
    def generate_lidar_observation_matrix(height: int, width: int, step: int):
        mat : np.ndarray = np.zeros(shape=(height, width))
        for i in range(step//2, height, step):
            mat[i] += 1 # setting the row to one
        return mat
