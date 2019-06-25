import colorsys

import numpy as np
import tensorflow as tf
from matplotlib.colors import Colormap


class Colorizer(Colormap):

    @staticmethod
    def colorise_depth_map(depth_matrix, is_sparse: bool,min_val=0., max_val=1., saturation=1., value=0.7):
        """ Converts the given depth map to a colorized image
        
        Returns: an RGB image of the same resolution where depth values map to hue
        
        """
        max_val = max_val * 0.999
        shape = depth_matrix.shape
        assert len(shape) == 2 # 2 dimensional images only
        color_matrix = np.vectorize(Colorizer.color_mapper)(depth_matrix, is_sparse, min_val=min_val, max_val=max_val, saturation=saturation, value=value)
        # This outputs a matrix of shape 3*780*1260 we transform it back to a 780*1260*3 matrix
        return np.swapaxes(np.swapaxes(color_matrix,0,2),0,1)

    @staticmethod
    def color_mapper(depth, is_sparse: bool, min_val: float, max_val: float, saturation: float, value: float):
        if (is_sparse and depth == min_val ) or depth >= max_val:
            return colorsys.hsv_to_rgb(0., 0., 0.)
        elif depth < min_val:
            return colorsys.hsv_to_rgb(0., saturation, value)
        else:
            return Colorizer.color_encoder(depth, min_val, max_val, saturation, value)

    @staticmethod
    def color_encoder(depth: float, min_val: float, max_val: float, saturation: float, value: float):
        hue = np.sqrt((depth - min_val) / (max_val - min_val))
        return colorsys.hsv_to_rgb(hue, saturation, value)

    @staticmethod
    def colorise_sparse_depth_map(depth_matrix, sparsity_matrix, min_val=0., max_val=1., saturation=1., value=0.7):
        """ Converts the given depth map to a colorized image
        
        Returns: an RGB image of the same resolution where depth values map to hue
        
        """
        max_val = max_val * 0.999
        assert depth_matrix.shape == sparsity_matrix.shape
        shape = depth_matrix.shape
        assert len(shape) == 2 # 2 dimensional images only
        h, w = shape
        black = tf.zeros((h, w, 3))
        tf.where(tf.less(depth_matrix, tf.constant(min_val, shape=(h,w))), tf.constant(min_val, shape=(h,w)), depth_matrix)
        tf.where(tf.less(depth_matrix, tf.constant(max_val, shape=(h,w))), depth_matrix, tf.constant(max_val, shape=(h,w)))
        
        full_color_matrix = np.vectorize(Colorizer.color_encoder)(depth_matrix, min_val, max_val, saturation, value)
        # This outputs a matrix of shape 3*780*1260 we transform it back to a 780*1260*3 matrix
        full_color_matrix = np.swapaxes(np.swapaxes(full_color_matrix,0,2),0,1)
        sparsity_matrix = tf.reshape(sparsity_matrix, shape=(h, w, 1))
        sparse_color_matrix = np.where(tf.equal(sparsity_matrix, tf.zeros_like(sparsity_matrix)), black, full_color_matrix,)
        return sparse_color_matrix
