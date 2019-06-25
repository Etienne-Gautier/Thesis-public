from typing import Callable, List

import numpy as np
from PIL import Image


class Interpolation:
    """Algorithms to interpolate RGB and depth images
    """

    @staticmethod
    def RGB_bicubic_interpolate(pil_image):
        pil_image.imresize((pil_image.size[0]*2, pil_image.size[1]*2), "bicubic")


    @staticmethod
    def bicubic_kernel(x: float, alpha: float=-0.5) -> float:
        abs_x = abs(x)
        if(abs_x< 1):
            result = (alpha + 2.) * np.power(abs_x, 3) - (alpha + 3.) * np.power(abs_x, 2) + 1.
        elif(abs_x<2):
            result = alpha * np.power(abs_x, 3) - 5 * alpha * np.power(abs_x, 2) + 8 * alpha * abs_x - 4 * alpha
        else:
            result = 0
        return result

    @staticmethod
    def create_mask(mask_size: int, kernel: Callable[[float], float]) -> np.ndarray:
        assert(mask_size % 2 == 1) # mask must have an odd length
        mask = np.zeros(shape=(mask_size, mask_size))
        centre = mask_size // 2
        for i in range(mask_size):
            for j in range(mask_size):
                mask[i, j] = kernel(2 * np.sqrt(np.power(i - centre, 2) + np.power(j - centre, 2))/mask_size)
        return mask

    @staticmethod
    def LIDAR_bicubic_interpolate(image_array: np.ndarray, mask_size=3):
        assert(len(image_array.shape) == 2) # check that the image is a 2D array
        height, length = image_array.shape
        mask = Interpolation.create_mask(mask_size, Interpolation.bicubic_kernel)
        image_interpolated = image_array.copy()
        for x in range(height):
            for y in range(length):
                image_interpolated[x, y] = Interpolation.sum_kernel_around(image_array, x, y, mask)
        return image_interpolated
    
    @staticmethod
    def sum_kernel_around(image_array: np.ndarray, x: int, y: int, mask: np.ndarray) -> float:
        assert(len(mask.shape) == 2) #can only apply 2D masks
        assert(len(image_array.shape) == 2) # check that the image is a 2D array
        height, length = image_array.shape
        mask_semi_height: int = mask.shape[0]//2
        mask_semi_width: int = mask.shape[1]//2
        significant_pixels : int = 0
        result: float = 0.
        for dx in range(-mask_semi_height, mask_semi_height + 1):
            x_dx_mirror: int = 2 * (height - 1) - (x + dx) if (x + dx> height -1) else abs(x + dx) # mirror the image at its extremities in x
            for dy in range(-mask_semi_width, mask_semi_width + 1):
                y_dy_mirror: int = 2 * (length - 1) - (y + dy) if (y + dy > length - 1) else abs(y + dy) # mirror the image at its extremities in y
                if(image_array[x_dx_mirror, y_dy_mirror] != 0):
                    significant_pixels += 1
                    result += mask[dx, dy] * image_array[x_dx_mirror, y_dy_mirror]
        result = result/significant_pixels if(significant_pixels != 0) else 0 #scale to consider sparse data
        return result
