from PIL import Image
import numpy as np
import cv2
class IOreader:
    SYNTHIA_SKY_DEPTH = 655.35
    @staticmethod
    def depth_read(filename: str)-> np.ndarray:
        # loads depth map D from png file
        # and returns it as a numpy array,
        # for details see readme.txt

        depth_png = np.array(Image.open(filename), dtype=int)
        # make sure we have a proper 16bit depth map here.. not 8bit!
        #assert(np.max(depth_png) > 255)

        depth = depth_png.astype(np.float32) / 256.
        #depth[depth_png == 0] = 1
        return depth
    
    @staticmethod
    def depth_read_synthia(filename: str)-> np.ndarray:
        # loads depth map D from png file
        # and returns it as a numpy array,
        # for details see readme.txt

        depth = cv2.imread(filename, flags=cv2.IMREAD_ANYDEPTH)
        # make sure we have a proper 16bit depth map here.. not 8bit!
        depth = depth.astype('float32') /100.
        
        return depth
