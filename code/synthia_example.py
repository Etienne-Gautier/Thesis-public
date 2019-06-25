from decimal import Decimal


class SynthiaExample:

    BACKWARDS_INDEX = 0
    FORWARD_INDEX = 1

    def __init__(self, path:str):
        self.id: str
        self.path : str = path
        #self.binary : bytes
        self.is_training: bool
        self.is_raw_data: bool
        self.sparsity_level: Decimal
        self.sequence: str
        self.is_right_camera: bool
        self.camera_index: int
        self.order_in_sequence: int
        self.timestamp : int
    