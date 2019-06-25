
class KITTIExample:
    def __init__(self, path:str):
        self.path : str = path
        #self.binary : bytes
        self.id : str
        self.is_training: bool
        self.is_raw_data: bool
        self.sequence: str
        self.is_right_camera: bool #image02 indicates left camera, image03 is right camera
        self.order_in_sequence: int
        self.timestamp : int
    