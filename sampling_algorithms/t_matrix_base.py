import numpy as np

class T_Matrix_Base:
    def __init__(self, camera_resolution: tuple[int, int], projector_resolution: tuple[int, int]):
        """
        resolution: a tuple of camera resolution pixels
        """
        self.camera_resolution = camera_resolution
        self.projector_resolution = projector_resolution
    
    def get_T_matrix(self) -> np.ndarray:
        raise NotImplementedError

