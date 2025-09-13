import odl
import numpy as np

from .data import XRay
from ..util.camera import build_camera_model


ASSUMED_GRID_SPACING =  0.8 # [mm] 


def xray_to_camera_model(xray, grid_size, grid_spacing=ASSUMED_GRID_SPACING):
    return build_camera_model(
        xray.alpha,
        xray.beta,
        xray.sid,
        xray.sod,
        grid_spacing,
        grid_size,
        xray.spacing,
        xray.size
)    


def preprocess(xray0: XRay, xray1: XRay, grid_size):
    '''
    get initial estimation of the shape
    '''

    trf0 = xray_to_camera_model(xray0, grid_size)
    trf1 = xray_to_camera_model(xray1, grid_size)

    bp0 = trf0.adjoint(xray0.img)
    bp1 = trf1.adjoint(xray1.img)
    intersect = np.minimum(bp0, bp1)    
    return intersect.asarray()
