import numpy as np
import cv2
from skimage.feature import corner_shi_tomasi, corner_foerstner, corner_kitchen_rosenfeld

def get_shi_tomasi(src):

    src_np = np.array(src)
    src_np.astype(int)
    score_matrix = corner_shi_tomasi(src_np, sigma=10)

    return score_matrix
