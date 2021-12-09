from skimage.feature import corner_harris, corner_kitchen_rosenfeld, corner_shi_tomasi, corner_moravec, corner_foerstner
import numpy as np

square = np.zeros([10, 10])
square[2:8, 2:8] = 1
square.astype(int)

w, q = corner_foerstner(square)
accuracy_thresh = 0.5
roundness_thresh = 0.3
score_matrix = (q > roundness_thresh) * (w > accuracy_thresh) * w

np.set_printoptions(formatter={'float': lambda x: "{0:0.1f}".format(x)})
print("{}".format(corner_shi_tomasi(square, sigma=10)))
# print("{}".format(score_matrix))
