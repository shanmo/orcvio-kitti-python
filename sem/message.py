from collections import OrderedDict, namedtuple

# image message that has colored image, grayscale image, timestamp and id
img_msg = namedtuple('img_msg', ['img_bgr', 'img_id'])

# feature message has unified obs format: use a dictionary, key is feature id,
# value is the obs format, which is a dictionary,
# that has feature type (geometric vs object), zg, zs, zb, starmap R.

# keys are feature id
feature_msg = {}

# keys are 'feature_type', 'img_id', 'zg', 'zs', 'zb', 'R0'
# img id is a list
# zg size is nx2
# zs size is nx12x2
# zb size is nx4
# R0 size is nx3x3
obs_msg = {}