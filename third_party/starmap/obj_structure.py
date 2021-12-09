import numpy as np
from collections import OrderedDict

"""need to use code from starmap_object_structure repo"""
obj_str_dict = OrderedDict([('upper_left_windshield', np.array([-0.09472257, -0.07266671,  0.10419698])),
('upper_right_windshield', np.array([ 0.09396329, -0.07186594,  0.10468729])),
('upper_right_rearwindow', np.array([0.100639  , 0.26993483, 0.11144333])),
('upper_left_rearwindow', np.array([-0.100402 ,  0.2699945,  0.111474 ])),
('left_front_light', np.array([-0.12014713, -0.40062513, -0.02047777])),
('right_front_light', np.array([ 0.1201513 , -0.4005558 , -0.02116918])),
('right_back_trunk', np.array([0.12190333, 0.40059162, 0.02385612])),
('left_back_trunk', np.array([-0.12194733,  0.40059462,  0.02387712])),
('left_front_wheel', np.array([-0.16116614, -0.2717491 , -0.07981283])),
('left_back_wheel', np.array([-0.16382502,  0.25057048, -0.07948726])),
('right_front_wheel', np.array([ 0.1615844 , -0.27168764, -0.07989835])),
('right_back_wheel', np.array([ 0.16347528,  0.2507412 , -0.07981754]))])

def find_semantic_part(cam_view_feat):

    cam_view_feat = cam_view_feat / 64 - 0.5

    min_dist = 1e10
    part_id = 0
    part_name = None

    for i, key in enumerate(obj_str_dict):
        dist = np.linalg.norm(cam_view_feat - obj_str_dict[key])
        if dist < min_dist:
            min_dist = dist
            part_id = i
            part_name = key

    return part_id, part_name

"""from car fusion"""
# 1.75 / wheel distance
# pt_scale = 3.3503951474452545
"""from shape prior"""
# 1.45 / wheel distance
pt_scale = 2.7760

cov_scale = pt_scale ** 2
# assume dp = 1, dp norm is also 1
# pt_cov_gt = (1 ** 2) * pt_cov * cov_scale
