import numpy as np
import itertools

import sem.se3
import sem.myobject
import sem.levenberg_marquardt as lm

class ObjectFeature():
    """
    this class is for object states
    initialization and optimization
    """

    def __init__(self, new_id = 0, feature_msg = None):

        # An unique identifier for the feature
        self.id = new_id

        # for obs format refer to message.py
        self.feature_msg = feature_msg

        self.oTw = np.zeros((0, 4, 4))

        self.my_object = sem.myobject.MyObject()

        self.feature_type = 'object'

        self.init_by_single_view_flag = True

    def add_cam_poses(self, cam_states):
        """
        this function selects the valid camera poses in the state
        :param cam_states: the state variables for camera poses
        """

        # this list keeps the id out of window
        obs_ids_to_remove = []

        for i, cam_id in enumerate(self.feature_msg['img_id']):

            if cam_id in cam_states.keys():
                cam_pose = cam_states[cam_id]
            else:
                # del obs outside window
                obs_ids_to_remove.append(i)
                continue

            wPo = cam_pose.position()
            wRo = cam_pose.orientation()

            oTw = sem.se3.SE3(wRo, wPo).inverse().matrix()

            self.oTw = np.concatenate((self.oTw, np.reshape(oTw, (1, -1, 4))), axis=0)

        self.remove_outdated_obs(obs_ids_to_remove)

    def remove_outdated_obs(self, to_remove):
        """
        remove the observations out of window
        :param img_ids_to_remove: a list of image ids out of window
        """

        img_id_list = self.feature_msg['img_id']
        zs_mat = self.feature_msg['zs']
        zb_mat = self.feature_msg['zb']
        # R_mat = self.feature_msg['R0']

        # in the simple sim test R0 does not exist
        # so we have to create it here
        if 'R0' not in self.feature_msg:
            R0 = np.tile(np.eye(3), (len(img_id_list), 1, 1))
            self.feature_msg['R0'] = R0
        R_mat = self.feature_msg['R0']

        # Note that you need to delete them in reverse order
        # so that you don't throw off the subsequent indexes.
        for index in sorted(to_remove, reverse=True):
            self.feature_msg['img_id'].pop(index)

        self.feature_msg['zs'] = np.delete(zs_mat, to_remove, axis = 0)
        self.feature_msg['zb'] = np.delete(zb_mat, to_remove, axis = 0)
        self.feature_msg['R0'] = np.delete(R_mat, to_remove, axis = 0)

    def initialize(self):
        """
        initialize the object states
        """

        self.init_by_single_view_flag, wTq = get_initial_wTq(self.feature_msg['zs'], self.feature_msg['zb'], self.oTw,
            self.my_object.mean_shape, self.feature_msg['R0'], self.my_object.v)

        # only considers yaw
        wTq = sem.se3.poseSE32SE2(wTq)

        self.my_object.set_wTq(wTq)
        self.my_object.set_kps_shape()

        # need this step to compare init
        # and lm
        wRq = wTq[:3, :3]
        wPq = wTq[:3, 3]
        self.my_object.wTq_init = sem.se3.SE3(wRq, wPq)

    def optimize_shape(self):
        """
        optimize the object states
        :return: is_inlier_flag if object is not inlier for LM,
        do not use it to update camera pose
        """

        meanShape = np.copy(self.my_object.mean_shape)

        # define error function
        # note we use M + D ie meanShape + x[2]
        err_feature = lambda x: lm.errorFeatureQuadric(self.feature_msg['zs'],
                    self.oTw, x[0], lm.point2homo(meanShape + x[2]))

        err_bbox = lambda x: lm.errorBBoxQuadric(self.feature_msg['zb'], self.oTw, x[0], x[1])

        err_d_reg = lambda x: lm.errorDeformReg(self.feature_msg['zs'],
                    lm.point2homo(meanShape + x[2]), meanShape)

        err_v_reg = lambda x: lm.errorQuadVReg(self.feature_msg['zs'], x[1])

        # define jacobian
        # note we use M + D ie meanShape + x[2]
        jacobi_feature = lambda x: lm.jacobianFeatureQuadric(self.feature_msg['zs'], self.oTw, x[0], lm.point2homo(meanShape + x[2]))

        jacobi_bbox = lambda x: lm.jacobianBBoxQuadric(self.feature_msg['zb'], self.oTw, x[0], x[1])

        jacobi_d_reg = lambda x: lm.jacobianDeformReg(self.feature_msg['zs'], lm.point2homo(meanShape + x[2]))

        jacobi_v_reg = lambda x: lm.jacobianQuadVReg(self.feature_msg['zs'])

        # define addition
        # for updating wTq, v, deformation
        # use right perturbation 
        addh = lambda x, dx: [x[0] @ sem.se3.axangle2pose(dx[:6]), x[1] + dx[6:9], x[2] + np.reshape(dx[9:], (-1, 3))]

        # note the object state is
        # lm input: wTq, v, deformation initialized by zeros
        # in that order!
        xhat = [self.my_object.wTq.matrix(), self.my_object.v, np.zeros((sem.myobject.NUM_KEYPOINTS, 3))]

        # for testing in simulation
        lm_config = LMConfig()

        # for real kitti seq.
        # lm_config = LMConfig(1e-2, .5, 10, 10, 10, 1e3)
        # lm_config = LMConfig(1e-2, 0.1, 30, .5, .5, 1e3)

        x2, c2, is_inlier_flag = lm.levenbergMarquardt(err_feature, err_bbox, err_d_reg, err_v_reg,
            jacobi_feature, jacobi_bbox, jacobi_d_reg, jacobi_v_reg,
            xhat, lm_config, addh)

        if not is_inlier_flag:
           return is_inlier_flag

        # note, here x2[1] is shape, x2[2] is mean shape deformation
        wTq_opt, v_opt, m_d_opt = x2[0], x2[1], x2[2]

        # updates wTq, v, D
        wTq_opt = sem.se3.poseSE32SE2(wTq_opt)

        self.my_object.set_wTq(wTq_opt)
        self.my_object.deformation = m_d_opt
        self.my_object.v = v_opt

        # finally, updates object shape
        # perform this step after updating T, v, D
        self.my_object.set_kps_shape()

        return is_inlier_flag

class LMConfig:
    """
    define weights
    WP: weight for kps zs
    WL: weight for bbox zb
    WD: weight for deformation D
    WV: weight for shape v
    """

    # def __init__(self, HUBER_EPSILON = np.inf,
    #              WP = 1., WL = 1., WD = 1., WV = 1.,
    #              trace_threshold = 0.):

    def __init__(self, HUBER_EPSILON = np.inf,
                 WP = 1., WL = 1., WD = 0., WV = 0.,
                 trace_threshold = 0.):

        # note that the regularization weights should be smaller
        # to enable large deformation

        self.HUBER_EPSILON = HUBER_EPSILON
        self.WP, self.WL, self.WD, self.WV = WP, WL, WD, WV
        self.trace_threshold = trace_threshold

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
utility functions
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def get_initial_wPq_single_view(oTw, bbox, wRq, v):
    """
    initialize wPq from a single frame
    :param oTw: size 4x4, world to optical axis transformation of camera pose
    :param bbox: size 1x4, normalized bbox in xyxy format
    :param wRq: size 3x3, object to world transformation from starmap
    :param v: size 1x3, ellipoid's shape param of object
    :return: size 1x3, initial position of object in world frame
    """

    oRw = oTw[0:3, 0:3]
    oPw = oTw[0:3, 3]

    # since we always use normalized coordinates, K is always identity
    K = np.eye(3)

    # yolo bbox is usually much smaller than the full object
    # this will confuse our algorithm and think that the object
    # is farther away, to solve this, we need a empirical value
    # to compensate

    # this scale should have different values for different
    # x, y, z, since uncertainty in z is much larger
    empirical_bbox_scale = np.array([.8, .6, .7])

    v = empirical_bbox_scale * v
    V = np.diag(v * v)

    A = wRq @ V @ wRq.T
    B = K @ oRw

    normalized_bbox = bbox_corner2wh(bbox)
    bbox_lines = bbox2lines(normalized_bbox)

    line_sum = 0
    denominator = 0

    for i in range(len(bbox_lines)):
        line = bbox_lines[i]
        line = np.reshape(line, (3, -1))
        # print("line is {}".format(line.T))

        line_sum += line @ line.T
        denominator += line.T @ B @ A @ B.T @ line

    E = B.T @ line_sum @ B / denominator
    center = get_bbox_center(normalized_bbox)

    b = np.zeros((3, 1))
    b[0:2, 0] = center
    b[2, 0] = 1
    d = 1 / np.sqrt(b.T @ np.linalg.inv(B).T @ E @ np.linalg.inv(B) @ b)
    d = np.asscalar(d)

    wPq = np.squeeze(d * np.linalg.inv(B) @ b) - np.squeeze(oRw.T @ oPw)
    # if something goes wrong with inv, just use 0
    wPq = np.nan_to_num(wPq)

    return wPq

def get_initial_wTq(zs, zb, oTw, MS, oRq, v):
    """
    initializeObjectStateV6
      @Input:
        zs = n x m x 2 = semantic keypoint observations (normalized pixel coordinates)
        zb = n x 4     = bounding box observations (normalized pixel coordinates)
        oTw = n x 4 x 4 = sequence of inverse camera poses, from world to optical axis
        MS  = m x 3 = object mean shape (in object frame)
        oRq = n x 3 x 3 = rotation matrix from object to camera frame obtained from starmap
        v: size 1x3, ellipoid's shape param of object
    :return:
        wTo = 4 x 4 = object pose
        v = 3 x 1 = object shape
    """

    init_by_single_view_flag = False

    MS = MS[None, ...] if MS.ndim < 2 else MS

    valid = np.isfinite(zs)[..., 0]  # n x m
    valid_g1 = np.sum(valid, axis=0) > 1
    num_valid = np.sum(valid_g1)

    oTq0 = np.eye(4)
    oTq0[:3, :3] = oRq[0, :, :]
    wTq0 = sem.se3.inversePose(oTw[0, :, :]) @ oTq0
    wRq0 = wTq0[:3, :3]

    wTq = np.eye(4)

    # initialize position using single view only
    wPq0 = get_initial_wPq_single_view(oTw[0, :, :], zb[0, :], wRq0, v)

    # use single view
    num_valid_obs_threshold = 3
    if num_valid < num_valid_obs_threshold:

        wTq[:3, :3] = wRq0
        wTq[:3, 3] = wPq0

        init_by_single_view_flag = True

        return init_by_single_view_flag, wTq

    # (3+n) x m
    valid_vars = np.nonzero(np.vstack((np.tile(valid_g1, (3, 1)), valid)))
    valid_id = np.nonzero(valid)

    # transform to world frame
    zs_homo = lm.point2homo(zs)
    zw = zs_homo @ oTw[:, :3, :3]

    # Construct linear system of equations for the unknown scales (12 x n)
    # and world frame object position (12 x 3) with constraints given by the n x m x 3 feature obsevations
    A = np.zeros((zs.shape[0], zs.shape[1], 3, 3 + zs.shape[0], zs.shape[1]))
    for t in range(A.shape[0]):
        for k in range(A.shape[1]):
            A[t, k, :, :3, k] = -np.eye(3)
            A[t, k, :, 3 + t, k] = zw[t, k, :]

    AA = A[valid_id[0], valid_id[1], ...]
    AA = AA[..., valid_vars[0], valid_vars[1]]
    AA = np.reshape(AA, (-1, AA.shape[2]))

    wTo = sem.se3.inversePose(oTw[:, :, :])
    b = np.squeeze(-wTo[valid_id[0], None, :3, 3])

    # Solve for the world frame object position and unknown scales:
    x = np.linalg.lstsq(AA, b.flatten(), rcond=None)[0]

    # Use Kabsch
    estimated_landmarks_world = np.reshape(x[valid_vars[0] < 3], (3, -1)).T

    # Kabsch cannot deal with less than 3 points
    num_valid_pts_threshold = 3
    if np.shape(estimated_landmarks_world)[0] < num_valid_pts_threshold:

        wTq[:3, :3] = wRq0
        wTq[:3, 3] = wPq0

        init_by_single_view_flag = True

        return init_by_single_view_flag, wTq

    # use ransac to find better kabsch estimate
    MS_kabsch = np.copy(MS[valid_g1, :])

    iterable = list(range(num_valid))
    combinations = itertools.combinations(iterable, num_valid_pts_threshold)
    max_num_inliers = 0
    opt_inliers = None

    for subset in combinations:
        selected_est = np.copy(estimated_landmarks_world[subset, :])
        selected_MS = np.copy(MS_kabsch[subset, :])

        wPq, wRq = findTransform(selected_est, selected_MS)
        num_inliers, inliers = evaluate_kabsch_ransac(wPq, wRq, estimated_landmarks_world, MS_kabsch)
        if num_inliers > max_num_inliers:
            max_num_inliers = num_inliers
            opt_inliers = inliers

    if max_num_inliers >= num_valid_pts_threshold:
        # re-estimate using all inliers from best model
        opt_est = np.copy(estimated_landmarks_world[opt_inliers, :])
        opt_MS = np.copy(MS_kabsch[opt_inliers, :])
        wPq, wRq = findTransform(opt_est, opt_MS)

        wTq = np.eye(4)
        wTq[:3,3], wTq[:3,:3] = wPq, wRq
    else:
        wTq[:3, :3] = wRq0
        wTq[:3, 3] = wPq0
        init_by_single_view_flag = True

    return init_by_single_view_flag, wTq

def evaluate_kabsch_ransac(wPq, wRq, estimated_landmarks_world, MS_kabsch):
    """
    evaluate kabsch
    :param wPq: size 1x3, position of the object
    :param wRq: size 3x3, rotation of object
    :param estimated_landmarks_world: estimated keypoints in world frame
    :param MS_kabsch: mean shape of the object
    :return: number of inliers and the inlier positions
    """

    inlier_threshold = 5

    transformed_mean_shape = wRq @ MS_kabsch.T + np.reshape(wPq, (3, -1))
    transformed_mean_shape = transformed_mean_shape.T

    distances = np.linalg.norm(estimated_landmarks_world - transformed_mean_shape, axis=1)
    inliers = distances <= inlier_threshold
    num_inliers = np.count_nonzero(inliers == True)

    return num_inliers, inliers

def findTransform( S1, S2 ):
    '''
    * Find the rotation and translation aligning the points  S2 (nx3) to S1 (nx3)
    '''
    S1m = S1.mean(axis=0,keepdims=True)
    S1c = S1 - S1m
    S2m = S2.mean(axis=0,keepdims=True)
    S2c = S2 - S2m
    R = findRotation( S1c, S2c )
    return np.squeeze(S1m - S2m @ R.T), R

def findRotation( S1, S2 ):
    '''
    * Returns the rotation matrix R that minimizes the error S1 - R*S2
    * S1, S2 are nx3 matrices of corresponding points
    '''
    M = S1.T @ S2 # 3 x 3
    U, _, VH = np.linalg.svd(M, full_matrices=False)
    R = U @ np.diag([1.0, 1.0, np.linalg.det(U @ VH)]) @ VH

    return R

def bbox_corner2wh(bbox):
    """
    change bbox from corner format
    to x, y, w, h format
    :param bbox: left, up, right, down format
    :return: x, y, w, h format
    """

    new_bbox = np.copy(bbox)

    left = bbox[0]
    up = bbox[1]
    right = bbox[2]
    down = bbox[3]

    new_bbox[2] = right - left
    new_bbox[3] = down - up

    return new_bbox

def bbox2lines(bbox):
    """
    output lines of bbox
    :param bbox: size 1x4, bbox is xywh format
    :return: size 4x3, a list that contains four lines
    """

    x1, y1, w, h = bbox

    w /= 2
    h /= 2

    C = [x1 + w, y1 + h]

    bbox_lines = []

    x1 = C[0] - w
    y1 = C[1] - h

    x2 = C[0] + w
    y2 = C[1] - h

    line = inhomo2line(x1, y1, x2, y2)
    bbox_lines.append(line)

    x1 = C[0] + w
    y1 = C[1] - h

    x2 = C[0] + w
    y2 = C[1] + h

    line = inhomo2line(x1, y1, x2, y2)
    bbox_lines.append(line)

    x1 = C[0] + w
    y1 = C[1] + h

    x2 = C[0] - w
    y2 = C[1] + h

    line = inhomo2line(x1, y1, x2, y2)
    bbox_lines.append(line)

    x1 = C[0] - w
    y1 = C[1] + h

    x2 = C[0] - w
    y2 = C[1] - h

    line = inhomo2line(x1, y1, x2, y2)
    bbox_lines.append(line)

    return bbox_lines

def inhomo2line(x1, y1, x2, y2):
    """
    convert two points in inhomogenous coordinates to a line
    that passes through them, l = p1 x p2
    """

    a = [x1, y1, 1]
    b = [x2, y2, 1]

    line = np.cross(a, b)

    return line

def get_bbox_center(bbox):
    """
    get the center of bbox
    :param bbox: size 1x4, bbox in x1, y1, w, h format
    :return: center, size 1x2, the center of bbox
    """

    x1, y1, w, h = bbox
    x0 = x1 + w / 2
    y0 = y1 + h / 2

    return np.array([x0, y0])