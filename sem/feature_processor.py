from collections import OrderedDict
import numpy as np
import scipy as sp

import sem.object_feature
import sem.se3 
import sem.levenberg_marquardt as lm
import sem.bbox_residual

import g2o

class EKFObjectFeature():
    def __init__(self): 
        self.sigma = np.eye(6)

    def obtain_kalman_gain(self, Nt, Ht, V):
        IV = np.kron(np.eye(Nt), V)
        Kt = self.sigma @ Ht.T @ np.linalg.inv(Ht @ self.sigma @ Ht.T + IV)
        self.Kt = Kt

    def perform_update(self, residual, Ht, V, wTc):
        Nt = np.shape(residual)[0]
        self.obtain_kalman_gain(Nt, Ht, V)
        delta_se3 = (self.Kt @ residual).T 
        delta_T = sem.se3.axangle2pose(delta_se3)
        delta_T = np.squeeze(delta_T)

        wTc_new = delta_T @ wTc
        self.sigma = (np.eye(6) - self.Kt @ Ht) @ self.sigma
        return wTc_new

class ObjectFeatProcessor: 
    def __init__(self): 
        # for object features
        # <FeatureID, Feature>
        self.map_server = OrderedDict()
        # we need a dictionary to store a window of camera poses
        self.cam_states = OrderedDict()

        self.ekf = EKFObjectFeature()

    def add_cam_poses(self, pose_g2o, img_id): 
        pose = sem.se3.SE3(pose_g2o.orientation().matrix(), pose_g2o.position())
        self.cam_states[img_id] = pose 

    def feature_callback(self, feat_obs_published):
        """
        process the feature msg published by the front end
        :param feat_obs_published: a dictionary, 
            whose key is id, value is feature message defined in message.py
        """
        # check whether the dictionary is empty
        if not feat_obs_published:
            return
        object_feat_ids_to_remove = []
        # need to obtain the Jacobian matrix size first, so we initialize the feature
        # and then compute the jacobians
        object_feat_max_jacobian_size = 0
        for feat_id, feat_obs in feat_obs_published.items():
            if feat_obs['type'] != 'object':
                continue 
            # process object features
            object_feature = sem.object_feature.ObjectFeature(feat_id, feat_obs)
            object_feature.add_cam_poses(self.cam_states)
            # init object state
            object_feature.initialize()
            # optimize object states
            is_inlier_flag = object_feature.optimize_shape()
            self.map_server[feat_id] = object_feature
            # for updating camera poses 
            if is_inlier_flag:
                object_feat_ids_to_remove.append(feat_id)
                # obs include zs + zb
                object_feat_max_jacobian_size += (2 * sem.myobject.NUM_KEYPOINTS + 4 * 1) \
                                * len(object_feature.feature_msg['img_id'])

        if object_feat_ids_to_remove:
            # only update the latest camera pose
            cam_states = OrderedDict()
            temp_state = next(reversed(self.cam_states.items()))
            cam_states[temp_state[0]] = temp_state[1]

            H_object = np.zeros((object_feat_max_jacobian_size, 6*len(cam_states)))
            r_object = np.zeros((object_feat_max_jacobian_size, 1))
            R_object = np.identity(len(H_object))
            stack_count = 0

            for feat_id in object_feat_ids_to_remove:
                object_feature = self.map_server[feat_id]
                success_flag, H_o_j, r_o_j, R_o_j = get_all_object_feat_JrR_per_frame(object_feature, 
                    cam_states, self.K)
                if not success_flag: 
                    continue 

                jacobian_size = np.shape(H_o_j)[0]
                H_object[stack_count:stack_count + jacobian_size, :] = H_o_j
                r_object[stack_count:stack_count + jacobian_size, :] = r_o_j
                R_object[stack_count:stack_count + jacobian_size, stack_count:stack_count + jacobian_size] = R_o_j
                stack_count += jacobian_size

            H_object = H_object[:stack_count, :]
            r_object = r_object[:stack_count, :]
            R_object = R_object[:stack_count, :stack_count]

            wTc = next(reversed(self.cam_states.items())).matrix()
            wTc_new = self.ekf.perform_update(self, r_object, H_object, R_object, wTc)
            temp_state = next(reversed(self.cam_states.items()))
            self.cam_states[temp_state[0]] = sem.se3.SE3(wTc_new[:3, :3], wTc_new[3, :3])

            return g2o.Isometry3d(wTc_new[:3, :3], wTc_new[3, :3])

def huber_cost(residual, huber_epsilon = np.inf):

    # apply huber robust cost

    # huber_epsilon = np.inf
    # huber_epsilon = 5e-2
    # huber_epsilon = 5e-2

    if np.isinf(huber_epsilon):
        # m x 1
        w = np.ones((residual.shape[0], 1))
    else:
        # m x 1
        w = lm.huberDerivative(residual, huber_epsilon)[...,None]

    return w

def remove_lost_object(zs, zb, S, T, M, v, K):
    '''
    jacobians for reproj error and line error wrt camera pose
    * @Input:
    *    zs = n x m x 2 = keypoint observations from n frames (normalized pixel coordinates)
    *    zb = n x 4 = bbox observations from n frames (normalized pixel coordinates)
    *    S = n x 4 x 4 = inverse camera pose, i.e., transform from world to optical frame
    *    T = 4 x 4 = object pose, i.e., transform from object to world frame
    *    M = m x 4 = object shape in homogeneous coordinates
    *    v = 3 x 1 = object shape
    *    K = 3 x 3 = intrinsic matrix
    * @Output:
    *    J = n x (2m + 4) x 6
    '''

    T = sem.se3.perturb_T(T)

    zs = zs[None, ...] if zs.ndim < 2 else zs
    zs = zs[None, ...] if zs.ndim < 3 else zs
    zb = zb[None, ...] if zb.ndim < 2 else zb
    S = S[None, ...] if S.ndim < 3 else S
    M = M[None, ...] if M.ndim < 2 else M

    J = np.zeros((zs.shape[0], zs.shape[1] * 2 + 4, 6))
    # zs[..., 0] is n x m, eg 20 x 12
    valids = np.isfinite(zs[..., 0])
    # valids_id entry represents the row, col of nonzero items in valids
    valids_id = np.nonzero(valids)

    # semantic kps residual wrt object pose
    # Proposition  2.

    # Mw is object shape in world frame
    # size is num_valid x 4 x 1
    # num_valid is the valid observation number, eg 188
    Mw = T @ M[valids_id[1], ..., None]
    # num_valid x 4 x 4
    S_valid = S[valids_id[0]]
    # num_valid x 4
    Mo = np.squeeze(S_valid @ Mw)
    # num_valid x 4 x 6
    # left perturbation 
    # Jp = -lm.projectionJacobian(Mo) @ S_valid @ src.se3.odotOperator(np.squeeze(Mw))
    # right perturbation 
    Jp = -lm.projectionJacobian(Mo) @ sem.se3.odotOperator(Mo)

    # this is equivalent to multiplying with P
    J[valids_id[0], valids_id[1] * 2 + 0, :] = Jp[:, 0, :]
    J[valids_id[0], valids_id[1] * 2 + 1, :] = Jp[:, 1, :]

    # bbox residual wrt camera pose
    # Proposition  3.

    # n x 1
    validb = np.all(np.isfinite(zb), axis=-1)

    # 3 x 3 
    U_square = np.array([[v[0] ** 2, 0.0, 0.0],
                        [0.0, v[1] ** 2, 0.0],
                        [0.0, 0.0, v[2] ** 2]])

    # projection matrix 
    P = np.zeros((3, 4))
    P[:3, :3] = np.eye(3)

    # use for loop instead of vectorized version for easy debugging 
    # TODO: use vectorized version later
    for i in range(zb.shape[0]):

        if ~validb[i]:
            continue 

        uline_zb_all = lm.bbox2lineh(zb[i, :])

        for j in range(uline_zb_all.shape[0]):

            uline_zb = uline_zb_all[j, :]
            uline_b = sem.bbox_residual.uline_zb_to_uline_b(T, S[i], P, uline_zb)

            norm_b = sem.bbox_residual.normalize_up(uline_b)

            x0 = np.array([0, 0, 0, 1])
            p_be_p_ulinebhat = sem.bbox_residual.ellipse_plane_dist_full.df(x0, U_square, norm_b)

            p_ulinebhat_p_ulineb = sem.bbox_residual.normalize_up.df(uline_b)

            # right perturbation 
            p_ulineb_p_Cxit = -T.T @ S[i].T @ sem.se3.circledCirc(P.T @ uline_zb).T

            J_Cxit = p_be_p_ulinebhat @ p_ulinebhat_p_ulineb @ p_ulineb_p_Cxit

            J[i, zs.shape[1] * 2 + j, :] = J_Cxit

    # covariance for object obs residual
    R_line_res_cov = 1e-1
    # R_line_res_cov = 1e-5

    # zs.shape[1] * 2 + 4 means covariance for semantic keypoints
    # and the four lines from bbox
    # TODO: how to deal with nan in zs? is ok if we set covariance to 0
    # since jacobian is 0?
    R_i_temp = np.zeros((zs.shape[0], zs.shape[1] * 2 + 4, zs.shape[1] * 2 + 4))

    for frame_id in range(zs.shape[0]):
        # R from mc dropout
        for part_id in range(zs.shape[1]):
            if valids[frame_id, part_id]:

                # R_dropout = kps_cov_mc_dropout[part_id]
                # R_j = np.array(R_dropout)
                #
                # R_j[0, 0] /= (K[0, 0] ** 2)
                # R_j[1, 1] /= (K[1, 1] ** 2)
                # R_j[0, 1] /= (K[0, 0] * K[1, 1])
                # R_j[1, 0] /= (K[0, 0] * K[1, 1])
                # # increase the covariance
                # R_j *= 1e2

                # use a uniform noise
                # R_j = np.eye(2) * 1e-3
                R_j = np.eye(2) * 1e-2

                R_i_temp[frame_id, 2 * part_id: 2 * (part_id + 1), 2 * part_id: 2 * (part_id + 1)] = R_j

        if validb[frame_id]:
            R_i_temp[frame_id, -4:, -4:] = np.eye(4) * R_line_res_cov

    R_i = []
    for frame_id in range(zs.shape[0]):
        if frame_id == 0:
            R_i = R_i_temp[frame_id, :, :]
        else:
            R_i = sp.linalg.block_diag(R_i, R_i_temp[frame_id, :, :])

    return J, R_i

def nullspace_proj(Hf, Hx, r, R):
    """
    project the nullspace of Hf to Hx
    :param Hf: size nx3, jacobian matrix to compute the left nullspace
    :param Hx: size nx(12+6m), jacobian matrix to decouple from Hf
    :param r: size nx1, residual
    :param R: size nxn, covariance of measurement model
    :return: success_flag, whether the nullspace exists, Ho, ro are the jacobian and
    residual after nullspace projection
    """

    success_flag = False
    Ho, ro, Ro = None, None, None 

    # Hf = [Q_1, Q_2] [ R ]
    #                 [ 0 ]
    Qs, Rs = np.linalg.qr(Hf, mode='complete')
    Q2 = Qs[:, Hf.shape[1]:]
    np.testing.assert_allclose(Q2.T @ Q2, np.eye(Q2.shape[1]), rtol=1e-4, atol=1e-6)

    if np.size(Q2) == 0:
        return success_flag, Ho, ro, Ro
    else:
        success_flag = True

    Ho = Q2.T @ Hx
    ro = Q2.T @ r

    Ro = Q2.T @ R @ Q2

    return success_flag, Ho, ro, Ro

def get_all_object_feat_JrR_per_frame(object_feature, cam_states, K):
    """
    marginalize object feature
    :param object_feature: a class that contains the observations and
    states of the object
    :param cam_states: camera states
    :param K: size 3x3, intrinsics
    """

    # note we should use mean_shape + deformation
    mean_shape = np.copy(object_feature.my_object.mean_shape)
    deformation = object_feature.my_object.deformation

    cam_N = len(cam_states)
    # M is no. of keypoints
    M = sem.myobject.NUM_KEYPOINTS

    # residual
    err_obj_i_feat = lm.errorFeatureQuadric(
        object_feature.feature_msg['zs'], object_feature.oTw,
        object_feature.my_object.wTq.matrix(), lm.point2homo(mean_shape + deformation))

    err_obj_i_bbox = lm.errorBBoxQuadric(
        object_feature.feature_msg['zb'], object_feature.oTw,
        object_feature.my_object.wTq.matrix(), object_feature.my_object.v)

    # for debugging object update
    # err_obj_i_feat *= 0
    # err_obj_i_bbox *= 0

    w_feat = huber_cost(err_obj_i_feat, 1e-3)
    w_bbox = huber_cost(err_obj_i_bbox, 5e-2)

    err_obj_i_feat = np.sqrt(w_feat) * err_obj_i_feat
    err_obj_i_bbox = np.sqrt(w_bbox) * err_obj_i_bbox

    err_obj_i = np.concatenate((err_obj_i_feat, err_obj_i_bbox), axis = 1)
    err_obj_i = np.reshape(err_obj_i, (-1, 1))

    # jacobian wrt camera pose
    J_obj_wrt_cam, R_i = remove_lost_object(
        object_feature.feature_msg['zs'], object_feature.feature_msg['zb'], object_feature.oTw,
        object_feature.my_object.wTq.matrix(), lm.point2homo(mean_shape + deformation), object_feature.my_object.v, K)

    w = np.ones((np.shape(J_obj_wrt_cam)[0], np.shape(J_obj_wrt_cam)[1], 1))
    w[:, :-4, :] *= w_feat[..., None]
    w[:, -4:, :] *= w_bbox[..., None]

    J_obj_wrt_cam = np.sqrt(w) * J_obj_wrt_cam

    H_x_i = np.zeros((0, 6 * cam_N))

    for i in range(np.shape(J_obj_wrt_cam)[0]):

        img_id = object_feature.feature_msg['img_id'][i]
        if img_id not in list(cam_states.keys()):
            continue 

        # frame id is the index of cam state id in list
        frame_id = list(cam_states.keys()).index(img_id)

        H_x = np.zeros((2 * M + 4, 6 * cam_N))
        H_x[:, frame_id * 6 : (frame_id + 1) * 6] = J_obj_wrt_cam[i, :, :]
        H_x_i = np.concatenate((H_x_i, H_x))

    # jacobian wrt object state
    H_obj_i_feat = lm.jacobianFeatureQuadric(
        object_feature.feature_msg['zs'], object_feature.oTw,
        object_feature.my_object.wTq.matrix(), lm.point2homo(mean_shape + deformation))

    H_obj_i_bbox = lm.jacobianBBoxQuadric(
        object_feature.feature_msg['zb'], object_feature.oTw,
        object_feature.my_object.wTq.matrix(), object_feature.my_object.v)

    H_obj_i_feat = np.sqrt(w_feat[..., None]) * H_obj_i_feat
    H_obj_i_bbox = np.sqrt(w_bbox[..., None]) * H_obj_i_bbox

    # H_obj_i size is (frame x no. observations) x object state
    H_obj_i = np.concatenate((H_obj_i_feat, H_obj_i_bbox), axis=1)

    # err_obj_i is same size
    H_obj_i = np.reshape(H_obj_i, (-1, np.shape(H_obj_i)[-1]))

    # Project the residual and Jacobians onto the nullspace of H_fj.
    success_flag, H_o_j, r_o_j, R_o_j = nullspace_proj(H_obj_i, H_x_i, err_obj_i, R_i)

    return success_flag, H_o_j, r_o_j, R_o_j