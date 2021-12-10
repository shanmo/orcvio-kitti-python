"""
Optimizes the OrcVIO state during initialization


Three are four functions that eventually get optimized in the Levenberg Marquardt

1. errorFeatureQuadric
   Optimizes geometric features

2. errorBBoxQuadric
   Optimizes bounding box


3. errorDeformReg
   Object shape with deformation

4. errorQuadVReg
   Object shape regularizer from mean
"""
import numpy as np
import math
from scipy.sparse.linalg import eigs
import warnings

import sem.se3
import sem.myobject
import sem.bbox_residual

def projectionJacobian(ph):
    '''
    jacobian of projection function
    input: ph = n x 4 = homogeneous point coordinates
    return: J = n x 4 x 4 = Jacobian of ph/ph[...,2]
    '''

    J = np.zeros(ph.shape + (4,))
    iph2 = 1.0 / ph[..., 2]
    ph2ph2 = ph[..., 2] ** 2
    J[..., 0, 0], J[..., 1, 1], J[..., 3, 3] = iph2, iph2, iph2
    J[..., 0, 2] = -ph[..., 0] / ph2ph2
    J[..., 1, 2] = -ph[..., 1] / ph2ph2
    J[..., 3, 2] = -ph[..., 3] / ph2ph2

    return J

def errorFeatureQuadric(zs, S, T, M):
    '''
     * @Input:
     *    zs = n x m x 2 = keypoint observations from n frames (normalized pixel coordinates)
     *    S = n x 4 x 4 = inverse camera pose, i.e., transform from world to optical frame
     *    T = 4 x 4 = object pose, i.e., transform from object to world frame
     *    M = m x 4 = object shape (with deformation) in homogeneous coordinates
     * @Output:
     *    err = n x (2m+4+3m+3)
    '''

    zs = zs[None, ...] if zs.ndim < 2 else zs
    zs = zs[None, ...] if zs.ndim < 3 else zs

    S = S[None, ...] if S.ndim < 3 else S
    M = M[None, ...] if M.ndim < 2 else M

    # reprojection error
    # n x m x 2
    errs = np.zeros(zs.shape)
    # n x m
    valids = np.isfinite(zs[..., 0])

    # nonzero returns a tuple
    # which contains coordinates of nonzero elements
    # 1st dim is frame
    # 2nd dim is part id
    valids_id = np.nonzero(valids)

    # S is transform from world to optical frame
    # T is transform from object to world frame
    # valids_id[0] is for which frame we have valid zs
    oTq = S[valids_id[0]] @ T  # num_valid x 4 x 4
    Mo = oTq @ M[valids_id[1], ..., None]  # num_valid x 4 x 1
    zhat = np.squeeze(Mo[..., :2, :] / Mo[..., 2, None, :])
    errs[valids, :] = zhat - zs[valids, :]

    errs = np.reshape(errs, (errs.shape[0], -1))

    return errs

def errorBBoxQuadric(zb, S, T, v):
    '''
     * @Input:
     *    zb = n x 4 = bbox observations from n frames (normalized pixel coordinates)
     *    S = n x 4 x 4 = inverse camera pose, i.e., transform from world to optical frame
     *    T = 4 x 4 = object pose, i.e., transform from object to world frame
     *    v = 3 x 1 = object shape
     * @Output:
     *    err = n x 1  
    '''
    zb = zb[None, ...] if zb.ndim < 2 else zb
    S = S[None, ...] if S.ndim < 3 else S

    # line residual
    errb = np.zeros(zb.shape)

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
    # TODO: convert to vectorized version later 
    for i in range(zb.shape[0]):

        # skip if there is no bbox detection 
        if ~validb[i]:
            continue 

        uline_zb_all = bbox2lineh(zb[i, :])

        for j in range(uline_zb_all.shape[0]):
            
            uline_zb = uline_zb_all[j, :]
            uline_b = sem.bbox_residual.uline_zb_to_uline_b(T, S[i], P, uline_zb)

            norm_b = sem.bbox_residual.normalize_up(uline_b)
            
            x0 = np.array([0, 0, 0, 1])
            bbox_residual_per_line = sem.bbox_residual.ellipse_plane_dist_full(x0, U_square, norm_b)
            
            # note errb contains one residual for each line 
            # we cannot add the residuals for one bbox 
            errb[i, j] = bbox_residual_per_line

    return errb

def errorDeformReg(zs, M, Mhat):
    '''
     * @Input:
     *    zs = n x m x 2 = keypoint observations from n frames (normalized pixel coordinates)
     *    M = m x 4 = object shape (with deformation) in homogeneous coordinates
     *    Mhat = 12 x 3 = object mean shape without deformation
     * @Output:
     *    err = n x (2m+4+3m+3)
    '''

    zs = zs[None, ...] if zs.ndim < 2 else zs

    # Normally regularization term is independent of number of frames. This term is implemented in this way,
    # because we combine different residual terms together, and those for semantic keypoints and bounding boxes are summed over number of frames.
    # Hence we divide the regularization term by the number of frames as well.
    num_frames = zs.shape[0]

    # (M[:, :3] - Mhat).flatten() size is 12 x 3 = 36
    err = np.tile((M[:, :3] - Mhat).flatten() / np.sqrt(num_frames), (num_frames, 1))

    return err

def errorQuadVReg(zs, v):
    '''
     * @Input:
     *    zs = n x m x 2 = keypoint observations from n frames (normalized pixel coordinates)
     *    v = 3 x 1 = object shape
     * @Output:
     *    err = n x (2m+4+3m+3)
    '''

    zs = zs[None, ...] if zs.ndim < 2 else zs

    # Normally regularization term is independent of number of frames. This term is implemented in this way,
    # because we combine different residual terms together, and those for semantic keypoints and bounding boxes are summed over number of frames.
    # Hence we divide the regularization term by the number of frames as well
    num_frames = zs.shape[0]

    # note that we do not allow v to deviate too much from the prior v
    # instead of the previous v
    v_mean = np.copy(sem.myobject.mean_v)
    err = np.tile((v - v_mean).flatten() / np.sqrt(num_frames), (num_frames, 1))

    return err




def jacobianFeatureQuadric(zs, S, T, M):
    '''
     * @Input:
     *    zs = n x m x 2 = keypoint observations from n frames (normalized pixel coordinates)
     *    S = n x 4 x 4 = inverse camera pose, i.e., transform from world to optical frame
     *    T = 4 x 4 = object pose, i.e., transform from object to world frame
     *    M = m x 4 = object shape (with deformation) in homogeneous coordinates
     * @Output:
     *    J = n x 2m x 45 (pose, shape deformation, mean shape deformation)

    note, we calculate Jacobian wrt function f(x), not cost function f(x)^2
    so there's no 1/2 in front
    also we always use all lines for bbox
    '''

    zs = zs[None, ...] if zs.ndim < 2 else zs
    zs = zs[None, ...] if zs.ndim < 3 else zs

    S = S[None, ...] if S.ndim < 3 else S
    M = M[None, ...] if M.ndim < 2 else M

    J = np.zeros((zs.shape[0], zs.shape[1] * 2, 45))
    valids = np.isfinite(zs[..., 0])  # n x m
    valids_id = np.nonzero(valids)

    # Pose Jacobian
    # semantic kps residual wrt object pose
    # Proposition  2.
    
    S_valid = S[valids_id[0]]  # num_valid x 4 x 4
    
    # left perturbation 
    # Mw = T @ M[valids_id[1], ..., None]  # num_valid x 4 x 1
    # Mo = np.squeeze(S_valid @ Mw)  # num_valid x 4
    # Jp = projectionJacobian(Mo) @ S_valid @ sem.se3.odotOperator(np.squeeze(Mw))  # num_valid x 4 x 6

    # right perturbation 
    Mw = T @ M[valids_id[1], ..., None]  # num_valid x 4 x 1
    Mo = np.squeeze(S_valid @ Mw)  # num_valid x 4
    Jp = projectionJacobian(Mo) @ S_valid @ T @ sem.se3.odotOperator(np.squeeze(M[valids_id[1], ..., None]))  # num_valid x 4 x 6

    # Jp is 4 x 6, but J is 2 x 6, because M is in homo coord.
    # we need to multiply by P, so 4 x 6 -> 2 x 6
    J[valids_id[0], valids_id[1] * 2 + 0, :6] = Jp[:, 0, :]
    J[valids_id[0], valids_id[1] * 2 + 1, :6] = Jp[:, 1, :]

    # Shape Jacobian is zero
    # semantic kp has nothing wrt shape v
    # J[..., :-4, 6:9] = 0.0

    # Feature Jacobian
    # semantic kps residual wrt mean shape delta s
    # Proposition  2.

    # num_valid x 4 x 3
    Jf = projectionJacobian(Mo) @ S_valid @ T[..., :3]

    indices = 9 + valids_id[1] * 3 + 0

    J[valids_id[0], valids_id[1] * 2 + 0, indices] = Jf[...,0, 0]
    J[valids_id[0], valids_id[1] * 2 + 0, indices + 1] = Jf[...,0, 1]
    J[valids_id[0], valids_id[1] * 2 + 0, indices + 2] = Jf[...,0, 2]

    J[valids_id[0], valids_id[1] * 2 + 1, indices] = Jf[...,1, 0]
    J[valids_id[0], valids_id[1] * 2 + 1, indices + 1] = Jf[...,1, 1]
    J[valids_id[0], valids_id[1] * 2 + 1, indices + 2] = Jf[...,1, 2]

    return J

def jacobianBBoxQuadric(zb, S, T, v):
    '''
    * @Input:
    *    zb = n x 4 = bbox observations from n frames (normalized pixel coordinates)
    *    S = n x 4 x 4 = inverse camera pose, i.e., transform from world to optical frame
    *    T = 4 x 4 = object pose, i.e., transform from object to world frame
    *    v = 3 x 1 = object shape
    * @Output:
    *    J = n x 4 x 45 (pose, shape deformation, mean shape deformation)
    '''

    T = sem.se3.perturb_T(T)

    zb = zb[None, ...] if zb.ndim < 2 else zb
    S = S[None, ...] if S.ndim < 3 else S

    J = np.zeros((zb.shape[0], 4, 45))

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

        uline_zb_all = bbox2lineh(zb[i, :])

        for j in range(uline_zb_all.shape[0]):
            
            uline_zb = uline_zb_all[j, :]
            uline_b = sem.bbox_residual.uline_zb_to_uline_b(T, S[i], P, uline_zb)

            norm_b = sem.bbox_residual.normalize_up(uline_b)

            # bbox residual wrt object pose
            # Proposition  3.

            # right perturbation 
            p_ulineb_p_Oxi = sem.se3.circledCirc(T.T @ S[i].T @ P.T @ uline_zb).T
            # left perturbation 
            # p_ulineb_p_Oxi = T.T @ sem.se3.circledCirc(S[i].T @ P.T @ uline_zb).T

            x0 = np.array([0, 0, 0, 1])
            p_be_p_ulinebhat = sem.bbox_residual.ellipse_plane_dist_full.df(x0, U_square, norm_b)

            p_ulinebhat_p_ulineb = sem.bbox_residual.normalize_up.df(uline_b)

            J_Oxi = p_be_p_ulinebhat @ p_ulinebhat_p_ulineb @ p_ulineb_p_Oxi
            J[i, j, :6] = J_Oxi

            # bbox residual wrt ellipsoid shape delta u
            # Proposition  3.
            
            term1 = v * np.squeeze(norm_b[:-1]) * np.squeeze(norm_b[:-1])

            J_u = term1 / sem.bbox_residual.get_sqrt_bU2b(U_square, norm_b) 
            J[i, j, 6:9] = J_u

    return J

def jacobianDeformReg(zs, M):
    '''
    * @Input:
    *    zs = n x m x 2 = keypoint observations from n frames (normalized pixel coordinates)
    *    M = m x 4 = object shape in homogeneous coordinates
    * @Output:
    *    J = n x (2m+4+3m+3) x 45 (pose, shape deformation, mean shape deformation)
    '''

    zs = zs[None, ...] if zs.ndim < 3 else zs

    M = M[None, ...] if M.ndim < 2 else M

    J = np.zeros((zs.shape[0], M.shape[0] * 3, 45))

    num_frames = J.shape[0]

    # mean shape deformation Regularization Jacobian
    # zs.shape[1] * 2 + 4 + 3 * k : zs.shape[1] * 2 + 4 + 3 * (k + 1) is for the mean shape deformation regularization
    # 9 + 3 * k : 9 + 3 * (k + 1) is for each keypoint in object state
    for k in range(M.shape[0]):
        J[:, 3 * k : 3 * (k + 1), 9 + 3 * k : 9 + 3 * (k + 1)] = np.tile(
            np.eye(3) / np.sqrt(num_frames), (J.shape[0], 1, 1))

    return J

def jacobianQuadVReg(zs):
    '''
    * @Input:
    *    zs = n x m x 2 = keypoint observations from n frames (normalized pixel coordinates)
    * @Output:
    *    J = n x (2m+4+3m+3) x 45 (pose, shape deformation, mean shape deformation)
    '''

    zs = zs[None, ...] if zs.ndim < 3 else zs

    J = np.zeros((zs.shape[0], 3, 45))

    num_frames = J.shape[0]

    # ellipsoid shape deformation Regularization Jacobian
    # zs.shape[1] * 2 + 4 + 36 : zs.shape[1] * 2 + 4 + 36 + 3: for shape s deformation regularization
    # 6:9: for shape deformation in the object state
    J[:, :, 6:9] = np.tile(np.eye(3) / np.sqrt(num_frames), (J.shape[0], 1, 1))

    return J

def levenbergMarquardt(err_feature, err_bbox, err_d_reg, err_v_reg,
    jacobi_feature, jacobi_bbox, jacobi_d_reg, jacobi_v_reg,
    x, LMConfig, add_handle = lambda x, dx: x + dx):
    """

    Solves the problem:  min_x \sum_m w_m \|error_handle(x)\|^2

    using Levenberg-Marquardt and huber weights
    error_handle(x) = m x p = residuals
    jacobian_handle(x) = m x p x d = jacobians
    m = no. of frames
    p = no. of residuals
    d = dim. of object state

    :param err_feature:
    :param err_bbox:
    :param err_d_reg:
    :param err_v_reg:
    :param jacobi_feature:
    :param jacobi_bbox:
    :param jacobi_d_reg:
    :param jacobi_v_reg:
    :param x:
    :param add_handle:
    :param LMConfig: includes HUBER_EPSILON, WP, WL, WD, WV
    :return:
    """

    HUBER_EPSILON = LMConfig.HUBER_EPSILON
    WP, WL, WD, WV = LMConfig.WP, LMConfig.WL, LMConfig.WD, LMConfig.WV

    ## default parameters
    precision = 5e-7
    damping = 1e-3
    inner_loop_max_iteration = 50
    outer_loop_max_iteration = 50
    max_damping = 1e12
    min_damping = 1e-10

    is_inlier_flag = True

    # Compute residual and initial cost
    r_feat, c_feat, w_feat = computeResidual(err_feature, x, HUBER_EPSILON)

    # consider object as outlier if no. of frames is small
    num_frames = np.shape(r_feat)[0]
    if num_frames <= 2:
      is_inlier_flag = False

    r_bbox, c_bbox, w_bbox = computeResidual(err_bbox, x, HUBER_EPSILON)
    r_dr, c_dr, w_dr = computeResidual(err_d_reg, x, HUBER_EPSILON)
    r_vr, c_vr, w_vr = computeResidual(err_v_reg, x, HUBER_EPSILON)

    cost_total = WP * c_feat + WL * c_bbox + WD * c_dr + WV * c_vr

    outer_loop_cntr = 0
    delta_norm2 = np.inf
    A = []

    while (outer_loop_cntr < outer_loop_max_iteration and delta_norm2 > precision):

        # Compute Jacobian
        # m x p x d
        J_feat = jacobi_feature(x)
        # m x p x d
        J_bbox = jacobi_bbox(x)
        # m x p x d
        J_dr = jacobi_d_reg(x)
        # m x p x d
        J_vr = jacobi_v_reg(x)

        # m x d x p
        JT_feat = np.swapaxes(J_feat, -1, -2)
        # m x d x p
        JT_bbox = np.swapaxes(J_bbox, -1, -2)
        # m x d x p
        JT_dr = np.swapaxes(J_dr, -1, -2)
        # m x d x p
        JT_vr = np.swapaxes(J_vr, -1, -2)

        # eq 51 in http://ethaneade.com/optimization.pdf
        # m x d x d
        A_feat = w_feat[..., None] * (JT_feat @ J_feat)
        # d x d
        A_feat = np.sum(A_feat, axis=0)
        diagA_feat = np.diag(np.diag(A_feat))

        # m x d x d
        A_bbox = w_bbox[..., None] * (JT_bbox @ J_bbox)
        # d x d
        A_bbox = np.sum(A_bbox, axis=0)
        diagA_bbox = np.diag(np.diag(A_bbox))

        # m x d x d
        A_dr = w_dr[..., None] * (JT_dr @ J_dr)
        # d x d
        A_dr = np.sum(A_dr, axis=0)
        diagA_dr = np.diag(np.diag(A_dr))

        # m x d x d
        A_vr = w_vr[..., None] * (JT_vr @ J_vr)
        # d x d
        A_vr = np.sum(A_vr, axis=0)
        diagA_vr = np.diag(np.diag(A_vr))

        A = WP * A_feat + WL * A_bbox + WD * A_dr + WV * A_vr
        diagA = WP * diagA_feat + WL * diagA_bbox + WD * diagA_dr + WV * diagA_vr

        # eq 51 in http://ethaneade.com/optimization.pdf
        # m x d
        b_feat = np.squeeze(w_feat[..., None] * JT_feat @ r_feat[..., None], axis=2)
        # d x 1
        b_feat = np.sum(b_feat, axis=0)

        # m x d
        b_bbox = np.squeeze(w_bbox[..., None] * JT_bbox @ r_bbox[..., None], axis=2)
        # d x 1
        b_bbox = np.sum(b_bbox, axis=0)

        # m x d
        b_dr = np.squeeze(w_dr[..., None] * JT_dr @ r_dr[..., None], axis=2)
        # d x 1
        b_dr = np.sum(b_dr, axis=0)

        # m x d
        b_vr = np.squeeze(w_vr[..., None] * JT_vr @ r_vr[..., None], axis=2)
        # d x 1
        b_vr = np.sum(b_vr, axis=0)

        b = WP * b_feat + WL * b_bbox + WD * b_dr + WV * b_vr

        # Take gradient step
        inner_loop_cntr = 0
        is_cost_reduced = False

        while (inner_loop_cntr < inner_loop_max_iteration and not is_cost_reduced):

            dx = np.linalg.lstsq(A + damping * diagA, b, rcond=None)[0]

            # try:
            #   dx = np.linalg.lstsq(A + damping * diagA, b, rcond=None)[0]
            # except:
            #   is_inlier_flag = False
            #   dx = np.zeros(45)

            delta_norm2 = np.sum(dx ** 2)

            # new solution
            new_x = add_handle(x, dx)

            # compute new cost
            r_feat_new, c_feat_new, w_feat_new = computeResidual(err_feature, new_x, HUBER_EPSILON)
            r_bbox_new, c_bbox_new, w_bbox_new = computeResidual(err_bbox, new_x, HUBER_EPSILON)
            r_dr_new, c_dr_new, w_dr_new = computeResidual(err_d_reg, new_x, HUBER_EPSILON)
            r_vr_new, c_vr_new, w_vr_new = computeResidual(err_v_reg, new_x, HUBER_EPSILON)

            new_cost_total = WP * c_feat_new + WL * c_bbox_new + WD * c_dr_new + WV * c_vr_new

            if (new_cost_total < cost_total):

                is_cost_reduced = True

                r_feat, c_feat, w_feat = r_feat_new, c_feat_new, w_feat_new
                r_bbox, c_bbox, w_bbox = r_bbox_new, c_bbox_new, w_bbox_new
                r_dr, c_dr, w_dr = r_dr_new, c_dr_new, w_dr_new
                r_vr, c_vr, w_vr = r_vr_new, c_vr_new, w_vr_new

                cost_total = new_cost_total
                x = new_x

                damping = max(min_damping, damping / 10)
            else:
                is_cost_reduced = False
                damping = min(max_damping, damping * 10)

            inner_loop_cntr += 1

        outer_loop_cntr += 1

    # A is similar to info. matrix
    # its trace the smaller the fewer information
    tr_A = np.trace(A)
    # print("tr_A {}".format(tr_A))
    if tr_A <= LMConfig.trace_threshold:
        is_inlier_flag = False

    return x, cost_total, is_inlier_flag


def computeResidual(error_handle, x, HUBER_EPSILON = np.inf):
    '''
    Computes the weighted residual \sum_m w_m \|error_handle(x)\|^2
    with approximate Huber weights
    '''

    r = -error_handle(x) # m x p
    if np.isinf(HUBER_EPSILON):
        w = np.ones((r.shape[0], 1)) # m x 1
        c = np.sum(r ** 2)
    else:
        w = huberDerivative(r, HUBER_EPSILON)[..., None] # m x 1
        c = np.sum(huber(r, HUBER_EPSILON))

    return r, c, w

def huberDerivative(r, HUBER_EPSILON = np.inf):
    '''
    huber loss weights
    r = m x p
    w = m x 1
    note w is applied to r norm, not r
    Implements eq. (50) from http://ethaneade.com/optimization.pdf
    '''

    # m x 1 = derivative of huber function with respect to r
    w = np.ones((r.shape[0],))
    if not np.isinf(HUBER_EPSILON):
        # m x 1 = residual norm
        r_norm = np.sqrt(np.sum(r**2,axis=-1))
        large_norm_idx = r_norm >= HUBER_EPSILON
        # note that r_norm is sqrt of L in eq. 50
        w[large_norm_idx] = HUBER_EPSILON / r_norm[large_norm_idx]

    return w

def huber(r, HUBER_EPSILON = np.inf):
    '''
    converts residual r into huber cost
    r = m x p
    cost = m x 1
    Implements eq. (42) from http://ethaneade.com/optimization.pdf
    '''

    cost = np.sum(r ** 2, axis = -1)
    if not np.isinf(HUBER_EPSILON):
        he2 = HUBER_EPSILON ** 2
        large_idx = cost >= he2
        cost[large_idx] = 2.0 * HUBER_EPSILON * np.sqrt(cost[large_idx]) - he2

    return cost







def bbox2lineh(bbox):
    '''
    : bbox = n x [left (x_min), up (y_min), right (x_max), down (y_max)]
    : line = n x 4 x 3
    '''
    line = np.empty(bbox.shape[:-1]+(4,3))
    line[...,0,:] = pointh2lineh(point2homo(bbox[...,[0,1]]),point2homo(bbox[...,[2,1]]))
    line[...,1,:] = pointh2lineh(point2homo(bbox[...,[2,1]]),point2homo(bbox[...,[2,3]]))
    line[...,2,:] = pointh2lineh(point2homo(bbox[...,[2,3]]),point2homo(bbox[...,[0,3]]))
    line[...,3,:] = pointh2lineh(point2homo(bbox[...,[0,3]]),point2homo(bbox[...,[0,1]]))
    return line

def point2homo(p):
    '''
    returns the nx4 or nx3 homogeneous coordinates of the nx3 or nx2 points p
    '''
    ee = np.ones(p.shape[:-1]+(1,))

    return np.concatenate((p,ee), axis=-1)

def pointh2lineh(a,b):
    '''
    convert two 2D points (lines) in homogenous coordinates to a line (point)
    that passes through them, l = a x b
    a = nx3
    b = nx3
    l = nx3
    '''
    return np.cross(a,b)