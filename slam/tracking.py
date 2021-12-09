import cv2
import numpy as np
import transforms3d as tf 

import g2o
from optimization import BundleAdjustment

class Tracking(object):
    def __init__(self, params):
        self.min_measurements = params.pnp_min_measurements
        self.max_iterations = params.pnp_max_iterations
        self.optimizer = BundleAdjustment()

    def refine_pose(self, pose, cam, measurements):
        if len(measurements) < self.min_measurements: 
            return pose 
        success_flag, pose_opt = self.pnp(cam, pose, measurements)
        if not success_flag: 
            self.optimizer.clear()            
            self.optimizer.add_pose(0, pose, cam, fixed=False)
            for i, m in enumerate(measurements):
                self.optimizer.add_point(i, m.mappoint.position, fixed=True)
                self.optimizer.add_edge(0, i, 0, m)
            self.optimizer.optimize(self.max_iterations)            
            pose_opt = self.optimizer.get_pose(0)
        
        return pose_opt

    def pnp(self, cam, pose, measurements):
        success_flag = True  
        # N x 3 points in 3D 
        model_points = np.zeros((0, 3))
        # N x 2 image points 
        image_points = np.zeros((0, 2))
        for _, m in enumerate(measurements):
            pt3d = np.reshape(m.mappoint.position, (-1, 3))
            model_points = np.vstack((model_points, pt3d))
            pt2d = np.reshape(m.xy, (-1, 2))
            image_points = np.vstack((image_points, pt2d))

        camera_matrix = np.array(
                [[cam.fx, 0, cam.cx],
                [0, cam.fy, cam.cy],
                [0, 0, 1]], dtype = "double"
                )
        dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion

        pose_inv = pose.inverse()
        direc, angle = tf.axangles.mat2axangle(pose_inv.orientation().matrix())
        raux = angle * direc
        taux = pose_inv.position()

        # use initial guess 
        retval, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(model_points, 
            image_points, camera_matrix, dist_coeffs, 
            raux, taux, useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE)

        # do not use initial guess 
        # retval, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(model_points, 
        #     image_points, camera_matrix, dist_coeffs,
        #     flags=cv2.SOLVEPNP_ITERATIVE)

        if inliers is None or len(inliers) < 200: 
            success_flag = False
            return success_flag, pose 
        print(f"PnP ransac inliers no.: {len(inliers)}")

        rotation_vector = np.squeeze(rotation_vector)
        angle = np.linalg.norm(rotation_vector)
        dirc = rotation_vector / angle 
        q = tf.quaternions.axangle2quat(dirc, angle)
        t = translation_vector
        pose_out = g2o.Isometry3d(g2o.Quaternion(q), t)
        return success_flag, pose_out.inverse()