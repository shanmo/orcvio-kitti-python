import numpy as np

import sem.se3

# smaller object initial scale
mean_v = np.array([1.3, 2.8, 1], dtype=float)

# load 12 kp model
# scaled version
# still need to use the original version for part names
# cannot use the scaled version
mean_shape = np.array([[-0.5757676, 0.45692027, 0.42172673, -0.55109255, -0.63167104,
                 0.50077807, 0.4336268, -0.59680422, -0.86061979, -0.79022376,
                 0.72905389, 0.64262001],
                [-0.32666928, -0.31517707, 0.87412286, 0.87559074, -2.00922593,
                 -1.97837645, 1.85077407, 1.83533425, -1.32156177, 1.15851448,
                 -1.29796222, 1.18035249],
                [0.84727895, 0.84746775, 0.86659066, 0.86754665, -0.10128908,
                 -0.10095322, 0.0928674, 0.09735433, -0.65325335, -0.53458293,
                 -0.65267643, -0.54863687]])

NUM_KEYPOINTS = 12

class MyObject():
    """
    this class defines the object state
    """

    def __init__(self):

        self.init_quadric()
        self.init_kps()

    def init_quadric(self):
        """
        init ellipsoid
        """

        # If v is a 1-D array, return a 2-D array with v on the k-th diagonal
        # AND note init v value has to be float, not int
        self.v = np.copy(mean_v)

        self.Q = np.array([[1.0,  0.0, 0.0, 0.0],
                            [0.0,  1.0, 0.0, 0.0],
                            [0.0,  0.0, 1.0, 0.0],
                            [0.0,  0.0, 0.0,-1.0]])

        self.wTq = sem.se3.SE3(np.array([
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]]
        ), np.array([0., 0., 0.]))

        # to hold results after init,
        # before lm
        self.wTq_init = sem.se3.SE3(np.array([
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]]
        ), np.array([0., 0., 0.]))

    def init_kps(self):
        """
        init semantic keypoints
        """

        # we want 12 x 3 instead of 3 x 12
        self.mean_shape = mean_shape.T

        # init deformation
        self.deformation = np.zeros((NUM_KEYPOINTS, 3), dtype=float)

        # 12 x 3 semantic keypoints in world frame
        self.object_shape = np.copy(self.mean_shape)

    def set_wTq(self, wTq):
        """
        set object to world transformation
        :param wTq: size 4x4, transformation matrix from object to world
        :return:
        """

        wRq = wTq[:3, :3]
        wPq = wTq[:3, 3]

        self.wTq = sem.se3.SE3(wRq, wPq)

    def set_kps_shape(self):
        """
        set object shape to keypoints in world frame
        """

        for i in range(NUM_KEYPOINTS):
            kp = self.mean_shape[i, :] + self.deformation[i, :]
            new_kp = self.wTq.R @ kp + self.wTq.t
            self.object_shape[i, :] = np.squeeze(new_kp)

    def get_Q_from_v(self):

        self.Q[0, 0], self.Q[1, 1], self.Q[2, 2] = \
            self.v[0] ** 2, self.v[1] ** 2, self.v[2] ** 2

        return self.Q