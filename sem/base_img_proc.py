from abc import ABC, abstractmethod
import numpy as np

import sem.myobject

class ImageProcessor(ABC):
    """
    this is the base class of the image processor
    """

    def __init__(self, K):
        """
        inits the essential variables for the image processor
        :param K: size 3x3, intrinsic matrix used for normalizing pixels
        """

        self.K = K

        self.img_id = 0

        # ID for the next new feature.
        self.next_feature_id = 0

        # dictionary that contains all features to be published
        # key is feat id, value is img id and all observations
        self.obs_all_dict = {}

    def img_callback(self, img_msg):
        """
        callback function to process the image msg received
        :param img_msg: contains color and grayscale images
        :param img_id: the index of the image
        :return: features to be published
        """

        self.img_id = img_msg.img_id

        self.add_new_frame(img_msg)

        return self.publish_features()

    def add_new_frame(self, img_msg = None):
        """
        add new observations from each frame
        :param img_msg: color and grayscale image
        """

        pass

    def add_img_id_to_obs_dict(self, feat_id, img_id):
        """
        add image id to obs_all_dict
        """

        self.obs_all_dict[feat_id]['img_id'].append(img_id)

    def add_zb_to_obs_dict(self, feat_id, bbox):
        """
        add bbox to obs_all_dict
        :param feat_id: id of feature
        :param bbox: size 1x4, bbox
        """

        zb = self.obs_all_dict[feat_id]['zb']
        zb = np.concatenate((zb, bbox), axis=0)
        self.obs_all_dict[feat_id]['zb'] = zb

    def add_R0_to_obs_dict(self, feat_id, new_R0):
        """
        add R0 to obs_all_dict
        :param feat_id: id of feature
        :param R0: size 3x3, initial rotation matrix
        """

        R0 = self.obs_all_dict[feat_id]['R0']
        R0 = np.concatenate((R0, np.reshape(new_R0, (1, -1, 3))), axis=0)
        self.obs_all_dict[feat_id]['R0'] = R0

    def add_zg_to_obs_dict(self, feat_id, new_zg):
        """
        add zg to obs_all_dict
        :param feat_id: id of feature
        :param new_zg: size 1x2, new geometric feature to be add
        """

        zg = self.obs_all_dict[feat_id]['zg']

        zg = np.concatenate((zg, new_zg), axis=0)

        self.obs_all_dict[feat_id]['zg'] = zg

    def init_zs_obs_dict_new_frame(self, feat_id):
        """

        :return:
        """

        zs = self.obs_all_dict[feat_id]['zs']
        zs = np.concatenate((zs, np.full([1, sem.myobject.NUM_KEYPOINTS, 2], np.nan)), axis=0)
        self.obs_all_dict[feat_id]['zs'] = zs

    def add_zs_to_obs_dict(self, feat_id, new_zs):
        """
        add zs to obs_all_dict
        :param feat_id: id of feature
        :param new_zs: size 12x2, new semantic feature to be add
        """

        if 'zs' not in self.obs_all_dict[feat_id]:
            self.obs_all_dict[feat_id]['zs'] = np.zeros((0, sem.myobject.NUM_KEYPOINTS, 2))

        new_zs = new_zs[None, ...] if new_zs.ndim < 3 else new_zs

        zs = self.obs_all_dict[feat_id]['zs']

        zs = np.concatenate((zs, new_zs), axis=0)

        self.obs_all_dict[feat_id]['zs'] = zs

    def publish_features(self):
        """
        publish features that are lost
        """

        pass

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
utils functions 
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def normalize_pixel(K, raw_pixel):
    """
    normalize the pixel using intrinsics in batch
    :param K: size 3x3, intrinsic matrix
    :param raw_pixel: size nx2 or nxmx2, array that contains all coordinates
    :return: size nx2 or nxmx2, tuple that contains all normalized coordinates
    """

    # K[[0, 1], [2, 2]] means K[0, 2], K[1, 2], ie cx, cy
    # K[[0, 1], [0, 1]] refers K[0, 0], K[1, 1], ie fx, fy

    normalized_pixel = (raw_pixel - K[[0, 1], [2, 2]]) / K[[0, 1], [0, 1]]

    return normalized_pixel