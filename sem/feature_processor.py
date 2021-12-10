from collections import OrderedDict
import numpy as np

import sem.object_feature
import sem.se3 

class ObjectFeatProcessor: 
    def __init__(self): 
        # for object features
        # <FeatureID, Feature>
        self.map_server = OrderedDict()
        # we need a dictionary to store a window of camera poses
        self.cam_states = OrderedDict()

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