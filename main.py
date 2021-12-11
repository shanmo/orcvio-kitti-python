import numpy as np
import time
from collections import defaultdict
import transforms3d as tf 
import argparse
from threading import Thread

from slam.components import Camera
from slam.components import StereoFrame
from slam.components import Measurement
from slam.feature import ImageFeature
from slam.params import ParamsKITTI
from slam.dataset import KITTIOdometry
from slam.covisibility import CovisibilityGraph
from slam.mapping import MappingThread
from slam.motion import MotionModel
from slam.tracking import Tracking

import sem.sem_img_proc
import sem.message
import sem.visualization
import sem.feature_processor
import mytest.kitti.path_def

import g2o

class SPTAM(object):
    def __init__(self, params):
        self.params = params
        self.tracker = Tracking(params)
        self.motion_model = MotionModel()

        self.graph = CovisibilityGraph()
        self.mapping = MappingThread(self.graph, params)
        
        self.reference = None        # reference keyframe
        self.preceding = None        # last keyframe
        self.current = None          # current frame
        self.status = defaultdict(bool)

        self.object_feature_proc = None 
        self.object_level_map = None 

    def stop(self):
        self.mapping.stop()

    def initialize(self, frame):
        mappoints, measurements = frame.triangulate()
        assert len(mappoints) >= self.params.init_min_points, (
            'Not enough points to initialize map.')

        keyframe = frame.to_keyframe()
        keyframe.set_fixed(True)
        self.graph.add_keyframe(keyframe)
        self.mapping.add_measurements(keyframe, mappoints, measurements)

        self.reference = keyframe
        self.preceding = keyframe
        self.current = keyframe
        self.status['initialized'] = True

        self.motion_model.update_pose(
            frame.timestamp, frame.position, frame.orientation)

    def track(self, frame):
        while self.is_paused():
            time.sleep(1e-4)
        self.set_tracking(True)

        self.current = frame
        print('Tracking:', frame.idx, ' <- ', self.reference.id, self.reference.idx)

        predicted_pose, _ = self.motion_model.predict_pose(frame.timestamp)
        frame.update_pose(predicted_pose)

        local_mappoints = self.filter_points(frame)
        measurements = frame.match_mappoints(
            local_mappoints, Measurement.Source.TRACKING)

        print('measurements:', len(measurements), '   ', len(local_mappoints))

        tracked_map = set()
        for m in measurements:
            mappoint = m.mappoint
            mappoint.update_descriptor(m.get_descriptor())
            mappoint.increase_measurement_count()
            tracked_map.add(mappoint)
        
        self.reference = self.graph.get_reference_frame(tracked_map)
        pose = self.tracker.refine_pose(frame.pose, frame.cam, measurements)

        # update pose using object residual 
        self.object_feature_proc.add_cam_poses(frame.pose, frame.idx)
        pose = self.object_feature_proc.feature_callback(feat_obs_published)
        self.object_feature_proc.object_level_map = self.object_feature_proc.map_server

        frame.update_pose(pose)
        self.motion_model.update_pose(
            frame.timestamp, frame.pose.position(), frame.pose.orientation())

        if self.should_be_keyframe(frame, measurements):
            print('new keyframe', frame.idx)
            keyframe = frame.to_keyframe()
            keyframe.update_reference(self.reference)
            keyframe.update_preceding(self.preceding)

            self.mapping.add_keyframe(keyframe, measurements)
            self.preceding = keyframe

        self.set_tracking(False)


    def filter_points(self, frame):
        local_mappoints = self.graph.get_local_map_v2(
            [self.preceding, self.reference])[0]

        can_view = frame.can_view(local_mappoints)
        print('filter points:', len(local_mappoints), can_view.sum(), 
            len(self.preceding.mappoints()),
            len(self.reference.mappoints()))
        
        checked = set()
        filtered = []
        for i in np.where(can_view)[0]:
            pt = local_mappoints[i]
            if pt.is_bad():
                continue
            pt.increase_projection_count()
            filtered.append(pt)
            checked.add(pt)

        for reference in set([self.preceding, self.reference]):
            for pt in reference.mappoints():  # neglect can_view test
                if pt in checked or pt.is_bad():
                    continue
                pt.increase_projection_count()
                filtered.append(pt)

        return filtered


    def should_be_keyframe(self, frame, measurements):
        if self.adding_keyframes_stopped():
            return False

        n_matches = len(measurements)
        n_matches_ref = len(self.reference.measurements())

        print('keyframe check:', n_matches, '   ', n_matches_ref)

        return ((n_matches / n_matches_ref) < 
            self.params.min_tracked_points_ratio) or n_matches < 20

    def is_initialized(self):
        return self.status['initialized']

    def pause(self):
        self.status['paused'] = True

    def unpause(self):
        self.status['paused'] = False

    def is_paused(self):
        return self.status['paused']

    def is_tracking(self):
        return self.status['tracking']

    def set_tracking(self, status):
        self.status['tracking'] = status

    def stop_adding_keyframes(self):
        self.status['adding_keyframes_stopped'] = True

    def resume_adding_keyframes(self):
        self.status['adding_keyframes_stopped'] = False

    def adding_keyframes_stopped(self):
        return self.status['adding_keyframes_stopped']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_viz', action='store_true', help='do not visualize')
    parser.add_argument('--odom', type=str, help='odom idx', 
        default="06")
    parser.add_argument('--load_det', type=str, help='whether to load saved front end detections', 
        default="True")
    args = parser.parse_args()

    params = ParamsKITTI()
    odom_path = "/mnt/disk2/kitti/Kitti_all_data/odometry/dataset/sequences/"
    dataset = KITTIOdometry(odom_path+args.odom)
    sptam = SPTAM(params)

    visualize = not args.no_viz
    if visualize:
        from slam.viewer import MapViewer
        viewer = MapViewer(sptam, params)

    cam = Camera(
        dataset.cam.fx, dataset.cam.fy, dataset.cam.cx, dataset.cam.cy, 
        dataset.cam.width, dataset.cam.height, 
        params.frustum_near, params.frustum_far, 
        dataset.cam.baseline)

    # front end 
    load_detection_flag = args.load_det 
    if args.odom == "06": 
        kitti_date = "2011_09_30"
        kitti_drive = "0020"
        kitti_end_index = 1100
    PG = mytest.kitti.path_def.PathGenerator(kitti_date, kitti_drive)
    IP = sem.sem_img_proc.SemImageProcessor(dataset.cam, (dataset.cam.width, dataset.cam.height), kitti_end_index-1, PG, load_detection_flag)
    FTV = sem.visualization.FeatureTrackingVis()
    OFP = sem.feature_processor.ObjectFeatProcessor()
    sptam.object_feature_proc = OFP

    durations = []
    for i in range(len(dataset)):
        featurel = ImageFeature(dataset.left[i], params)
        featurer = ImageFeature(dataset.right[i], params)
        timestamp = dataset.timestamps[i]

        time_start = time.time()  
        t = Thread(target=featurer.extract)
        t.start()
        featurel.extract()
        t.join()
        
        frame = StereoFrame(i, g2o.Isometry3d(), featurel, featurer, cam, timestamp=timestamp)

        # process object features 
        feat_obs_published = IP.img_callback(sem.message.img_msg(frame.image, i))
        sptam.current.image = FTV.plot_all(frame.image, IP.bbox_trackers, IP.my_tracker.kps_tracker, i)

        if not sptam.is_initialized():
            sptam.initialize(frame)
        else:
            sptam.track(frame)

        duration = time.time() - time_start
        durations.append(duration)
        print('duration', duration)
        print()
        print()
        
        if visualize:
            viewer.update()

    print('num frames', len(durations))
    print('num keyframes', len(sptam.graph.keyframes()))
    print('average time', np.mean(durations))


    sptam.stop()
    if visualize:
        viewer.stop()