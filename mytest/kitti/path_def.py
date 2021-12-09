import os, sys

class PathGenerator():

    def __init__(self, date_name, drive_name):
        """

        :param date: date of kitti sequence
        :param drive: drive of kitti sequence
        """

        # find out the root path of the directory
        self.dir_path = os.path.abspath(os.path.join(__file__, "../../../"))

        # get the kitti path
        self.kitti_date = date_name
        self.kitti_drive = drive_name

        # get cache path
        self.cache_path = self.dir_path + '/mytest/' + 'cache/'
        # self.cache_path = '/media/disk1/orcvio_cache/'

        # generate the path
        self.generate_kitti_path()
        self.generate_sim_path()

        self.generate_gt_bbox_path()

        self.generate_yolo_path()
        self.generate_starmap_path()

        self.generate_traj_visualization_path()
        self.generate_traj_eval_path()

        self.generate_tracking_visualization_path()

        self.generate_object_eval_path()

        self.generate_msckf_path()

    def generate_kitti_path(self):

        using_ubuntu = True
        # using_ubuntu = False

        if using_ubuntu:
            self.kitti_dataset_path = '/media/erl/disk2/kitti'
        else:
            self.kitti_dataset_path = '/Users/moshan/Documents/PhD/research/datasets/kitti'

        # self.kitti_dir = self.kitti_date + '_' + self.kitti_drive + '/'
        self.kitti_dir = self.kitti_date + '_' + self.kitti_drive

        # for loading tracklets
        self.drive_path = self.kitti_date + '_drive_' + self.kitti_drive + '_sync'

        self.tracklet_xml_path = os.path.join(self.kitti_dataset_path, self.kitti_date,
                                    self.drive_path, "tracklet_labels.xml")

    def generate_sim_path(self):

        self.sim_data_path = self.dir_path + '/mytest/data/'

        # seq. name that contains the measurements
        seq_path = '2011_09_30/2011_09_30_drive_0027_sync/'

        # for kitti simulation
        self.kps_2d_file = self.dir_path + '/mytest/data/' + seq_path + "kps_fov.pkl"
        self.kps_positions_file = self.dir_path + '/mytest/data/' + seq_path + "/kps_gt.pkl"

    def generate_yolo_path(self):

        # for yolo
        self.yolo_path = self.dir_path + '/third_party/yolov3/'

        # note, need to put pytorch_models inside third_party folder
        self.yolo_weights_path = self.dir_path + '/third_party/pytorch_models/yolo/trained_model/yolov3.weights'

        self.yolo_results_path = self.cache_path + self.kitti_dir + '/yolo_results/'

        if not os.path.exists(self.yolo_results_path):
            os.makedirs(self.yolo_results_path)

    def generate_traj_visualization_path(self):

        self.traj_vis_save_dir = self.cache_path + self.kitti_dir + '/traj_plt_figs/'

        if not os.path.exists(self.traj_vis_save_dir):
            os.makedirs(self.traj_vis_save_dir)

    def generate_traj_eval_path(self):

        self.traj_eval_result_dir = self.cache_path + self.kitti_dir + '/evaluation/'

        if not os.path.exists(self.traj_eval_result_dir):
            os.makedirs(self.traj_eval_result_dir)

    def generate_gt_bbox_path(self):

        self.gt_bbox_results_path = self.cache_path + self.kitti_dir + '/gt_bboxes_results/'

        if not os.path.exists(self.gt_bbox_results_path):
            os.makedirs(self.gt_bbox_results_path)

    def generate_tracking_visualization_path(self):

        self.tracking_vis_save_dir = self.cache_path + self.kitti_dir + '/tracking_plt_figs/'

        if not os.path.exists(self.tracking_vis_save_dir):
            os.makedirs(self.tracking_vis_save_dir)

    def generate_starmap_path(self):

        self.starmap_model_path = self.dir_path + '/third_party/pytorch_models/starmap/trained_models/no_dropout/model_cpu.pth'

        self.starmap_results_path = self.cache_path + self.kitti_dir + '/starmap_results/'

        if not os.path.exists(self.starmap_results_path):
            os.makedirs(self.starmap_results_path)

    def generate_object_eval_path(self):

        self.pr_table_dir = self.cache_path + self.kitti_dir + '/evaluation/'

        if not os.path.exists(self.pr_table_dir):
            os.makedirs(self.pr_table_dir)

    def generate_msckf_path(self):

        self.map_server_path = self.cache_path + self.kitti_dir + '/msckf_results/'

        if not os.path.exists(self.map_server_path):
            os.makedirs(self.map_server_path)