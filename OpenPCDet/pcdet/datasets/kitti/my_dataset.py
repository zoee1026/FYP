import glob
import numpy as np
import pandas as pd
from pathlib import Path
import json
from ..dataset import DatasetTemplate
import copy

VehicaleClasses = {

    "one-box": 0,
    "three-box": 0,
    "two-box": 0,
    "black-one-box": 0,
    "black-three-box": 0,
    "black-two-box": 0,

    "taxi": 1,
    "privateminibus": 2,
    "publicminibus": 3,
    "motorbike": 4,
    "pedestrian": 5,

    "construction-vehicle": 6,
    "crane-truck": 6,
    "cylindrical-truck": 6,


    "black-cargo-mpv": 7,
    "cargo-mpv": 7,

    "black-mpv": 8,
    "mpv": 8,


    "smalltruck": 9,
    "black-smalltruck": 9,

    "black-cargo-one-box": 10,
    "cargo-one-box": 10,

    "mediumtruck": 11,
    "bigtruck": 12,
    "flatbed-truck": 13,
    "coachbus": 14,
    "dd": 15,
}


class MyDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names,
                         training=training, root_path=root_path, logger=logger)
        # point_file_list=glob.glob(str(l=- /'training/points/*.bin'))
        # labels_file_list=glob.glob(str(self.root_path/'training/labels/*.txt'))
        # point_file_list.sort()
        # labels_file_list.sort()
        self.sample_file_list = pd.read_csv(self.root_path)['lidar_files'].tolist()
        self.samplelabel_file_list = pd.read_csv(self.root_path)['label_files'].tolist()

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        sample_idx = Path(self.sample_file_list[index]).stem
        points = np.fromfile(
            self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        # loc, dims, rots loc_lidar, l, w, h
        # points=points[:,[2,0,1]]
        # points[:,0]=-points[:,0]
        # points[:,1]=-points[:,1]
        # points_label = np.loadtxt(
        #     self.samplelabel_file_list[index], dtype=np.float32).reshape(-1, 7)
        # points_label=points_label[:,[2,0,1,5,3,4,6]]

        with open(self.samplelabel_file_list[index]) as json_file:
            data = json.load(json_file)
            elements = []
            boundingBoxes = data['bounding_boxes']

            for box in boundingBoxes:
                if box["object_id"] not in list(VehicaleClasses.keys()):
                    continue
                element = list(box['center'].values())+[box['width'], box['length'],
                                                        box['height'], box['angle'], VehicaleClasses[str(box["object_id"])]]
                elements.append(element)

            points_label = np.array(elements)[:, :-1]
            gt_names = np.array(elements)[:, -1].reshape(-1, 1)

        input_dict = {
            'points': points,
            'frame_id': sample_idx,
            'gt_names': gt_names,
            'gt_boxes': points_label
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples),
                # 'truncated': np.zeros(num_samples),
                # 'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
                # 'bbox': np.zeros([num_samples, 4]),
                'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples)
                # , 'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(batch_index, box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            # calib = batch_dict['calib'][batch_index]
            # image_shape = batch_dict['image_shape'][batch_index].cpu().numpy()
            # pred_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(pred_boxes, calib)
            # pred_boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
            #     pred_boxes_camera, calib, image_shape=image_shape
            # )

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            # pred_dict['alpha'] = -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]
            # pred_dict['bbox'] = pred_boxes_img
            pred_dict['dimensions'] = pred_boxes[:, 3:6]
            pred_dict['location'] = pred_boxes[:, 0:3]
            pred_dict['rotation_y'] = pred_boxes[:, 6]
            pred_dict['score'] = pred_scores
            # pred_dict['boxes_lidar'] = pred_boxes

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict['frame_id'][index]

            single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)

            if output_path is not None:
                cur_det_file = output_path / ('%s.txt' % frame_id)
                with open(cur_det_file, 'w') as f:
                    bbox = single_pred_dict['name']
                    loc = single_pred_dict['location']
                    dims = single_pred_dict['dimensions']  # lhw -> hwl

                    for idx in range(len(bbox)):
                        print('%s %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                              % (single_pred_dict['name'][idx],
                                single_pred_dict['alpha'][idx],
                                dims[idx][0], dims[idx][1], dims[idx][2],
                                loc[idx][0],
                                loc[idx][1],
                                loc[idx][2],
                                single_pred_dict['rotation_y'][idx],
                                single_pred_dict['score'][idx]),
                                file=f)

        return annos

    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.kitti_infos[0].keys():
            return None, {}

        from .kitti_object_eval_python import eval as kitti_eval

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos'])
                         for info in self.kitti_infos]
        ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
            eval_gt_annos, eval_det_annos, class_names)

        return ap_result_str, ap_dict

# ————————————————
# 版权声明：本文为CSDN博主「CVplayer111」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
# 原文链接：https://blog.csdn.net/slamer111/article/details/127117402
