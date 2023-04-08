import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import json
import copy
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ..dataset import DatasetTemplate

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
        self.sample_file_list = pd.read_csv('/media/sdb1/zoe/FYP/folder_root/All.csv')['lidar_files'].tolist()
        self.samplelabel_file_list = pd.read_csv('/media/sdb1/zoe/FYP/folder_root/All.csv')['label_files'].tolist()
        self.infos = []

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):

        # loc, dims, rots loc_lidar, l, w, h
        # points=points[:,[2,0,1]]
        # points[:,0]=-points[:,0]
        # points[:,1]=-points[:,1]
        # points_label = np.loadtxt(
        #     self.samplelabel_file_list[index], dtype=np.float32).reshape(-1, 7)
        # points_label=points_label[:,[2,0,1,5,3,4,6]]

        info={}
        sample_idx = Path(self.sample_file_list[index]).stem
        pc_info = {'num_features': 4, 'lidar_idx': sample_idx}

        info['point_cloud'] = pc_info

        points = np.fromfile(
            self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)


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

            elements=np.array(elements)
            points_label = elements[:, :-1]
            gt_names = elements[:, -1].reshape(-1, 1)

        input_dict = {
            'points': points,
            'frame_id': sample_idx,
            'gt_names': gt_names,
            'gt_boxes': points_label
        }

        annotations={
            'name':gt_names,
            'dimesions':elements[:, :3],
            'location':elements[:, 3:6],
            'rotation_y':elements[:, 6]
                     }

        info['annos'] = annotations
        self.infos.append(info)

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        import torch

        database_save_path = Path(self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
        db_info_save_path = Path(self.root_path) / ('kitti_dbinfos_%s.pkl' % split)

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        for k in range(len(infos)):
             print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]
            sample_idx = info['point_cloud']['lidar_idx']
            points = self.get_lidar(sample_idx)
            annos = info['annos']
            names = annos['name']
            difficulty = annos['difficulty']
            bbox = annos['bbox']
            gt_boxes = annos['gt_boxes_lidar']

            num_obj = gt_boxes.shape[0]
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[point_indices[i] > 0]

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0],
                               'difficulty': difficulty[i], 'bbox': bbox[i], 'score': annos['score'][i]}
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

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
                'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples)
            }
            return ret_dict

        def generate_single_sample_dict(batch_index, box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['dimensions'] = pred_boxes[:, 3:6]
            pred_dict['location'] = pred_boxes[:, 0:3]
            pred_dict['rotation_y'] = pred_boxes[:, 6]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes

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
                                loc[idx][0], loc[idx][1], loc[idx][2],
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

# def create_kitti_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
#     dataset = MyDataset(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False)
#     train_split, val_split = 'train', 'val'

#     train_filename = save_path / ('kitti_infos_%s.pkl' % train_split)
#     val_filename = save_path / ('kitti_infos_%s.pkl' % val_split)
#     trainval_filename = save_path / 'kitti_infos_trainval.pkl'
#     test_filename = save_path / 'kitti_infos_test.pkl'

#     print('---------------Start to generate data infos---------------')

#     dataset.set_split(train_split)
#     kitti_infos_train = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
#     with open(train_filename, 'wb') as f:
#         pickle.dump(kitti_infos_train, f)
#     print('Kitti info train file is saved to %s' % train_filename)

#     dataset.set_split(val_split)
#     kitti_infos_val = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
#     with open(val_filename, 'wb') as f:
#         pickle.dump(kitti_infos_val, f)
#     print('Kitti info val file is saved to %s' % val_filename)

#     with open(trainval_filename, 'wb') as f:
#         pickle.dump(kitti_infos_train + kitti_infos_val, f)
#     print('Kitti info trainval file is saved to %s' % trainval_filename)

#     dataset.set_split('test')
#     kitti_infos_test = dataset.get_infos(num_workers=workers, has_label=False, count_inside_pts=False)
#     with open(test_filename, 'wb') as f:
#         pickle.dump(kitti_infos_test, f)
#     print('Kitti info test file is saved to %s' % test_filename)

#     print('---------------Start create groundtruth database for data augmentation---------------')
#     dataset.set_split(train_split)
#     dataset.create_groundtruth_database(train_filename, split=train_split)

#     print('---------------Data preparation Done---------------')


# if __name__ == '__main__':
#     import sys
#     if sys.argv.__len__() > 1 and sys.argv[1] == 'create_kitti_infos':
#         import yaml
#         from pathlib import Path
#         from easydict import EasyDict
#         dataset_cfg = EasyDict(yaml.safe_load(open(sys.argv[2])))
#         ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
#         create_kitti_infos(
#             dataset_cfg=dataset_cfg,
#             class_names=['Car', 'Pedestrian', 'Cyclist'],
#             data_path=ROOT_DIR / 'data' / 'kitti',
#             save_path=ROOT_DIR / 'data' / 'kitti'
#         )

