import os
# import cv2
from glob import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from processors import DataProcessor
from inference_utils import generate_bboxes_from_pred, rotational_nms, \
    dump_predictions, get_formated_label, ReadGTLabel, \
    pillar_net_predict_server, BBox, \
    cal_precision, Get_finalPrecisions

from readers import KittiDataReader
from config import Parameters, OutPutVehecleClasees
from nets.network import build_point_pillar_graph
# from nets.network_yolo import build_point_pillar_graph
# from nets.network_yolo_basechannel import build_point_pillar_graph
# from nets.network_yolo_concat import build_point_pillar_graph
import argparse
import logging
from easydict import EasyDict as edict
import time
from tqdm import tqdm
from read_file_location import ReadFileFromPath
import h5py


precisions = {
    'TP': 0,
    'FP': 0,
    0: [],
    1: [],
    2: [],
    3: [],
    4: [],
    5: [],
    6: [],
    7: [],
    8: [],
    9: [],
    10: [],
    11: [],
    12: [],
    13: [],
    14: [],
    15: [],
}
RESULT_LABEL_CSV='/media/sdb1/zoe/FYP/folder_root/Val2.csv'
MODEL='zoe_pointpillars.h5'
EVAL_PATH='test.csv'
MODEL_ROOT = "./log"
MODEL_SAVE = "train3.h5"
pretrained= os.path.join(MODEL_ROOT,MODEL_SAVE)
# EVAL_PATH='/media/sdb1/zoe/FYP/folder_root/Eval_CleanFiles.csv'
SAVE=False

def generate_config_from_cmd_args():
    parser = argparse.ArgumentParser(
        description='PointPillars inference on test data.')
    parser.add_argument('--data_root', default=EVAL_PATH, type=str, required=False,
                        help='Test data root path holding folders velodyne, calib')
    parser.add_argument('--result_dir', default="./Result", type=str, required=False,
                        help='Path for dumping result labels in KITTI format')
    parser.add_argument('--model_path', default=MODEL, type=str, required=False,
                        help='Path to the model weights to be used for inference')
    parser.add_argument('--occ_thresh', default=0.4, type=float, required=False,
                        help='Occlusion threshold for predicted boxes')
    parser.add_argument('--nms_thresh', default=0.5, type=float, required=False,
                        help='IoU threshold for NMS')

    configs = edict(vars(parser.parse_args()))
    return configs

    # return image_file_names, lidar_file_names, calib_file_names


def load_model_and_run_inference(configs):
    params = Parameters()  # Load all model related parameters
    csv =pd.DataFrame()
    pillar_net = build_point_pillar_graph(params, batch_size=1)

    logging.info("Loading model from path: {}".format(configs.model_path))

    pillar_net.load_weights(configs.model_path)

    logging.info("Model loaded.=================================================")

    lidar_files, label_files = ReadFileFromPath(configs.data_root)

    data_reader = KittiDataReader()
    point_cloud_processor = DataProcessor()
    model_exec_time = []

    out_labels_path = os.path.join(configs.result_dir, "labels")
    os.makedirs(out_labels_path, exist_ok=True)

    for idx in tqdm(range(len(lidar_files))):
        file_name = lidar_files[idx].split('.')[0]
        file_name = file_name.split("/")[-1]
        logging.debug("Running for file: {}".format(file_name))
        lidar_data = data_reader.read_lidar(lidar_files[idx])

        pillars, voxels = point_cloud_processor.make_point_pillars(
            points=lidar_data, print_flag=False)

        start = time.time()
        occupancy, position, size, angle, heading, classification = pillar_net_predict_server(
            [pillars, voxels], pillar_net)
        stop = time.time()
        print(
            "Single frame PointPillars inference time using predict server: {}".format(stop-start))
        model_exec_time.append(stop-start)

        logging.debug("occupancy shape: {}".format(occupancy.shape))
        logging.debug("position shape: {}".format(position.shape))
        logging.debug("size shape: {}".format(size.shape))
        logging.debug("angle shape: {}".format(angle.shape))
        logging.debug("heading shape: {}".format(heading.shape))
        logging.debug("classification shape: {}".format(classification.shape))

        if occupancy.shape[0] == 1:
            logging.debug(
                "Single image inference, reducing batch dim for output tensors")
            occupancy = np.squeeze(occupancy, axis=0)
            position = np.squeeze(position, axis=0)
            size = np.squeeze(size, axis=0)
            angle = np.squeeze(angle, axis=0)
            heading = np.squeeze(heading, axis=0)
            classification = np.squeeze(classification, axis=0)

        start = time.time()
        boxes = generate_bboxes_from_pred(occupancy, position, size, angle, heading, classification,
                                          params.anchor_dims, occ_threshold=configs.occ_thresh)
        stop = time.time()
        confidences = [float(box.conf) for box in boxes]
        print(len(confidences))

        start = time.time()
        nms_indices = rotational_nms(
            boxes, confidences, occ_threshold=configs.occ_thresh, nms_iou_thr=configs.nms_thresh)
        stop = time.time()
        logging.debug(
            "Single frame rotational NMS time: {}".format(stop-start))

        logging.debug("Number of boxes post-nms: {}".format(len(nms_indices)))

        # Print out prediction
        print(len(boxes))
        nms_boxes = [boxes[i] for i in nms_indices]
        for i in nms_boxes:
            print(i)
        print(len(nms_boxes))
        gt = ReadGTLabel(label_files[idx])
        cal_precision(nms_boxes, gt, precisions)

        if SAVE:
            csv=dump_predictions(get_formated_label(boxes, nms_indices), os.path.join(
                out_labels_path, "{}.txt".format(file_name)),csv)
        print('----------------------------------------------------------------------------')
    

    model_exec_time = model_exec_time[1:]
    total_model_exec_time = sum(model_exec_time)
    model_fps = len(model_exec_time) / total_model_exec_time
    logging.info("PointPillars model inference FPS: {}".format(model_fps))

    if SAVE:
        print('----------------------------------------------------------------')
        print(csv.info())
        csv.to_csv(RESULT_LABEL_CSV)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - [%(levelname)s]: %(message)s")
    pred_config = generate_config_from_cmd_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '1,3'
    # os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    logging.info("Results will be saved at path: {}".format(
        pred_config.result_dir))
    
    load_model_and_run_inference(pred_config)
    Get_finalPrecisions(precisions)


