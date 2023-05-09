import os
import sys
import time
import logging
import argparse
import pickle
import numpy as np
import tensorflow as tf
from easydict import EasyDict as edict

from config import Parameters
from loss import PointPillarNetworkLoss
# from nets.network_yolo_FPN import build_point_pillar_graph
# from nets.network_yolo_3feature import build_point_pillar_graph
from nets.network_yolo_4features import build_point_pillar_graph
from processors import SimpleDataGenerator
from readers import KittiDataReader
import h5py
from read_file_location import GetMatchedDatafile, TestModel
import pandas as pd

import wandb

# tf.get_logger().setLevel("ERROR")
# wandb.init(config=tf.compat.v1.flags.FLAGS, sync_tensorboard=True)

# DATA_ROOT = '..folder_root//MatchFile.csv'
DATA_ROOT = '/media/sdb1/zoe/FYP/folder_root/All.csv'
MODEL_ROOT = "./log"
MODEL_SAVE = "train21.h5"
pb_MODEL = 'my_model21'

zoe_pointpillars = 'zoe_pp_yolo11.h5'


def train_PillarNet():

    strategy = tf.distribute.MirroredStrategy()
    params = Parameters()

    kdt = pickle.load(open('kdt.sav', 'rb'))
    y_map=pd.read_csv('knn_y_map.csv')['angle'].values

    BATCH_SIZE_PER_REPLICA = params.batch_size
    BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

    with strategy.scope():

        pillar_net = build_point_pillar_graph(
            params, batch_size=BATCH_SIZE_PER_REPLICA, gpu=strategy.num_replicas_in_sync)
        pretrained = os.path.join(MODEL_ROOT, MODEL_SAVE)
        if os.path.exists(zoe_pointpillars):
            # with h5py.File('zoe_pointpillars3', 'r') as f:
            #     pillar_net.load_weights(f)
            pillar_net.load_weights(zoe_pointpillars)

            logging.info(
                "Using pre-trained weights found at path: {}".format(zoe_pointpillars))

            print("load")
        else:
            logging.info(
                "No pre-trained weights found. Initializing weights and training model.")

    data_reader = KittiDataReader()

    lidar_train, label_train, lidar_val, label_val = GetMatchedDatafile(
        DATA_ROOT)
    print(len(lidar_train), len(label_train),
          '---------------------------------------------------------------------')
    print(len(lidar_val), len(label_val),
          '---------------------------------------------------------------------')

    # "Input dirs require equal number of files."
    assert len(lidar_train) == len(label_train)
    assert len(lidar_val) == len(label_val)

    training_gen = SimpleDataGenerator(
        data_reader, BATCH_SIZE, lidar_train, kdt,y_map, label_train)
    validation_gen = SimpleDataGenerator(
        data_reader, BATCH_SIZE, lidar_val, kdt,y_map, label_val)

    with strategy.scope():
        loss = PointPillarNetworkLoss(params)

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=params.learning_rate)

        pillar_net.compile(optimizer, loss=loss.losses())
        epoch_to_decay = int(np.round(params.iters_to_decay / BATCH_SIZE * int(
            np.ceil(float(len(label_train)+len(label_val)) / BATCH_SIZE))))

        log_dir = MODEL_ROOT
        callbacks = [
            tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq=10),
            tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(log_dir, MODEL_SAVE),
                                               monitor='val_loss', save_best_only=True),
            tf.keras.callbacks.LearningRateScheduler(
                lambda epoch, lr: lr * 0.8 if ((epoch % epoch_to_decay == 0) and (epoch != 0)) else lr, verbose=True),
            tf.keras.callbacks.EarlyStopping(patience=20, monitor='val_loss'),
        ]

    try:
        # with strategy.scope():
        pillar_net.fit(training_gen,
                       validation_data=validation_gen,
                       steps_per_epoch=len(training_gen),
                       callbacks=callbacks,
                       use_multiprocessing=True,
                       epochs=int(params.total_training_epochs),
                       #    epochs=1,
                       workers=6)
        pillar_net.save(pb_MODEL)
        pillar_net.save(zoe_pointpillars)
        print('save========================================================================================')

    except KeyboardInterrupt:
        model_str = "interrupted_%s.h5" % time.strftime("%Y%m%d-%H%M%S")
        pillar_net.save(zoe_pointpillars)
        pillar_net.save(pb_MODEL)
        print('<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        print(model_str)


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - [%(levelname)s]: %(message)s")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

    train_PillarNet()
