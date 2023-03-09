import os
import sys
import time
import logging
import argparse
import numpy as np
import tensorflow as tf
from easydict import EasyDict as edict

from config import Parameters
from loss import PointPillarNetworkLoss
from network import build_point_pillar_graph
from processors import SimpleDataGenerator
from readers import KittiDataReader

from read_file_location import GetMatchedDatafile, TestModel

tf.get_logger().setLevel("ERROR")

# DATA_ROOT = '../MatchFile.csv'
DATA_ROOT = '../TestFile.csv'

MODEL_ROOT = "./log"
MODEL_PATH = "model.h5"
MODEL_SAVE = "train.h5"


def generate_config_from_cmd_args():
    parser = argparse.ArgumentParser(description='PointPillars training')
    parser.add_argument('--gpu_idx', default=3, type=int, required=False, 
        help='GPU index to use for inference')
    parser.add_argument('--model_root', default='./log/', required=True,
        help='Path for dumping training logs and model weights')
    
    configs = edict(vars(parser.parse_args()))
    return configs


def train_PillarNet(configs):

    params = Parameters()

    pillar_net = build_point_pillar_graph(params)
    pillar_net.load_weights(os.path.join(MODEL_ROOT, MODEL_PATH))

    if os.path.exists(os.path.join(configs.model_root, "model.h5")):
        logging.info("Using pre-trained weights found at path: {}".format(configs.model_root))
        pillar_net.load_weights('zoe_pointpillars.h')
    else:
        logging.info("No pre-trained weights found. Initializing weights and training model.")

    loss = PointPillarNetworkLoss(params)

    optimizer = tf.keras.optimizers.Adam(learning_rate=params.learning_rate)

    pillar_net.compile(optimizer, loss=loss.losses())

    data_reader = KittiDataReader()

    lidar_train, label_train,lidar_val, label_val = GetMatchedDatafile(DATA_ROOT)
    print(len(lidar_train),len(label_train),'---------------------------------------------------------------------')
    print(len(lidar_val),len(label_val),'---------------------------------------------------------------------')


    # "Input dirs require equal number of files."
    assert len(lidar_train) == len(label_train)
    assert len(lidar_val) == len(label_val)

  
    training_gen = SimpleDataGenerator(data_reader, params.batch_size, lidar_train, label_train)
    validation_gen = SimpleDataGenerator(data_reader, params.batch_size, lidar_val, label_val)


    log_dir = MODEL_ROOT
    epoch_to_decay = int( np.round(params.iters_to_decay / params.batch_size * int(np.ceil(float(len(label_train)+len(label_val)) / params.batch_size))))
       
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq=10),
        tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(log_dir,MODEL_SAVE ),
                                           monitor='val_loss', save_best_only=True),
        tf.keras.callbacks.LearningRateScheduler(
            lambda epoch, lr: lr * 0.8 if ((epoch % epoch_to_decay == 0) and (epoch != 0)) else lr, verbose=True),
        tf.keras.callbacks.EarlyStopping(patience=20, monitor='val_loss'),
    ]

    try:
        pillar_net.fit(training_gen,
                       validation_data = validation_gen,
                       steps_per_epoch=len(training_gen),
                       callbacks=callbacks,
                       use_multiprocessing=True,
                    #    epochs=int(params.total_training_epochs),
                       epochs=1,
                       workers=6)
        pillar_net.save('my_model')
        pillar_net.save('zoe_pointpillars.h')

    except KeyboardInterrupt:
        model_str = "interrupted_%s.h5" % time.strftime("%Y%m%d-%H%M%S")
        pillar_net.save(os.path.join(log_dir, model_str))
        print("Interrupt. Saving output to %s" % os.path.join(os.getcwd(), log_dir[1:], model_str))
        print('<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        print(model_str)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s]: %(message)s")
    train_configs = generate_config_from_cmd_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(train_configs.gpu_idx)
    # tf.get_logger().setLevel("ERROR")

    logging.info("Model training logs and trained weights will be saved at path: {}".format(train_configs.model_root))
    train_PillarNet(train_configs)