import numpy as np
import pandas as pd

# Anchor_file='/media/sdb1/zoe/FYP/folder_root/ZoeAnchor.csv'
Anchor_file='/media/sdb1/zoe/FYP/folder_root/AnchorKmeans.csv'

# Anchor_file=r'C:\Users\Chan Kin Yan\Documents\GitHub\FYP\folder_root\ZoeAnchor.csv'

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

OutPutVehecleClasees = {
    0: "Car",
    1: "taxi",
    2: "privateminibus",
    3: "publicminibus",
    4: "motorbike",
    5: "pedestrian",
    6: "construction-vehicle",
    7: "cargo-mpv",
    8: "mpv",
    9: "smalltruck",
    10: "cargo-one-box",
    11: "mediumtruck",
    12: "bigtruck",
    13: "flatbed-truck",
    14:  "coachbus",
    15: "dd",
}


class GridParameters:

    # zoe_pointpillars2
    # x_min = -50.4
    # x_max = 23.52
    # x_step = 0.28

    # y_min = -43.68
    # y_max = 70.56
    # y_step = 0.28

    # zoe_pointpillars3
    # x_min = -51.2
    # x_max = 29.44
    # x_step = 0.16

    # y_min = -39.68
    # y_max = 69.12
    # y_step = 0.16

    # zoe_pointpillars4
    x_min = -56.32
    x_max = 25.6
    x_step = 0.16

    y_min = -46.08
    y_max = 56.32
    y_step = 0.16

    z_min = 1.2
    z_max = 6.4

    # derived parameters
    Xn_f = float(x_max - x_min) / x_step
    Yn_f = float(y_max - y_min) / y_step
    Xn = int(Xn_f)
    Yn = int(Yn_f)

    def __init__(self):
        super(GridParameters, self).__init__()


class DataParameters:

    classes = VehicaleClasses
    nb_classes = len(np.unique(list(classes.values())))
    assert nb_classes == np.max(
        np.unique(list(classes.values()))) + 1, 'Starting class indexing at zero.'

    def __init__(self):
        super(DataParameters, self).__init__()


class NetworkParameters:

    max_points_per_pillar = 100
    # max_pillars = 10000
    max_pillars = 10000

    nb_features = 9
    nb_channels = 64
    downscaling_factor = 2

    # anchor_dims=np.round(np.array(pd.read_csv(Anchor_file,index_col=0).iloc[1:,:].values, dtype=np.float32).tolist(),3)
    anchor_dims=np.round(np.array(pd.read_csv(Anchor_file,index_col=0).values, dtype=np.float32).tolist(),3)
    nb_dims = 3

    # positive_iou_threshold = 0.5
    # negative_iou_threshold = 0.35
    positive_iou_threshold = 0.7
    negative_iou_threshold = 0.35

    batch_size = 4
    total_training_epochs = 80
    # total_training_epochs = 20

    # 101040.    # 15 * 4 * ceil(6733. / 4) --> every 15 epochs on 6733 kitti samples, cf. pillar paper
    iters_to_decay = 54540
    learning_rate = 2e-4
    decay_rate = 1e-8
    L1 = 0
    L2 = 0
    alpha = 0.25
    gamma = 2.0
    # original pillars paper values
    focal_weight = 3.0      # 1.0
    loc_weight = 2.0        # 2.0
    size_weight = 2.0       # 2.0
    angle_weight = 2.0      # 2.0
    heading_weight = 0.2    # 0.2
    class_weight = 0.5      # 0.2

    def __init__(self):
        super(NetworkParameters, self).__init__()


class Parameters(GridParameters, DataParameters, NetworkParameters):

    def __init__(self):
        super(Parameters, self).__init__()
