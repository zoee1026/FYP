import numpy as np

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


class GridParameters:
    x_min = -50.4
    x_max = 23.52
    x_step = 0.28

    y_min = -43.68
    y_max = 70.56
    y_step = 0.28

    z_min = 0.6
    z_max = 12.0

    # derived parameters
    Xn_f = float(x_max - x_min) / x_step
    Yn_f = float(y_max - y_min) / y_step
    Xn = int(Xn_f)
    Yn = int(Yn_f)

    def __init__(self):
        super(GridParameters, self).__init__()


class DataParameters:

    classes = {

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

    nb_classes = len(np.unique(list(classes.values())))
    assert nb_classes == np.max(
        np.unique(list(classes.values()))) + 1, 'Starting class indexing at zero.'

    def __init__(self):
        super(DataParameters, self).__init__()


class NetworkParameters:

    max_points_per_pillar = 100
    max_pillars = 8000
    nb_features = 9
    nb_channels = 128
    downscaling_factor = 2

    # length, width, height, z-center, orientation
    anchor_dims = np.array([
        [2.24, 1.12, 1.96, 2.52, 0],
        [5.6, 2.52, 2.24, 2.52, 0],
        [8.12, 2.8, 3.36, 3.08, 0],
        [10.08, 2.8, 3.36, 3.08, 0],
        [11.76, 3.36, 4.76, 3.36, 0],
        [13.44, 3.08, 3.92, 3.36, 0],
        [2.24, 1.12, 1.96, 2.52, 1.5708],
        [5.6, 2.52, 2.24, 2.52, 1.5708],
        [8.12, 2.8, 3.36, 3.08, 1.5708],
        [10.08, 2.8, 3.36, 3.08, 1.5708],
        [11.76, 3.36, 4.76, 3.36, 1.5708],
        [13.44, 3.08, 3.92, 3.36, 1.5708],
    ], dtype=np.float32).tolist()

    nb_dims = 3

    positive_iou_threshold = 0.6
    negative_iou_threshold = 0.3
    batch_size = 4
    total_training_epochs = 10
    # total_training_epochs = 20

    # 101040.    # 15 * 4 * ceil(6733. / 4) --> every 15 epochs on 6733 kitti samples, cf. pillar paper
    iters_to_decay = 95760
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
    angle_weight = 1.0      # 2.0
    heading_weight = 0.2    # 0.2
    class_weight = 0.5      # 0.2

    def __init__(self):
        super(NetworkParameters, self).__init__()


class Parameters(GridParameters, DataParameters, NetworkParameters):

    def __init__(self):
        super(Parameters, self).__init__()
