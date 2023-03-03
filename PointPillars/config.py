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
    # "motorbike": 4,
    # "pedestrian": 5,

    # "construction-vehicle": 6,
    # "crane-truck": 6,
    # "cylindrical-truck": 6,


    # "black-cargo-mpv": 7,
    # "cargo-mpv": 7,

    # "black-mpv": 8,
    # "mpv": 8,


    # "smalltruck": 9,
    # "black-smalltruck": 9,

    # "black-cargo-one-box": 10,
    # "cargo-one-box": 10,

    # "mediumtruck": 11,
    # "bigtruck": 12,
    # "flatbed-truck": 13,
    # "coachbus": 14,
    # "dd": 15,
}


class GridParameters:
    x_min = -50.4
    x_max = 23.52
    x_step = 0.28

    y_min = -77.28
    y_max = 70.65
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
        # "motorbike": 4,
        # "pedestrian": 5,

        # "construction-vehicle": 6,
        # "crane-truck": 6,
        # "cylindrical-truck": 6,


        # "black-cargo-mpv": 7,
        # "cargo-mpv": 7,

        # "black-mpv": 8,
        # "mpv": 8,


        # "smalltruck": 9,
        # "black-smalltruck": 9,

        # "black-cargo-one-box": 10,
        # "cargo-one-box": 10,

        # "mediumtruck": 11,
        # "bigtruck": 12,
        # "flatbed-truck": 13,
        # "coachbus": 14,
        # "dd": 15,
    }
    # classes = {
    #     "black-one-box": 0,
    #     "black-three-box": 0,
    #     "black-two-box": 0,
    #     "one-box": 0,
    #     "three-box": 0,
    #     "two-box": 0,
    #     "taxi": 0,

    #     "black-cargo-mpv":0,
    #     "black-mpv":0,
    #     "cargo-mpv":0,
    #     "mpv":0,

    #     "pedestrian": 1,

    #     "bigtruck": 2,
    #     "black-smalltruck": 2,
    #     "crane-truck": 2,
    #     "cylindrical-truck": 2,
    #     "flatbed-truck": 2,
    #     "mediumtruck": 2,
    #     "smalltruck": 2,

    #     "privateminibus": 3,
    #     "publicminibus": 3,

    #     # "motorbike": 5,
    #     # "coachbus": 6,
    #     # "construction-vehicle": 7,

    #     # "black-cargo-one-box": 2,

    #     # "cargo-one-box": 9,
    #     # "dd": 14,

    # }

    nb_classes = len(np.unique(list(classes.values())))
    assert nb_classes == np.max(
        np.unique(list(classes.values()))) + 1, 'Starting class indexing at zero.'

    def __init__(self):
        super(DataParameters, self).__init__()


class NetworkParameters:

    max_points_per_pillar = 100
    max_pillars = 12000
    nb_features = 9
    nb_channels = 64
    downscaling_factor = 2

    # length, width, height, z-center, orientation
    anchor_dims = np.array([[3.9, 1.6, 1.56, -1, 0],  # car anchor
                            # car-anchor rotated by 90 degrees
                            [3.9, 1.6, 1.56, -1, 1.5708],
                            [0.8, 0.6, 1.73, -0.6, 0],  # pedestrian-anchor
                            # pedestrian-anchor rotated by 90 degrees
                            [0.8, 0.6, 1.73, -0.6, 1.5708],
                            # cyclist-anchor rotated by 90 degrees
                            [1.76, 0.6, 1.73, -0.6, 0],
                            # cyclist-anchor rotated by 90 degrees
                            [1.76, 0.6, 1.73, -0.6, 1.5708],
                            ], dtype=np.float32).tolist()
    nb_dims = 3

    positive_iou_threshold = 0.6
    negative_iou_threshold = 0.3
    batch_size = 4
    total_training_epochs = 160
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
