import numpy as np

VehicaleClasses = {
        "bigtruck": 0,
        "black-smalltruck": 0,
        "crane-truck": 0,
        "cylindrical-truck": 0,
        "flatbed-truck": 0,
        "mediumtruck": 0,
        "smalltruck": 0,

        "black-cargo-mpv":1,
        "black-mpv":1,
        "cargo-mpv":1,
        "mpv":1,

        "privateminibus": 2,
        "publicminibus": 2,

        "pedestrian": 3,
        "taxi": 4,
        "motorbike": 5,
        "coachbus": 6,
        "construction-vehicle": 7,

        # "black-cargo-one-box": 2,
        # "black-one-box": 4,
        # "black-three-box": 6,
        # "black-two-box": 7,
        # "cargo-one-box": 9,
        # "dd": 14,
        # "one-box": 19,
        # "three-box": 25,
        # "two-box": 26,

    }

class GridParameters:
    x_min = 0.0
    x_max = 80.64
    x_step = 0.16

    y_min = -40.32
    y_max = 40.32
    y_step = 0.16

    z_min = -1.0
    z_max = 3.0

    # derived parameters
    Xn_f = float(x_max - x_min) / x_step
    Yn_f = float(y_max - y_min) / y_step
    Xn = int(Xn_f)
    Yn = int(Yn_f)

    def __init__(self):
        super(GridParameters, self).__init__()


class DataParameters:

    # classes = {"Car":               0,
    #            "Pedestrian":        1,
    #            "Person_sitting":    1,
    #            "Cyclist":           2,
    #            "Truck":             3,
    #            "Van":               3,
    #            "Tram":              3,
    #            "Misc":              3,
    #            }

    # classes = {
    #     "bigtruck": 0,
    #     "black-cargo-mpv": 1,
    #     "black-cargo-one-box": 2,
    #     "black-mpv": 3,
    #     "black-one-box": 4,
    #     "black-smalltruck": 5,
    #     "black-three-box": 6,
    #     "black-two-box": 7,
    #     "cargo-mpv": 8,
    #     "cargo-one-box": 9,
    #     "coachbus": 10,
    #     "construction-vehicle": 11,
    #     "crane-truck": 12,
    #     "cylindrical-truck": 13,
    #     "dd": 14,
    #     "flatbed-truck": 15,
    #     "mediumtruck": 16,
    #     "motorbike": 17,
    #     "mpv": 18,
    #     "one-box": 19,
    #     "pedestrian": 20,
    #     "privateminibus": 21,
    #     "publicminibus": 22,
    #     "smalltruck": 23,
    #     "taxi": 24,
    #     "three-box": 25,
    #     "two-box": 26,

    # }
    classes = {
        "bigtruck": 0,
        "black-smalltruck": 0,
        "crane-truck": 0,
        "cylindrical-truck": 0,
        "flatbed-truck": 0,
        "mediumtruck": 0,
        "smalltruck": 0,

        "black-cargo-mpv":1,
        "black-mpv":1,
        "cargo-mpv":1,
        "mpv":1,

        "privateminibus": 2,
        "publicminibus": 2,

        "pedestrian": 3,
        "taxi": 4,
        "motorbike": 5,
        "coachbus": 6,
        "construction-vehicle": 7,

        # "black-cargo-one-box": 2,
        # "black-one-box": 4,
        # "black-three-box": 6,
        # "black-two-box": 7,
        # "cargo-one-box": 9,
        # "dd": 14,
        # "one-box": 19,
        # "three-box": 25,
        # "two-box": 26,

    }

    nb_classes = len(np.unique(list(classes.values())))
    assert nb_classes == np.max(
        np.unique(list(classes.values()))) + 1, 'Starting class indexing at zero.'

    def __init__(self):
        super(DataParameters, self).__init__()


class NetworkParameters:

    max_points_per_pillar = 100
    max_pillars = 12000
    nb_features = 7
    nb_channels = 64
    downscaling_factor = 2

    # length, width, height, z-center, orientation
    anchor_dims = np.array([[3.9, 1.6, 1.56, -1, 0],
                            [3.9, 1.6, 1.56, -1, 1.5708],
                            [0.8, 0.6, 1.73, -0.6, 0],
                            [0.8, 0.6, 1.73, -0.6, 1.5708],
                            ], dtype=np.float32).tolist()
    nb_dims = 3

    positive_iou_threshold = 0.6
    negative_iou_threshold = 0.3
    batch_size = 4
    # total_training_epochs = 160
    total_training_epochs = 20

    # 15 * 4 * ceil(6733. / 4) --> every 15 epochs on 6733 kitti samples, cf. pillar paper
    iters_to_decay = 101040.
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
