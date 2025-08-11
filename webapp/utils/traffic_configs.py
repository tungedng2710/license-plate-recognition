import numpy as np

VIDEO_SOURCE = "/media/tungn197/hdd1/traffic_data/HPG_ngatu363_2.mp4"
REGIONS = {
    "road1": {
        "vertices": np.array([[1386, 167], [889, 843], [2078, 843], [1535, 167]], np.int32), # clockwise
        "color": (0, 255, 255),
        "direction": "s2n", # south to north
        "light": [1401, 43, 1447, 178],    # x1y1x2y2
        "stop_line": None
    },
    "road1_turn_left": {
        "vertices": np.array([[1145, 311], [481, 843], [875, 843], [1300, 267]], np.int32),
        "color": (255, 255, 0),
        "direction": "n2sw", # north to south west
        "light": [504, 45, 549, 168], # x1y1x2y2
        "stop_line": np.array([[777, 984], [768, 996], [2213, 996], [2205, 984]])
    },
    # "road2": {
    #     "vertices": np.array([[543, 334], [5, 524], [677, 524], [967, 334]], np.int32),
    #     "direction": "n2s", # north to south
    #     "light": None, # x1y1x2y2
    #     "stop_line": np.array([[777, 984], [768, 996], [2213, 996], [2205, 984]])
    # },
}
