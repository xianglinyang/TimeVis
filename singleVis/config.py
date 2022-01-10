"""
Training config for different datasets.
"""

dataset_config = {
    "cifar10": {
        "TRAINING_LEN": 50000,
        "TESTING_LEN": 10000,
        "LAMBDA":10.,
        "DOWNSAMPLING_RATE": .1,
        "L_BOUND":0.6,
        "MAX_HAUSDORFF":0.2,
        "ALPHA":0,
        "BETA":.1,
        "INIT_NUM":300,
        "training_config":{
            "EPOCH_NUM": 6,
            "TIME_STEPS": 11,
            "TEMPORAL_PERSISTENT": 1,
            "NUMS": 5,   # the number of epochs to go through in one go
            "PATIENT": 3,    # early stopping patient
            "TEMPORAL_EDGE_WEIGHT":100,
        }
        
    },
    "online": {
        "TRAINING_LEN": 50000,
        "TESTING_LEN": 10000,
        "LAMBDA":10.,
        "DOWNSAMPLING_RATE": .1,
        "L_BOUND":0.6,
        "MAX_HAUSDORFF":0.2,
        "ALPHA":0,
        "BETA":1,
        "INIT_NUM":300,
        "training_config":{
            "EPOCH_NUM": 6,
            "TIME_STEPS": 5,
            "TEMPORAL_PERSISTENT": 1,
            "NUMS": 5,   # the number of epochs to go through in one go
            "PATIENT": 3,    # early stopping patient
            "TEMPORAL_EDGE_WEIGHT":100,
        }
        
    },
    "mnist": {
        "TRAINING_LEN": 60000,
        "TESTING_LEN": 10000,
        "LAMBDA":1.,
        "DOWNSAMPLING_RATE": .1,
        "L_BOUND":0.4,
        "MAX_HAUSDORFF":0.25,
        "ALPHA":1,
        "BETA":1,
        "INIT_NUM":500,
        "training_config":{
            "EPOCH_NUM": 6,
            "TIME_STEPS": 10,
            "TEMPORAL_PERSISTENT": 1,
            "NUMS": 5,   # the number of epochs to go through in one go
            "PATIENT": 3,    # early stopping patient
            "TEMPORAL_EDGE_WEIGHT":100,
        }
    },
    "fmnist": {
        "TRAINING_LEN": 60000,
        "TESTING_LEN": 10000,
        "LAMBDA":20.,
        "DOWNSAMPLING_RATE": .1,
        "L_BOUND":0.5,
        "MAX_HAUSDORFF":.06,
        "ALPHA":2,
        "BETA":1.3,
        "INIT_NUM":300,
        "training_config":{
            "EPOCH_NUM": 6,
            "TIME_STEPS": 10,
            "TEMPORAL_PERSISTENT": 1,
            "NUMS": 5,   # the number of epochs to go through in one go
            "PATIENT": 3,    # early stopping patient
            "TEMPORAL_EDGE_WEIGHT":100,
        }
    }
}