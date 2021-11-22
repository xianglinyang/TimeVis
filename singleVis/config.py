dataset_config = {
    "cifar10": {
        "TRAINING_LEN": 50000,
        "TESTING_LEN": 10000,
    },
    "mnist": {
        "TRAINING_LEN": 60000,
        "TESTING_LEN": 10000,
    },
    "fmnist": {
        "TRAINING_LEN": 60000,
        "TESTING_LEN": 10000,
    }
}
training_config = {
    "EPOCH_NUM": 100,
    "TIME_STEPS": 7,
    "TEMPORAL_PERSISTENT": 2,
    "NUMS": 5,   # the number of epochs to go through in one go
    "PATIENT": 3,    # early stopping patient
    "DOWNSAMPLING_RATE": .1,
}