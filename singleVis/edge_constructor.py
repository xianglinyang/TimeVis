import numpy as np


class EdgesConstructor(object):

    def __init__(self, data_provider, init_num, n_epochs, spatial_constructor, temporal_constructor) -> None:
        """
        This class is the Base class for complex edge constructor

        Parameters
        ----------
        data_provider : Object
            the data provider for feature vectors
        init_num: int
            the init number of samples in each time step
        TIME_STEPS: int
            the number of time steps of subject models
        n_epochs: int
            the number of epoch that we iterate each round
        spatial: str
            the strategy name used for spatial edge constructor
        temporal: str
            the strategy name used for temporal edge constructor
        """
        self.data_provider = data_provider
        self.init_num = init_num
        self.n_epochs = n_epochs
        self.spatial_constructor = spatial_constructor
        self.temporal_constructor = temporal_constructor
    
    def construct(self):
        return NotImplemented



