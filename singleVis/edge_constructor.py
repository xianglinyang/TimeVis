from msilib.schema import Error
import singleVis.spatial_edge_constructor as spat
import singleVis.temporal_edge_constructor as temp


class EdgesConstructor(object):

    def __init__(self, data_provider, init_num, n_epochs, spatial="kc", temporal="global") -> None:
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
        # initialize spatial and temporal edge constructor
        if spatial == "kc":
            self.spatial_cons = spat.kcSpatialEdgeConstructor()
        elif spatial == "random":
            self.spatial_cons = spat.RandomSpatialEdgeConstructor()
        else:
            raise NotImplemented("Strategy {} not implemented!".format(spatial))

        if temporal == "local":
            self.temporal_cons = temp.LocalTemporalEdgeConstructor()
        elif temporal == "global":
            self.temporal_cons = temp.GlobalTemporalEdgeConstructor()
        else:
            raise NotImplemented("Strategy {} not implemented!".format(temporal))

    
    def construct(self):
        return NotImplemented



