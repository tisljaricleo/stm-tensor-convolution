from misc import database, config
from misc.misc import plot_heatmap
import numpy as np


class SpeedTransitionMatrix(object):
    """ Class for generating the speed transition matrices (STM).

    This class is used for:
    (1) importing spatial matrix generated via pre-processing step
    (2) getting the STM data from consecutive links
    (3) ploting or saving the STM data

    Attributes
    ----------
    No atributes needed for init. Path to spatial matrix must be defined in the config.py file.

    Methods
    -------
    get_consecutive_data(self, links)
        Gets STM from consecutive links and saves to self.data object.
    plot_consecutive_data(self, dataset_type='all', output='show', intervals='all')
        Plots STM matrices from self.data object.
    """

    def __init__(self, db, client, collection_name):
        config.initialize_paths()   # Init of the path variables in config.py file.
        config.initialize_stm_setup()   # Init of the speed transition matrix related variables in config.py file.
        self.db = db
        self.client = client
        self.col_name = collection_name
        # self.spatial_matrix = open_pickle(config.SPATIAL_MATRIX_PKL_NAME)
        self.data = list([])    #

    def get_consecutive_data(self, links):
        """Gets STM from consecutive links and saves to self.data object.

        :param links:
        :return:
        """

        # db.spatialMatrixabs.find().count()

        self.data = list([])
        for i in range(0, len(links) - 1):
            origin_id = links[i]
            destination_id = links[i + 1]
            self.data.append(database.selectSome(db=self.db,
                                                 collection=self.col_name,
                                                 query={'origin_id': origin_id, 'destination_id': destination_id})[0])

            # self.data.append({'origin_id': origin_id,
            #                   'destination_id': destination_id,
            #                   'matrix': self.spatial_matrix[destination_id][origin_id]})

    # TODO: def save_consecutive data as numpy arrays (ex. output='numpy')
    def plot_consecutive_data(self, dataset_type='all', days_type='all', output='show', intervals='all'):

        if len(self.data) < 1:
            print('Empty data, nothing to plot!')
            return

        arg_type = str(type(intervals))

        if 'int' in arg_type:
            if 0 <= intervals <= 7:
                for d in self.data:
                    interval = d['intervals'][intervals]
                    title = str(d['origin_id']) + ' to ' + str(d['destination_id'])
                    title = title + ' interval=' + str(intervals) + ' dataset=' + dataset_type + ' days=' + days_type
                    plot_heatmap(data=np.array(interval[dataset_type][days_type]).astype('int'),
                                 title=title,
                                 output=output)
            else:
                print('Arg intervals must be in range 0 - 7!')

        if 'str' in arg_type:
            if 'all' in intervals:
                for d in self.data:
                    i_id = 0
                    for interval in d['intervals']:
                        title = str(d['origin_id']) + ' to ' + str(d['destination_id'])
                        title = title + ' interval=' + str(i_id) + ' dataset=' + dataset_type + ' days=' + days_type
                        plot_heatmap(data=np.array(interval[dataset_type][days_type]).astype('int'),
                                     title=title,
                                     output=output)
                        i_id += 1
            else:
                print('Arg intervals must be in range 0 - 7 or \'all\'!')




