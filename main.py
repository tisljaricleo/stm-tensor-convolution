from STM import SpeedTransitionMatrix
from misc import database, config
from misc.misc import plot_heatmap
import numpy as np
from scipy.spatial import distance
import math
import pandas as pd

import tensorly as ty
from tensorly.decomposition import non_negative_parafac


def get_mass_center(m):
    max_val = 0.2 * np.max(m)   # Filter: remove 10% of maximal value.
    m = np.where(m < max_val, 0, m)
    m = m / np.sum(m)
    # marginal distributions
    dx = np.sum(m, 1)
    dy = np.sum(m, 0)
    # expected values
    X, Y = m.shape
    cx = np.sum(dx * np.arange(X))
    cy = np.sum(dy * np.arange(Y))
    return int(cx), int(cy)


def diag_dist(point):
    # Max distance to the diagonal (square matrix m x m) is: diagonal_length / 2.
    max_d = (config.MAX_INDEX * math.sqrt(2)) / 2
    distan = []
    for d in config.DIAG_LOCS:
        distan.append(distance.euclidean(d, point))
    return round(min(distan) / max_d * 100, 2)  # Relative distance.


def get_link_ids_square(points, link_info):

    xb = link_info.x_b
    yb = link_info.y_b
    p1x = points[0][0]
    p1y = points[0][1]
    p2x = points[1][0]
    p2y = points[0][1]

    link_ids = link_info[(link_info.x_b > points[0][0])
                         & (link_info.y_b < points[0][1])
                         & (link_info.x_b < points[1][0])
                         & (link_info.y_b > points[1][1])]
    return link_ids.link_id.values


def main():

    config.initialize_paths()
    config.initialize_db_setup()
    config.initialize_stm_setup()

    db, client = database.init('SpeedTransitionDB')
    col_name = "spatialMatrixRWLNEWrel"

    '''

    p1 = (15.942489, 45.781501)  # upped left
    p2 = (15.961205, 45.774961)  # lower right

    info = pd.read_csv(r'links_info.csv', sep=';')

    links_inside = get_link_ids_square(points=(p1, p2), link_info=info)

    c = 0
    frontal_slices = []
    temp = []

    try:
        n_intervals = 8
        for interval in range(0, n_intervals):
            for link in links_inside:
                transitions = database.selectSome(db, col_name, {'origin_id': int(link)})
                for tran in transitions:
                    matrix = np.array(tran['intervals'][interval]['winter']['working'])
                    if int(np.sum(matrix)) > 20:
                        temp.append(list(matrix.flatten()))
                        c += 1
            # temp = np.array(temp).reshape((400, len(temp)))
            frontal_slices.append(temp)
            temp = []
    except:
        print('Warning: There are no transitions with oringin_id: %s' % link)

    slices_length = [len(slice) for slice in frontal_slices]
    n_trans = min(slices_length)

    tensor = np.zeros((400, n_trans, 8))

    for f_slice_id in range(0, len(frontal_slices)):
        for matrix_id in range(0, len(frontal_slices[f_slice_id])):
            if matrix_id >= n_trans:
                continue
            tensor[:, matrix_id, f_slice_id] = frontal_slices[f_slice_id][matrix_id]

    factors = non_negative_parafac(tensor=ty.tensor(tensor), rank=10, verbose=0)

    # xxx = factors.factors[0][:, 0].reshape(20, 20)
    # xxx = xxx / np.sum(xxx)
    # xxx = np.round(xxx, decimals=2)
    # plot_heatmap(xxx, 'ddd')
    #
    #
    # yyy = xxx.tolist()



    i = 0
    for column in range(0, factors.factors[0].shape[1]):
        xxx = factors.factors[0][:, column].reshape(20, 20)
        xxx = xxx / np.sum(xxx)
        xxx = np.round(xxx, decimals=2)
        plot_heatmap(xxx, 'ddd')
        #plot_heatmap(factors.factors[0][:, column].reshape(20, 20), 'Factor: ' + str(i))
        i += 1


    # char_matrices = list([])
    # chm = {'orig': xxx,
    #        'rounded': xxx,
    #        'xy_position': [i, j],
    #        'com_position': [cx, cy],
    #        'com_diag_dist': 0,
    #        'class': 0
    #        }



    '''

    links = [214697, 214696, 214695, 214694]

    stm = SpeedTransitionMatrix(db=db, client=client, collection_name=col_name)
    stm.get_consecutive_data(links=links)

    # m = np.array(stm.data[2]['intervals'][4]['winter']['working'])

    # x, y = get_mass_center(m)

    # dd = diag_dist(point=(x, y))

    # print()

    stm.plot_consecutive_data(dataset_type='winter', days_type='working', intervals='all', output='show')

    database.closeConnection(client)


if __name__ == "__main__":
    main()
