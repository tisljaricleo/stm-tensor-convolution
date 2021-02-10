from STM import SpeedTransitionMatrix
from misc import database, config
from misc.misc import plot_heatmap, save_pickle_data, get_time
import numpy as np
from scipy.spatial import distance
import math
import pandas as pd

import tensorly as ty
from tensorly.decomposition import non_negative_parafac


def create_coordinate_matrix(sp, xn, yn, lons, lats):
    """
    Creates xn times yn matrix of GNSS points.
    :param sp: Starting GNSS point.
    :param xn: Number of rectangles (columns).
    :param yn: Number of rectangles (rows).
    :param lons: Longitude step.
    :param lats: Latitude step.
    :return: Matrix of GNSS points for rectangle drawing. Every cell consists of a tuple with four points (lon1, lat1, lon2, lat2, lon3, lat3, lon4, lat4)
    """
    coordinate_matrix = []
    column_values = []
    for ii in range(1, yn + 1):
        for jj in range(1, xn + 1):
            lon1 = sp[0] + ((jj - 1) * lons)
            lat1 = sp[1] - ((ii - 1) * lats)
            lon2 = sp[0] + (jj * lons)
            lat2 = sp[1] - (ii * lats)
            lon3 = lon1 + lons
            lat3 = lat1
            lon4 = lon2 - lons
            lat4 = lat2
            column_values.append((lon1, lat1, lon2, lat2, lon3, lat3, lon4, lat4))
        coordinate_matrix.append(column_values)
        column_values = []
    return coordinate_matrix


def get_mass_center(m):
    max_val = 0.2 * np.max(m)   # Filter: remove 20% of maximal value.
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
    try:
        link_ids = link_info[(link_info.x_b > points[0][0])
                             & (link_info.y_b < points[0][1])
                             & (link_info.x_b < points[1][0])
                             & (link_info.y_b > points[1][1])]
        if len(link_ids.link_id.values) == 0:
            return None
        return link_ids.link_id.values
    except:
        return None


print('Script {0} started ... '.format(__file__))
t_start = get_time()

config.initialize_paths()
config.initialize_db_setup()
config.initialize_stm_setup()

db, client = database.init('SpeedTransitionDB')
#col_name = "spatialMatrixRWLNEWrel"

col_name = config.SM_COLLECTION+'rel'
tensor_col_name = config.TENSOR_COLLECTION

tensor_rank = 10

spatial_square = dict({})
total_data = list([])


lon_step = 0.006545  # ~500[m]
lat_step = 0.004579  # ~500[m]
x_num = 50   # Number of rectangles (columns).
y_num = 20   # Number of rectangles (rows).
lon_start = 15.830326
lat_start = 45.827299
start_point = (lon_start, lat_start)

coordinate_matrix = create_coordinate_matrix(sp=start_point,
                                             xn=x_num,
                                             yn=y_num,
                                             lons=lon_step,
                                             lats=lat_step)

info = pd.read_csv(r'links_info.csv', sep=';')

none_counter = 0
total_counter = 0
for i in range(0, len(coordinate_matrix)):
    for j in range(0, len(coordinate_matrix[0])):

        print("i=%d\t\tj=%d" % (i, j))

        total_counter += 1

        p1 = (coordinate_matrix[i][j][0], coordinate_matrix[i][j][1])
        p2 = (coordinate_matrix[i][j][2], coordinate_matrix[i][j][3])

        links_inside = get_link_ids_square(points=(p1, p2), link_info=info)


        if links_inside is not None:
            c = 0
            frontal_slices = []
            valid_transitions = []
            temp = []
            temp_tran = []

            try:
                n_intervals = 8
                for interval in range(0, n_intervals):
                    for link in links_inside:
                        # transitions = database.selectSome(db, col_name, {'$or': [{'origin_id': int(link)}, {'destination_id': int(link)}]})
                        transitions = database.selectSome(db, col_name, {'origin_id': int(link)})
                        for tran in transitions:
                            matrix = np.array(tran['intervals'][interval]['winter']['working'])
                            if int(np.sum(matrix)) > 20:
                                temp.append(list(matrix.flatten()))
                                c += 1
                                # temp_tran.append((tran['origin_id'], tran['destination_id']))
                                valid_transitions.append((tran['origin_id'], tran['destination_id']))

                    # temp = np.array(temp).reshape((400, len(temp)))
                    frontal_slices.append(temp)
                    # valid_transitions.append(temp_tran)
                    temp = []
                    temp_tran = []
            except:
                print('Warning: There are no transitions with oringin_id: %s' % link)


            slices_length = [len(slice) for slice in frontal_slices]
            # print(slices_length)
            n_trans = min(slices_length)

            if n_trans == 0:
                continue

            # print()
            # valid_transitions = [x[0:n_trans] for x in valid_transitions]
            valid_transitions = valid_transitions[0:n_trans]


            tensor = np.zeros((400, n_trans, 8))

            for f_slice_id in range(0, len(frontal_slices)):
                for matrix_id in range(0, len(frontal_slices[f_slice_id])):
                    if matrix_id >= n_trans:
                        continue
                    tensor[:, matrix_id, f_slice_id] = frontal_slices[f_slice_id][matrix_id]

            factors = non_negative_parafac(tensor=ty.tensor(tensor), rank=tensor_rank, verbose=0)

            spatial_square = dict({})
            spatial_square['p1'] = p1
            spatial_square['p2'] = p2
            spatial_square['tensor'] = tensor
            spatial_square['links_inside'] = links_inside
            spatial_square['valid_transitions'] = valid_transitions
            spatial_square['xy_position'] = [i, j]
            spatial_square['char_matrices'] = list([])
            spatial_square['spatial_matrix'] = factors.factors[1].tolist()
            spatial_square['temporal_matrix'] = factors.factors[2].tolist()

            factor_index = 0
            for column in range(0, factors.factors[0].shape[1]):
                orig = factors.factors[0][:, column].reshape(20, 20)
                rounded = orig / np.sum(orig)
                rounded = np.round(rounded, decimals=2)
                cx, cy = get_mass_center(orig)
                dist = diag_dist(point=(cx, cy))

                anomaly = False
                if dist >= 46:
                    anomaly = True

                chm = {'orig': orig.tolist(),
                       'rounded': rounded.tolist(),
                       'com_position': [cx, cy],
                       'com_diag_dist': dist,
                       'factor_id': factor_index,
                       'anomaly': anomaly,
                       'class': 0
                       }

                if anomaly:
                    sm = factors.factors[1][:, factor_index].tolist()
                    spatial_max_id = sm.index(max(sm))

                    tm = factors.factors[2][:, factor_index].tolist()
                    temporal_max_id = tm.index(max(tm))

                    chm['max_spatial_id'] = spatial_max_id
                    chm['spatial_anomaly_char'] = sm
                    chm['anomalous_trans'] = valid_transitions[spatial_max_id]
                    chm['max_temporal_id'] = temporal_max_id
                    chm['temporal_anomaly_char'] = tm


                spatial_square['char_matrices'].append(chm)
                factor_index += 1

            total_data.append(spatial_square)

        else:
            none_counter += 1
            spatial_square = dict({})

            spatial_square['p1'] = p1
            spatial_square['p2'] = p2
            spatial_square['tensor'] = None
            spatial_square['links_inside'] = None
            spatial_square['xy_position'] = [i, j]
            spatial_square['char_matrices'] = None

            total_data.append(spatial_square)

        # TODO: spatial_square insert into database

t1 = get_time()
save_pickle_data('spatialTensors5.pkl', total_data)
t2 = get_time()
print('Pickle save time: {0}'.format(t2 - t1))

t_end = get_time()
print('Exe time: {0}'.format(t_end - t_start))

