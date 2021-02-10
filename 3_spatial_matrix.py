"""Script generates spatial matrix as pandas DataFrame.
"""

import numpy as np
from misc.misc import rtm, get_time
from misc import database, config
import pandas as pd

__author__ = "Leo Tisljaric"
__license__ = "GPL"
__version__ = "0.0.1"
__email__ = "ltisljaric@fpz.hr"
__status__ = "Development"


def get_speed_limit(link_id):
    #speed_limit = speed_data[speed_data.link_id == link_id].speed_limit.values
    try:
        speed_limit = speed_data[speed_data.link_id == str(link_id)].speed_limit.values[0]
        #print()
        # if speed_limit == 0:  # ako je speed limit nepoznat
        #     speed_limit = 60
    except:
        speed_limit = 0  # ako nema zapisa u csv datoteci
    return int(speed_limit)



def generate_spatial_matrix():
    speed_type = config.SPEED_TYPE
    intervals = list(range(0, 8, 1))

    # All unique transitions by origin and destination id
    unique_vals = list(database.groupBy(db=db,
                                        collection=config.TRANSITION_COLLECTION,
                                        query={'_id': {'origin_id': '$origin_id',
                                                       'destination_id': '$destination_id'}}))

    # List of tuples (origin_id, destination_id)
    ids = list([])
    for uv in unique_vals:
        sl_origin = get_speed_limit(uv['_id']['origin_id'])
        sl_dest = get_speed_limit(uv['_id']['destination_id'])

        if (config.SL_DOWN <= sl_origin <= config.SL_UP) and (config.SL_DOWN <= sl_dest <= config.SL_UP):
            ids.append((uv['_id']['origin_id'], uv['_id']['destination_id']))

        # if (sl_origin >= config.SL_DOWN and sl_dest >= config.SL_DOWN) and \
        #         (sl_origin <= config.SL_UP and sl_dest <= config.SL_UP):
        #     ids.append((uv['_id']['origin_id'], uv['_id']['destination_id']))

    br = 0
    for tr_id in range(0, len(ids)):

        br += 1
        if br % 1000 == 0:
            print(br)

        origin = ids[tr_id][0]
        destination = ids[tr_id][1]

        transition = (database.selectSome(db=db,
                                          collection=config.TRANSITION_COLLECTION,
                                          query={'origin_id': origin,
                                                 'destination_id': destination}))
        interval_dict = list([])

        winter_origin_speeds_all = list([])
        winter_dest_speeds_all = list([])
        winter_origin_speeds_work = list([])
        winter_destination_speeds_work = list([])
        winter_origin_speeds_weekend = list([])
        winter_destination_speeds_weekend = list([])

        summer_origin_speeds_all = list([])
        summer_dest_speeds_all = list([])
        summer_origin_speeds_work = list([])
        summer_destination_speeds_work = list([])
        summer_origin_speeds_weekend = list([])
        summer_destination_speeds_weekend = list([])

        all_origin_speeds_all = list([])
        all_dest_speeds_all = list([])
        all_origin_speeds_work = list([])
        all_destination_speeds_work = list([])
        all_origin_speeds_weekend = list([])
        all_destination_speeds_weekend = list([])

        for i in intervals:

            for t in transition:

                orig_speed = rtm(t['origin_' + speed_type + '_speed'], config.RESOLUTION, speed_type)
                dest_speed = rtm(t['destination_' + speed_type + '_speed'], config.RESOLUTION, speed_type)

                if t['interval'] == i and t['summer'] == 0:
                    winter_origin_speeds_all.append(orig_speed)
                    winter_dest_speeds_all.append(dest_speed)

                    if t['working_day'] == 1:
                        winter_origin_speeds_work.append(orig_speed)
                        winter_destination_speeds_work.append(dest_speed)
                    else:
                        winter_origin_speeds_weekend.append(orig_speed)
                        winter_destination_speeds_weekend.append(dest_speed)

                if t['interval'] == i and t['summer'] == 1:
                    summer_origin_speeds_all.append(orig_speed)
                    summer_dest_speeds_all.append(dest_speed)

                    if t['working_day'] == 1:
                        summer_origin_speeds_work.append(orig_speed)
                        summer_destination_speeds_work.append(dest_speed)
                    else:
                        summer_origin_speeds_weekend.append(orig_speed)
                        summer_destination_speeds_weekend.append(dest_speed)

                if t['interval'] == i:
                    all_origin_speeds_all.append(orig_speed)
                    all_dest_speeds_all.append(dest_speed)

                    if t['working_day'] == 1:
                        all_origin_speeds_work.append(orig_speed)
                        all_destination_speeds_work.append(dest_speed)
                    else:
                        all_origin_speeds_weekend.append(orig_speed)
                        all_destination_speeds_weekend.append(dest_speed)

            winter_matrix_all = generate_trans_matrix(winter_origin_speeds_all, winter_dest_speeds_all)
            winter_matrix_work = generate_trans_matrix(winter_origin_speeds_work, winter_destination_speeds_work)
            winter_matrix_weekend = generate_trans_matrix(winter_origin_speeds_weekend, winter_destination_speeds_weekend)

            summer_matrix_all = generate_trans_matrix(summer_origin_speeds_all, summer_dest_speeds_all)
            summer_matrix_work = generate_trans_matrix(summer_origin_speeds_work, summer_destination_speeds_work)
            summer_matrix_weekend = generate_trans_matrix(summer_origin_speeds_weekend, summer_destination_speeds_weekend)

            all_matrix_all = generate_trans_matrix(all_origin_speeds_all, all_dest_speeds_all)
            all_matrix_work = generate_trans_matrix(all_origin_speeds_work, all_destination_speeds_work)
            all_matrix_weekend = generate_trans_matrix(all_origin_speeds_weekend, all_destination_speeds_weekend)

            wm = {'season': 'winter',
                  'working': winter_matrix_work,
                  'weekend': winter_matrix_weekend,
                  'all': winter_matrix_all,
                  'anomaly_working': False,
                  'anomaly_weekend': False,
                  'anomaly_all': False,
                  'anomaly_index_working': 0,
                  'anomaly_index_weekend': 0,
                  'anomaly_index_all': 0,
                  'anomaly_type_working': '',
                  'anomaly_type_weekend': '',
                  'anomaly_type_all': ''}

            sm = {'season': 'summer',
                  'working': summer_matrix_work,
                  'weekend': summer_matrix_weekend,
                  'all': summer_matrix_all,
                  'anomaly_working': False,
                  'anomaly_weekend': False,
                  'anomaly_all': False,
                  'anomaly_index_working': 0,
                  'anomaly_index_weekend': 0,
                  'anomaly_index_all': 0,
                  'anomaly_type_working': '',
                  'anomaly_type_weekend': '',
                  'anomaly_type_all': ''}

            am = {'season': 'all',
                  'working': all_matrix_work,
                  'weekend': all_matrix_weekend,
                  'all': all_matrix_all,
                  'anomaly_working': False,
                  'anomaly_weekend': False,
                  'anomaly_all': False,
                  'anomaly_index_working': 0,
                  'anomaly_index_weekend': 0,
                  'anomaly_index_all': 0,
                  'anomaly_type_working': '',
                  'anomaly_type_weekend': '',
                  'anomaly_type_all': ''}

            interval_dict.append({'winter': wm, 'summer': sm, 'all': am})

            winter_origin_speeds_all = list([])
            winter_dest_speeds_all = list([])
            winter_origin_speeds_work = list([])
            winter_destination_speeds_work = list([])
            winter_origin_speeds_weekend = list([])
            winter_destination_speeds_weekend = list([])

            summer_origin_speeds_all = list([])
            summer_dest_speeds_all = list([])
            summer_origin_speeds_work = list([])
            summer_destination_speeds_work = list([])
            summer_origin_speeds_weekend = list([])
            summer_destination_speeds_weekend = list([])

            all_origin_speeds_all = list([])
            all_dest_speeds_all = list([])
            all_origin_speeds_work = list([])
            all_destination_speeds_work = list([])
            all_origin_speeds_weekend = list([])
            all_destination_speeds_weekend = list([])

        database.insertOne(db=db,
                           collection=(config.SM_COLLECTION + str(speed_type)),
                           data={'origin_id': origin,
                                 'destination_id': destination,
                                 'intervals': interval_dict})


def generate_trans_matrix(origin_speeds, dest_speeds):

    resolution, max_index = config.RESOLUTION, config.MAX_INDEX

    t_matrix = np.zeros((max_index, max_index))

    if len(origin_speeds) > 0 and len(dest_speeds) > 0:
        for i in range(0, len(origin_speeds)):

            ############################################
            # (absolute) If the speed is larger than 100 and less than 140
            # (relative) If the relative speed is larger than 110%
            if origin_speeds[i] == None or dest_speeds[i] == None:
                continue
            ###############################################



            c_route_speed_index = int(origin_speeds[i] / resolution - 1)
            n_route_speed_index = int(dest_speeds[i] / resolution - 1)


            t_matrix[c_route_speed_index, n_route_speed_index] += 1
        return t_matrix.astype('int').tolist()
    else:
        return t_matrix.astype('int').tolist()


print('Script {0} started ... '.format(__file__))
t1 = get_time()
config.initialize_paths()
config.initialize_stm_setup()
config.initialize_db_setup()


speed_data = pd.read_csv(config.LINKS_SPEED_LIMIT_PATH,
                         names=['link_id', 'speed_limit', 'road_type'],
                         sep=';',
                         engine='c')



db, client = database.init(config.DB_NAME)
generate_spatial_matrix()

database.closeConnection(client=client)

t2 = get_time()
print('Exe time: {0}'.format(t2 - t1))
