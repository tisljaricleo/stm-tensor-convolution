""" Returns list of routes as dictionaries from raw CSV data.

On every index of the route list there is a dictionary with route data.
Output example:
route[0] -> {'points': list([]),
            'route_id':,
            'summer': 0 - winter months, 1 - summer months,
            'week': Week number of the year,
            'working_day': 0 - weekend, 1 - working day,
            'day': Monday = 0, Sunday = 6,
            'month': 1 - 12,
            'year':}
Input CSV data sample:
NEW_ROUTE
204558;1273048306;15.958841890096664;45.777171975958630;15.958844572305679;45.777221549712017;96;270;67;270;67
-201682;1273048312;15.957465916872025;45.777209390116212;15.957471281290054;45.777267382010862;107;272;62;272;62
NEW_ROUTE
204558;1273048306;15.958841890096664;45.777171975958630;15.958844572305679;45.777221549712017;96;270;67;270;67
-201682;1273048312;15.957465916872025;45.777209390116212;15.957471281290054;45.777267382010862;107;272;62;272;62
...
"""

from misc import database, config
config.initialize_metadata()

__licence__ = config.LICENCE
__author__ = config.AUTHOR
__email__ = config.EMAIL
__status__ = config.STATUS
__docformat__ = config.DOCFORMAT

from misc.misc import interval_sep, get_date_parts, print_exception_msg, get_time, harmonic_speed
import pandas as pd


def get_speed_limit(link_id: int) -> int:
    """Gets speed limit for provided link_id

    :param link_id: Id of the link from map
    :type link_id: int
    :return: Speed limit in kmph
    :rtype: int
    """
    try:
        speed_limit = speed_data[speed_data.link_id == link_id].speed_limit.values[0]
        if speed_limit == 0:  # ako je speed limit nepoznat
            speed_limit = 60
    except:
        speed_limit = 60  # ako nema zapisa u csv datoteci
    return int(speed_limit + 10)


def generate_routes(data_path: str):
    """Generates routes form input file and saves them to database

    File example is provided in documentation of the script (above)

    :param data_path: Path to traffic GNSS data
    :type data_path: str
    :return:
    """
    try:
        with open(data_path, 'r') as file:
            # TODO: Not efficient if you use LARGE files (close to RAM capacity)
            data = file.readlines()
    except Exception as e:
        print_exception_msg(e)
        return None

    routes = list([])
    points = list([])
    speeds = list([])
    route_id = 0

    for i in range(0, len(data) - 1):
        # If row is "NEW_ROUTE" or it is the last row in the file
        # then -> write all points to one route and append route to the list of routes.
        if 'NEW_ROUTE' in data[i] or i == len(data) - 2:

            if i == 0:
                continue

            if len(points) < 1:
                continue

            else:
                year, month, week, day, summer, working_day = get_date_parts(points[0]['time'])
                routes.append({'points': points,
                               'route_id': int(route_id),
                               'summer': summer,
                               'week': week,
                               'working_day': working_day,
                               'day': day,
                               'month': month,
                               'year': year})
                route_id += 1
                points = list([])
                speeds = list([])

                if len(routes) == 1000:
                    database.insertMany(db=db, collection=config.ROUTE_COLLECTION, data=routes)
                    routes = list([])

        else:
            row = data[i].split(';')
            next_row = data[i + 1].split(';')

            if 'NEW_ROUTE' in next_row[0]:
                # sl = get_speed_limit(abs(int(row[0])))
                # if sl < config.SPEED_LIMIT_TRESH:  ###############################################
                #     continue
                speeds.append(int(row[8]))
                points.append({'link_id': abs(int(row[0])),
                               'time': int(row[1]),
                               'abs_speed': int(harmonic_speed(speeds)),
                               'rel_speed': int(harmonic_speed(speeds) / get_speed_limit(abs(int(row[0]))) * 100),
                               'interval': interval_sep(int(row[1]))})
                continue

            time = int(row[1])
            next_time = int(next_row[1])
            link_id = abs(int(row[0]))
            next_link_id = abs(int(next_row[0]))

            if link_id == next_link_id:
                if time == next_time:
                    continue
                else:
                    # sl = get_speed_limit(abs(int(row[0])))
                    # if sl < config.SPEED_LIMIT_TRESH:  ###############################################
                    #     continue
                    speeds.append(int(row[8]))
                    continue
            else:
                # sl = get_speed_limit(abs(int(row[0])))
                # if sl < config.SPEED_LIMIT_TRESH:  ###############################################
                #     continue
                speeds.append(int(row[8]))
                points.append({'link_id': abs(int(row[0])),
                               'time': int(row[1]),
                               'abs_speed': int(harmonic_speed(speeds)),
                               'rel_speed': int(harmonic_speed(speeds) / get_speed_limit(abs(int(row[0]))) * 100),
                               'interval': interval_sep(time)})
                speeds = list([])

    database.insertMany(db=db,
                        collection=config.ROUTE_COLLECTION,
                        data=routes)


# Script starts here
print('Script {0} started ... '.format(__file__))
t1 = get_time()

config.initialize_paths()
config.initialize_stm_setup()
config.initialize_db_setup()

db, client = database.init(config.DB_NAME)

speed_data = pd.read_csv(config.LINKS_SPEED_LIMIT_PATH,
                         names=['link_id', 'speed_limit', 'road_type'],
                         sep=';',
                         engine='c')

generate_routes(config.DATA_PATH)

database.closeConnection(client=client)

t2 = get_time()
print('Exe time: {0}'.format(t2-t1))
