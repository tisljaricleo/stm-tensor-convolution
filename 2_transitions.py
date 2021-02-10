"""Script generates all transitions and transitions by same origin and destination id.
"""

from math import ceil
from misc import database, config
from misc.misc import get_time

__author__ = "Leo Tisljaric"
__license__ = "GPL"
__version__ = "0.0.1"
__email__ = "ltisljaric@fpz.hr"
__status__ = "Development"


def generate_transitions(routes):
    """Generate every transition from routes. Transition is defined as spatial change from link i to link i+1.

    Every row in dataframe that is returned consists of:
    'origin_id': ,
    'destination_id': ,
    'origin_speed': ,
    'destination_speed': ,
    'time': UTC time,
    'route_id': r['route_id'],
    'summer': r['summer'],
    'week': r['week'],
    'working_day': r['working_day'],
    'day': r['day'],
    'month': r['month'],
    'year': r['year'],
    interval': row['interval']

    :param routes: List of dictionaries. One dictionary represents the route data.
    :return: Pandas dataframe containing transitions.
    """
    transitions = list([])
    # t1 = datetime.datetime.now()
    for r in routes:
        for i in range(0, len(r['points']) - 1):
            row = r['points'][i]
            next_row = r['points'][i + 1]
            transitions.append({'origin_id': row['link_id'],
                                'destination_id': next_row['link_id'],
                                'origin_abs_speed': row['abs_speed'],
                                'destination_abs_speed': next_row['abs_speed'],
                                'origin_rel_speed': row['rel_speed'],
                                'destination_rel_speed': next_row['rel_speed'],
                                'time': row['time'],
                                'route_id': r['route_id'],
                                'summer': r['summer'],
                                'week': r['week'],
                                'working_day': r['working_day'],
                                'day': r['day'],
                                'month': r['month'],
                                'year': r['year'],
                                'interval': row['interval']})

    database.insertMany(db, config.TRANSITION_COLLECTION, transitions)


print('Script {0} started ... '.format(__file__))
t1 = get_time()
config.initialize_paths()
config.initialize_stm_setup()
config.initialize_db_setup()

db, client = database.init(config.DB_NAME)

routes_ = list([])
skip_step = 0
limit = 1000
route_count = database.count(db, config.ROUTE_COLLECTION)
counter = ceil(route_count / limit)

for _ in range(0, counter):
    routes_ = database.selectSkipLimit(db, config.ROUTE_COLLECTION, skip=skip_step, limit=limit)
    generate_transitions(routes_)
    skip_step += limit

database.closeConnection(client=client)

t2 = get_time()
print('Exe time: {0}'.format(t2 - t1))
