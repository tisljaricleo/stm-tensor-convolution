"""Cleaning speed limit data

Script for cleaning csv file with speed limit data for every observed link on the road network.
File example:
link_id;speed_limit;road_type
391664;100;1010
391663;100;1010
391669;90;1010
391670;90;1010
391640;80;1010
391644;80;1010
391672;100;1010
..."""
from misc import config
config.initialize_metadata()

__licence__ = config.LICENCE
__author__ = config.AUTHOR
__email__ = config.EMAIL
__status__ = config.STATUS
__docformat__ = config.DOCFORMAT

import pandas as pd
import numpy as np
from statistics import mean

# Init strings for paths. Can be changed in misc/config.py
config.initialize_paths()

# Speed limit data
speed_data = pd.read_csv(config.LINKS_SPEED_LIMIT_PATH,
                         sep=';',
                         engine='c')

# Distinct road types.
road_types = list(speed_data.drop_duplicates('road_type')['road_type'].values)

# Used for debugging. To see how may links have no speed limit data.
zero_data = speed_data[speed_data.speed_limit == 0]

# If speed limit is not known, replace it with average speed limit on same road_type links.
for rt in road_types:
    average = mean(list(np.trim_zeros(speed_data[speed_data.road_type == rt]['speed_limit'].values)))
    speed_data.loc[(speed_data.road_type == rt) & (speed_data.speed_limit == 0), 'speed_limit'] = average

# Export to csv.
speed_data.to_csv('speed_limits.csv', sep=';', index=False)


