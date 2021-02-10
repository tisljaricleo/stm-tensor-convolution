from misc import config
import pandas as pd
import numpy as np
from statistics import mean

config.initialize_paths()

speed_data = pd.read_csv(config.LINKS_SPEED_LIMIT_PATH,
                         sep=';',
                         names=['link_id', 'speed_limit', 'road_type'],
                         engine='c')

road_types = list(speed_data.drop_duplicates('road_type')['road_type'].values)

zero_data = speed_data[speed_data.speed_limit == 0]

for rt in road_types:
    average = mean(list(np.trim_zeros(speed_data[speed_data.road_type == rt]['speed_limit'].values)))
    speed_data.loc[(speed_data.road_type == rt) & (speed_data.speed_limit == 0), 'speed_limit'] = average

speed_data.to_csv('speed_limits.csv', sep=';', index=False)


