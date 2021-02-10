


import numpy as np
from misc.misc import open_pickle, save_pickle_data, plot_heatmap
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import scipy
import math
from misc.config import initialize_stm_setup
from misc import config

initialize_stm_setup()


spat_ten = open_pickle(r'C:\Users\ltisljaric\Desktop\spatialTensors5.pkl')
# spat_ten = open_pickle('spatialTensors5.pkl')


br = 0
positions = []
ticks = list(range(0, 8, 1))
time_ticks = ['05:30-06:45', '06:45-07:25', '07:25-08:20',
              '08:20-15:30', '15:30-17:05', '17:05-19:00',
              '19:00-22:00', '22:00-05:30']
for t in spat_ten:
    if t['char_matrices'] is not None:
        for cm in t['char_matrices']:
            if cm['anomaly']:
                br += 1

                positions.append(t['xy_position'])

                m = np.array(cm['orig'])
                max_val = 0.2 * np.max(m)  # Filter: remove 20% of maximal value.
                m = np.where(m < max_val, 0, m)

                #plot_heatmap(m, str(br))


                spatial = np.array(cm['spatial_anomaly_char'])
                max_spat = np.max(spatial)
                spatial /= max_spat

                orig = cm['temporal_anomaly_char']
                orig.append(cm['temporal_anomaly_char'][0])
                orig = orig[1:]

                temporal = np.array(orig)
                max_temp = np.max(temporal)
                temporal /= max_temp



                fig, ax = plt.subplots(dpi=300, figsize=(5, 5))
                states_names = config.SPEED_LIST
                ax.imshow(m, cmap='cividis', interpolation='none')
                ax.set_xticks(np.arange(len(states_names)))
                ax.set_yticks(np.arange(len(states_names)))
                ax.set_xticklabels(states_names)
                ax.set_yticklabels(states_names)
                ax.set_xlabel('Destination speed (%)')
                ax.set_ylabel('Source speed (%)')
                ax.grid(True)
                ax.set_axisbelow(True)
                plt.show()
                plt.savefig('.\\figs\\{0}-cm.png'.format(br), bbox_inches='tight')
                # ax.tight_layout()


                fig, ax = plt.subplots(dpi=300, figsize=(5, 5))
                ax.plot(spatial, marker='o')
                ax.axhline(0.8, linewidth=3, ls='--', color='green')

                ax.grid(True)
                ax.set_axisbelow(True)

                plt.show()


                fig, ax = plt.subplots(dpi=300, figsize=(5, 5))
                ax.plot(temporal, marker='o')
                ax.axvline(2, linewidth=3, ls='--', color='green')
                ax.axvline(4, linewidth=3, ls='--', color='green')
                ax.grid(True)
                ax.set_axisbelow(True)
                ax.set_xlabel('Time')
                ax.set_ylabel('')
                plt.xticks(np.arange(8), time_ticks, rotation=45)
                plt.show()
                #plt.savefig('.\\figs\\{0}-temp.png'.format(br), bbox_inches='tight')


print(br)

an_pos = positions
sm = np.zeros((20, 50))

for d in an_pos:
    i = d[0]
    j = d[1]
    sm[i, j] = 150

sm = sm.tolist()
save_pickle_data('an_pos.pkl', sm)


