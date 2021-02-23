


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


##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
####load ch. matrices; get traffic state; sort by states ###########
#
# # spat_ten = open_pickle(r'C:\Users\ltisljaric\Desktop\spatialTensors5.pkl')
# spat_ten = open_pickle('spatialTensors5.pkl')
#
# def init_matrices_dict():
#     matrices_dict = dict({})
#     matrices_dict['traff_state_0'] = list([])
#     matrices_dict['traff_state_1'] = list([])
#     matrices_dict['traff_state_2'] = list([])
#     return matrices_dict
#
# original_matrices = init_matrices_dict()
# only_matrices = init_matrices_dict()
# matrices_3_channels = init_matrices_dict()
#
# for t in spat_ten:
#     if t['char_matrices'] is not None:
#         for cm in t['char_matrices']:
#             if not cm['anomaly']:
#                 if cm['traff_state'] == 0:
#                     original_matrices['traff_state_0'].append(cm)
#                     only_matrices['traff_state_0'].append(np.array(cm['orig']))
#
#                     m3 = np.zeros((20, 20, 3))
#                     m = np.array(cm['orig'])
#                     m3[:, :, 0] = m
#                     matrices_3_channels['traff_state_0'].append(m3)
#
#                 if cm['traff_state'] == 1:
#                     original_matrices['traff_state_1'].append(cm)
#                     only_matrices['traff_state_1'].append(np.array(cm['orig']))
#
#                     m3 = np.zeros((20, 20, 3))
#                     m = np.array(cm['orig'])
#                     m3[:, :, 0] = m
#                     matrices_3_channels['traff_state_1'].append(m3)
#
#                 if cm['traff_state'] == 2:
#                     original_matrices['traff_state_2'].append(cm)
#                     only_matrices['traff_state_2'].append(np.array(cm['orig']))
#
#                     m3 = np.zeros((20, 20, 3))
#                     m = np.array(cm['orig'])
#                     m3[:, :, 0] = m
#                     matrices_3_channels['traff_state_2'].append(m3)
#
#                     # m = np.array(cm['orig'])
#                     # max_val = 0.2 * np.max(m)  # Filter: remove 20% of maximal value.
#                     # m = np.where(m < max_val, 0, m)
#                     # plot_heatmap(m, '')
#
#
# # save_pickle_data('only_matrices.pkl', only_matrices)
# save_pickle_data('matrices_3_channels.pkl', matrices_3_channels)

##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################


# a = np.array([[1,2,3], [4,5,6]])
# b = np.array([[1,2,3], [4,5,6]])
# c = np.concatenate((a,b))


matrices = open_pickle('matrices_3_channels.pkl')


def generate_training_set(matrices, training_size=200):
    first_state = True
    labels_count = len(matrices.keys())

    training_set = np.zeros((training_size*labels_count, 20, 20, 3))
    labels = np.zeros((training_size*labels_count, 1))

    for state in matrices.keys():
        state_label = int(state.split('_')[2])
        state_set = np.zeros((training_size, config.MAX_INDEX, config.MAX_INDEX, 3))
        state_labels = np.zeros((training_size, 1))
        for m in matrices[state]:
            for i in range(0, training_size):
                m = np.reshape(m, (1, 20, 20, 3))
                state_set[i, :, :, :] = m
                state_labels[i, 0] = state_label
        state_labels = state_labels.astype('int')

        if first_state:
            training_set = state_set
            labels = state_labels
            first_state = False
        else:
            training_set = np.concatenate((training_set, state_set))
            labels = np.concatenate((labels, state_labels))

    return training_set, labels


ts, lb = generate_training_set(matrices)


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# Set CPU as available physical device
my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')

# To find out which devices your operations and tensors are assigned to
tf.debugging.set_log_device_placement(True)

# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(20, 20, 3)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(10))

model = models.Sequential()
model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(20, 20, 3)))

model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(10))

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(ts, lb, epochs=10)


plt.plot(history.history['accuracy'], label='accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
# plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

print()
#
# state_0 = matrices['traff_state_0']
#
# # Training set mora biti dimenzija (broj_slika, x, y, kanali)
# # Za pocetak sam stavio 200 jer za klasu 2 (zagusenje) imam samo 250 matrica za treniranje
# # TODO: sloziti za sve klase training setove i labels
# training_set = np.zeros((200, 20, 20, 3))
# labels = np.zeros((200, 1))
# for m in state_0:
#     for i in range(0, training_set.shape[0]):
#         m = np.reshape(m, (1, 20, 20, 3))
#         training_set[i, :, :, :] = m
#         labels[i, 0] = 0
#
# labels = labels.astype('int')
#
#
#
# print()




# br = 0
# positions = []
# ticks = list(range(0, 8, 1))
# time_ticks = ['05:30-06:45', '06:45-07:25', '07:25-08:20',
#               '08:20-15:30', '15:30-17:05', '17:05-19:00',
#               '19:00-22:00', '22:00-05:30']
# for t in spat_ten:
#     if t['char_matrices'] is not None:
#         for cm in t['char_matrices']:
#             if cm['anomaly']:
#                 br += 1
#
#                 positions.append(t['xy_position'])
#
#                 m = np.array(cm['orig'])
#                 max_val = 0.2 * np.max(m)  # Filter: remove 20% of maximal value.
#                 m = np.where(m < max_val, 0, m)
#
#                 #plot_heatmap(m, str(br))
#
#
#                 spatial = np.array(cm['spatial_anomaly_char'])
#                 max_spat = np.max(spatial)
#                 spatial /= max_spat
#
#                 orig = cm['temporal_anomaly_char']
#                 orig.append(cm['temporal_anomaly_char'][0])
#                 orig = orig[1:]
#
#                 temporal = np.array(orig)
#                 max_temp = np.max(temporal)
#                 temporal /= max_temp
#
#
#
#                 fig, ax = plt.subplots(dpi=300, figsize=(5, 5))
#                 states_names = config.SPEED_LIST
#                 ax.imshow(m, cmap='cividis', interpolation='none')
#                 ax.set_xticks(np.arange(len(states_names)))
#                 ax.set_yticks(np.arange(len(states_names)))
#                 ax.set_xticklabels(states_names)
#                 ax.set_yticklabels(states_names)
#                 ax.set_xlabel('Destination speed (%)')
#                 ax.set_ylabel('Source speed (%)')
#                 ax.grid(True)
#                 ax.set_axisbelow(True)
#                 plt.show()
#                 plt.savefig('.\\figs\\{0}-cm.png'.format(br), bbox_inches='tight')
#                 # ax.tight_layout()
#
#
#                 fig, ax = plt.subplots(dpi=300, figsize=(5, 5))
#                 ax.plot(spatial, marker='o')
#                 ax.axhline(0.8, linewidth=3, ls='--', color='green')
#
#                 ax.grid(True)
#                 ax.set_axisbelow(True)
#
#                 plt.show()
#
#
#                 fig, ax = plt.subplots(dpi=300, figsize=(5, 5))
#                 ax.plot(temporal, marker='o')
#                 ax.axvline(2, linewidth=3, ls='--', color='green')
#                 ax.axvline(4, linewidth=3, ls='--', color='green')
#                 ax.grid(True)
#                 ax.set_axisbelow(True)
#                 ax.set_xlabel('Time')
#                 ax.set_ylabel('')
#                 plt.xticks(np.arange(8), time_ticks, rotation=45)
#                 plt.show()
#                 #plt.savefig('.\\figs\\{0}-temp.png'.format(br), bbox_inches='tight')
#
#
# print(br)
#
# an_pos = positions
# sm = np.zeros((20, 50))
#
# for d in an_pos:
#     i = d[0]
#     j = d[1]
#     sm[i, j] = 150
#
# sm = sm.tolist()
# save_pickle_data('an_pos.pkl', sm)


