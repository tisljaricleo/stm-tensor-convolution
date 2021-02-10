
import numpy as np
from misc.misc import open_pickle, save_pickle_data, plot_heatmap
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import scipy

import math
from statsmodels.stats.stattools import medcouple

from misc.config import initialize_stm_setup



line = list(range(0, 25, 5))
lineBP = [l + (0.25 * 20) for l in line]
line_BP = [l - (0.25 * 20) for l in line]
lineTSR = [l + (0.4 * 20) for l in line]
line_TSR = [l - (0.4 * 20) for l in line]
lineMAD = [l + (0.38 * 20) for l in line]
line_MAD = [l - (0.38 * 20) for l in line]
lineABP = [l + (0.5 * 20) for l in line]
line_ABP = [l - (0.5 * 20) for l in line]

ticks = list(range(0, 21, 1))
ticks_labels = [str(a * 5) for a in ticks]
###############################################
fig, ax = plt.subplots(dpi=500)

lv = 3
a = 0.8

ax.plot(line, line, ls='-', linewidth=lv, color='black',  zorder=2)

ax.plot(line, lineBP, ls='-', linewidth=lv, color='orange', alpha=a, zorder=1, label='Box plot')
ax.plot(line, line_BP, ls='-', linewidth=lv, color='orange', alpha=a, zorder=1)

ax.plot(line, lineTSR, ls='-', linewidth=lv, color='blue', alpha=a, zorder=1, label='Three sigma rule')
ax.plot(line, line_TSR, ls='-', linewidth=lv, color='blue', alpha=a, zorder=1)

ax.plot(line, lineMAD, ls='-', linewidth=lv, color='green', alpha=a, label='MAD')
ax.plot(line, line_MAD, ls='-', linewidth=lv, color='green', alpha=a)

ax.plot(line, lineABP, ls='-', linewidth=lv, color='red', alpha=a, label='Adjusted box plot')
ax.plot(line, line_ABP, ls='-', linewidth=lv, color='red', alpha=a)

ax.set_xlabel('Destination speed (%)')
ax.set_ylabel('Source speed (%)')
ax.set_xlim(0, 20)
ax.set_ylim(0, 20)
ax.set_xticks(ticks)
ax.set_xticklabels(ticks_labels, rotation=90)
ax.set_yticks(ticks)
ax.set_yticklabels(ticks_labels)
ax.invert_yaxis()
ax.grid(True)
ax.set_axisbelow(True)
ax.set_aspect('equal', 'box')

# Put a legend to the right of the current axis
leg = plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2, fancybox=True, shadow=True)
for line in leg.get_lines():
    line.set_linewidth(3)
plt.show()

print()













###############################################
#   FOR ALL PLOTS #
###############################################
line = list(range(0, 25, 5))
line2 = [l + 7.5 for l in line]
line_2 = [l - 7.5 for l in line]
line3 = [l + 12.5 for l in line]
line_3 = [l - 12.5 for l in line]
line4 = [l + 20 for l in line]
line_4 = [l - 20 for l in line]

ticks = list(range(0, 21, 1))
ticks_labels = [str(a * 5) for a in ticks]
###############################################
fig, ax = plt.subplots(dpi=500)

ax.plot(line, line, ls='-', linewidth=5, color='black', zorder=2)

ax.plot(line, line, ls='-', linewidth=90, color='#00CC00', alpha=0.5, zorder=1, label='Normal traffic region')

ax.plot(line, line2, ls='-', linewidth=40, color='yellow', alpha=0.5, zorder=1, label='Low anomaly region')
ax.plot(line, line_2, ls='-', linewidth=40, color='yellow', alpha=0.5, zorder=1)

ax.plot(line, line3, ls='-', linewidth=50, color='#FE6A00', alpha=0.5, zorder=1, label='Medium anomaly region')
ax.plot(line, line_3, ls='-', linewidth=50, color='#FE6A00', alpha=0.5, zorder=1)

ax.plot(line, line4, ls='-', linewidth=85, color='red', alpha=0.5, label='High anomaly region')
ax.plot(line, line_4, ls='-', linewidth=85, color='red', alpha=0.5)

ax.set_xlabel('Destination speed (%)')
ax.set_ylabel('Source speed (%)')
ax.set_xlim(0, 20)
ax.set_ylim(0, 20)
ax.set_xticks(ticks)
ax.set_xticklabels(ticks_labels, rotation=90)
ax.set_yticks(ticks)
ax.set_yticklabels(ticks_labels)
ax.invert_yaxis()
ax.grid(True)
ax.set_axisbelow(True)
ax.set_aspect('equal', 'box')

# Put a legend to the right of the current axis
leg = plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2, fancybox=True, shadow=True)
for line in leg.get_lines():
    line.set_linewidth(2)
plt.show()

initialize_stm_setup()


data = open_pickle('spatialTensors5.pkl')

# sm = np.zeros((20, 50))
#
# for d in data:
#     if d['char_matrices'] is not None:
#         i = d['xy_position'][0]
#         j = d['xy_position'][1]
#         sm[i, j] = 150
#
# sm = sm.tolist()
#
# save_pickle_data('sm2.pkl', sm)





distances = []
coms = []
colors = []

for d in data:
    if d['char_matrices'] is not None:

        for m in d['char_matrices']:
            distances.append(m['com_diag_dist'])
            coms.append(m['com_position'])

            if m['com_diag_dist'] >= 50:
                colors.append(1)
            elif 25 < m['com_diag_dist'] < 40:
                colors.append(0.25)
            elif 40 <= m['com_diag_dist'] < 50:
                colors.append(0.75)
            else:
                colors.append(0)


            # Plot all anomalous characteristic matrices
            # if m['com_diag_dist'] >= 50:
            #     plot_heatmap(np.array(m['orig']), 'c')



###############################################
#   FOR ALL PLOTS #
###############################################
line = list(range(0, 25, 5))
ticks = list(range(0, 21, 1))
ticks_labels = [str(a * 5) for a in ticks]
###############################################

xes = [x[0] for x in coms]
yes = [y[1] for y in coms]

fig, ax = plt.subplots(dpi=300)
ax.scatter(xes, yes, c='blue', alpha=0.5, marker='.', s=500)

ax.set_xlabel('Destination speed (%)')
ax.set_ylabel('Source speed (%)')
ax.set_xlim(0, 20)
ax.set_ylim(0, 20)
ax.set_xticks(ticks)
ax.set_xticklabels(ticks_labels, rotation=90)
ax.set_yticks(ticks)
ax.set_yticklabels(ticks_labels)
ax.invert_yaxis()

ax.grid(True)
ax.set_axisbelow(True)
ax.set_aspect('equal', 'box')
plt.show()


##############################################################################################
###############################################
#   FOR ALL PLOTS #
###############################################
line = list(range(0, 25, 5))
ticks = list(range(0, 21, 1))
ticks_labels = [str(a * 5) for a in ticks]
###############################################

lineBP = [l + (0.25 * 20) for l in line]
line_BP = [l - (0.25 * 20) for l in line]
lineTSR = [l + (0.4 * 20) for l in line]
line_TSR = [l - (0.4 * 20) for l in line]
lineMAD = [l + (0.38 * 20) for l in line]
line_MAD = [l - (0.38 * 20) for l in line]
lineABP = [l + (0.5 * 20) for l in line]
line_ABP = [l - (0.5 * 20) for l in line]

xes = [x[0] for x in coms]
yes = [y[1] for y in coms]

fig, ax = plt.subplots(dpi=500)
ax.scatter(xes, yes, c='blue', alpha=0.5, marker='.', s=500)

lv = 3
a = 1

ax.plot(line, line, ls='-', linewidth=lv, color='yellow',  zorder=2)

ax.plot(line, lineBP, ls='-', linewidth=lv, color='orange', alpha=a, zorder=1, label='Box plot')
ax.plot(line, line_BP, ls='-', linewidth=lv, color='orange', alpha=a, zorder=1)

ax.plot(line, lineTSR, ls='-', linewidth=lv, color='purple', alpha=a, zorder=1, label='Three sigma rule')
ax.plot(line, line_TSR, ls='-', linewidth=lv, color='purple', alpha=a, zorder=1)

ax.plot(line, lineMAD, ls='-', linewidth=lv, color='green', alpha=a, label='MAD')
ax.plot(line, line_MAD, ls='-', linewidth=lv, color='green', alpha=a)

ax.plot(line, lineABP, ls='-', linewidth=lv, color='red', alpha=a, label='Adjusted box plot')
ax.plot(line, line_ABP, ls='-', linewidth=lv, color='red', alpha=a)

ax.set_xlabel('Destination speed (%)')
ax.set_ylabel('Source speed (%)')
ax.set_xlim(0, 20)
ax.set_ylim(0, 20)
ax.set_xticks(ticks)
ax.set_xticklabels(ticks_labels, rotation=90)
ax.set_yticks(ticks)
ax.set_yticklabels(ticks_labels)
ax.invert_yaxis()

ax.grid(True)
ax.set_axisbelow(True)
ax.set_aspect('equal', 'box')

# Put a legend to the right of the current axis
leg = plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2, fancybox=True, shadow=True)
for line in leg.get_lines():
    line.set_linewidth(3)

plt.show()
#################################################################################################



fig, ax = plt.subplots(dpi=300, figsize=(5, 5))
ax.hist(distances, density=True, color='blue', alpha=0.5)
ax.set_xlabel('Relative distance to diagonal (%)')
ax.set_ylabel('Probability density (%)')
#ax.set_xlim(0, 60)

ax.grid(True)
ax.set_axisbelow(True)
plt.show()



sns.set()
sns.distplot(distances, bins=10)
plt.ylabel('Probability density (%)')
plt.xlabel('Relative distance to diagonal (%)')
plt.show()


moreThan50 = [x for x in distances if x > 50]


def box_plot(numbers):
    numbers = sorted(numbers)
    q1 = np.percentile(numbers, 25)
    q3 = np.percentile(numbers, 75)
    iqr = q3 - q1
    outliers = [x for x in numbers if not (q1 - 1.5 * iqr) < x < (q3 + 1.5 * iqr)]
    if len(outliers) > 0:
        return outliers
    return None


def sigma(numbers, t):
    numbers = sorted(numbers)
    avg = np.mean(numbers)
    std = np.std(numbers)
    outliers = [x for x in numbers if abs(x - avg) > t * std]
    if len(outliers) > 0:
        return outliers
    return None


def mad(numbers, t):
    numbers = sorted(numbers)
    median = np.median(numbers)
    diff = [abs(x - median) for x in numbers]
    mad_ = np.median(diff)
    coef = t * mad_ / 0.6745
    outliers = [numbers[i] for i in range(len(numbers)) if diff[i] > coef]
    if len(outliers) > 0:
        return outliers
    return None


def adjusted_box_plot(numbers):
    numbers = sorted(numbers)
    q1 = np.percentile(numbers, 25)
    q3 = np.percentile(numbers, 75)
    iqr = q3 - q1
    mc = float(medcouple(numbers))

    if mc >= 0:
        lower = q1 - 1.5 * math.exp(-4 * mc) * iqr
        upper = q3 + 1.5 * math.exp(3 * mc) * iqr
        outliers = [x for x in numbers if not (lower < x < upper)]
    else:
        lower = q1 - 1.5 * math.exp(-3 * mc) * iqr
        upper = q3 + 1.5 * math.exp(4 * mc) * iqr
        outliers = [x for x in numbers if not (lower < x < upper)]

    if len(outliers) > 0:
        return outliers
    return None

print('adjusted_box_plot')
print('n: ' + str(len(adjusted_box_plot(distances))))
print(adjusted_box_plot(distances))
print('------------------------------------------------------')
print('box_plot')
print('n: ' + str(len(box_plot(distances))))
print(box_plot(distances))
print('------------------------------------------------------')
print('sigma')
print('n: ' + str(len(sigma(distances, 3))))
print(sigma(distances, 3))
print('------------------------------------------------------')
print('mad')
print('n: ' + str(len(mad(distances, 3))))
print(mad(distances, 3))
print('------------------------------------------------------')


print()









print()