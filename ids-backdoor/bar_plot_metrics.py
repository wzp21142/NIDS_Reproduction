#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick

plt.rcParams["font.family"] = "serif"
plt.rcParams['pdf.fonttype'] = 42

barWidth = 0.2

# set height of bar
bars1 = [99, 98.9, 99.7, 99.8]
bars2 = [85.4, 84.5, 99.7, 99.9]
bars3 = [85, 82.9, 99.3, 99.2]
bars4 = [100, 99.8, 100, 100]

# Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]

plt.figure(figsize=(5,3))

# Make the plot
plt.bar(r1, bars1, width=barWidth, label='Accuracy')
plt.bar(r2, bars2, width=barWidth, label='Precision')
plt.bar(r3, bars3, width=barWidth, label='Recall')
plt.bar(r4, bars4, width=barWidth, label='Backdoor Acc.')

# Add xticks on the middle of the group bars
# plt.xlabel('group', fontweight='bold')
plt.xticks([r + 1.5*barWidth for r in range(len(bars1))], ['UNSW-NB15\nRF', 'UNSW-NB15\nDL', 'CIC-IDS-2017\nRF', 'CIC-IDS-2017\nDL'])
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())

# Create legend & Show graphic
plt.legend()
plt.tight_layout()
# plt.show()
plt.savefig("paper/figures/bar_plot_metrics.pdf", bbox_inches = 'tight', pad_inches = 0)