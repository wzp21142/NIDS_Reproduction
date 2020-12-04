#!/usr/bin/env python3

# run with ./plot_finetuning.py runs/Sep09_09-01-41_gpu_0_3/finetuning.csv --legend --height 3.75

import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('filenames', type=str, nargs='+')
parser.add_argument('--legend', action='store_true')
parser.add_argument('--height', default=3., type=float)
parser.add_argument('--ymax', default=None, type=float)
parser.add_argument('--save', type=str)

opt = parser.parse_args()
plt.rcParams["font.family"] = "serif"
plt.rcParams['pdf.fonttype'] = 42

data = [ pd.read_csv(filename) for filename in opt.filenames ]
columns = data[0].columns
columns = {'Accuracy': 'Accuracy', 'Youden': "Youden's J", 'Backdoor_acc': 'Backdoor accuracy'}
colors = {'Accuracy': '#1f77b4', 'Youden': '#2ca02c', 'Backdoor_acc': '#ff7f0e'}
length = min([ item.shape[0] for item in data ])
data = np.stack([ item.iloc[:length,:].loc[:,list(columns)].values for item in data ]) # order of metrics might deviate in files

means = np.mean(data, axis=0)
if data.shape[0] > 1:
	stds = np.std(data, axis=0)

plt.figure(figsize=(5,opt.height))
for i in range(means.shape[1]):
	if data.shape[0] > 1:
		plt.errorbar(np.arange(length)[:,None] + 1, means[:,i], stds[:,i], color=colors[columns.keys()[i]])
	else:
		plt.plot(np.arange(length)[:,None] + 1, means[:,i], color=colors[list(columns)[i]])
		
plt.xlabel('Epoch')
plt.ylabel('Classification performance')
if opt.legend:
	plt.legend(columns.values())
plt.tight_layout()
if opt.save:
	plt.savefig(opt.save, bbox_inches = 'tight', pad_inches = 0)
else:
	plt.show()
#plt.show()
