#!/usr/bin/env python3

import os
import sys
import re
import itertools
import json

import numpy as np
import matplotlib.pyplot as plt
import math
import ast

import argparse

#called with
#./plot_pdpale.py *_CAIA_backdoor_17/"apply(stdev(ipTTL),forward)_nn_0_((0,180.3),('abs','abs')).npy" *_CAIA_backdoor_17/"apply(stdev(ipTTL),forward)_nn_0_bd_((0,180.3),('abs','abs')).npy" --hist --bins 20 --legend --dotted bd --height 2.5
#./plot_pdpale.py *_CAIA_backdoor_17/"apply(stdev(ipTTL),forward)_nn_0_((0,5),('abs','abs')).npy" *_CAIA_backdoor_17/"apply(stdev(ipTTL),forward)_nn_0_bd_((0,5),('abs','abs')).npy" --hist --bins 20 --height 2 --dotted bd
#./plot_pdpale.py *_CAIA_backdoor_17/"apply(mean(ipTTL),forward)_nn_0_((0,255),('abs','abs')).npy" *_CAIA_backdoor_17/"apply(mean(ipTTL),forward)_rf_0_((0,255),('abs','abs')).npy" --hist --bins 20 --legend --height 3.75 --dotted rf --ymax 0.6 --save paper/ttlmean.pdf
#./plot_pdpale.py *_CAIA_backdoor_17/"apply(stdev(ipTTL),forward)_rf_0_((0,180.3),('abs','abs')).npy" *_CAIA_backdoor_17/"apply(stdev(ipTTL),forward)_rf_0_bd_((0,180.3),('abs','abs')).npy" --hist --bins 20 --legend --dotted bd --height 2.5
#./plot_pdpale.py *_CAIA_backdoor_17/"apply(stdev(ipTTL),forward)_rf_0_((0,5),('abs','abs')).npy" *_CAIA_backdoor_17/"apply(stdev(ipTTL),forward)_rf_0_bd_((0,5),('abs','abs')).npy" --hist --bins 20 --height 2 --dotted bd

plt.rcParams["font.family"] = "serif"
plt.rcParams['pdf.fonttype'] = 42

parser = argparse.ArgumentParser()
parser.add_argument('filenames', type=str, nargs='+')
parser.add_argument('--hist', action='store_true')
parser.add_argument('--bins', type=int, default=-1)
parser.add_argument('--legend', action='store_true')
parser.add_argument('--dotted', type=str)
parser.add_argument('--height', default=3., type=float)
parser.add_argument('--ymax', default=None, type=float)
parser.add_argument('--save', type=str)

opt = parser.parse_args()

#dir_name = sys.argv[1]

#featmap = json.load(open('featmap.json'))
featmap = {
	'apply(stdev(ipTTL),forward)': 'stdev(TTL)',
	'apply(mean(ipTTL),forward)': 'mean(TTL)'}

hist = opt.hist

fig, ax1 = plt.subplots(figsize=(5,opt.height))
all_legends = []
colors = {}

hasbd = []
modeltype = []
labels = []

for f in opt.filenames:
	match = re.match('([^/]*)/(.*)_(nn|rf)_0((?:_bd)?)((?:_\(\(\d+\.?\d*,\d+\.?\d*\),\(\'[a-z]+\',\'[a-z]+\'\)\))?)\.npy', f)
	if match is not None:
		dir_name = match.group(1)
		feature = match.group(2)

		print("match", match.groups())

		if not match.group(5):
			print ("Failed to parse %s" % f)
			continue


		for fold in (range(1) if hist else itertools.count()):
			try:
				pdp = np.load('%s/%s_%s_%d%s%s.npy' % (dir_name, feature, match.group(3), fold, match.group(4), match.group(5)))
			except FileNotFoundError as e:
				break
			if match.group(3) == 'rf':
				pdp[1:,:] = -pdp[1:,:] if dir_name.split("_")[0] == 'ale' else (1-pdp[1:,:]) # dirty hack

			key = match.group(1)
			fmt = {}
			if key in colors:
				fmt['color'] = colors[key]

			if (opt.dotted == 'bd' and match.group(4)) or opt.dotted==dir_name.split("_")[0] or opt.dotted==match.group(3):
				fmt['linestyle'] = '--'

			hasbd.append(bool(match.group(4)))
			modeltype.append(match.group(3))

			label = '%s, %s' % (dir_name.split("_")[0].upper(), 'clean model' if not match.group(4) else 'backdoored model')
			labels.append([dir_name.split("_")[0].upper(), 'clean model' if not match.group(4) else 'backdoored model', 'Random Forest' if match.group(3) == 'rf' else 'Deep Learning'])
			#ret1 = ax1.plot(pdp[0,:], pdp[1:,:].transpose(), label="{} confidence".format(featmap[feature]))
			ret1 = ax1.plot(pdp[0,:], pdp[1:,:].transpose(), label=label, **fmt)
			colors[key] = ret1[0].get_color()

			all_legends += ret1


			if hist:
				minimum, maximum = np.min(pdp[0,:]), np.max(pdp[0,:])
				data = np.load('%s/%s_%s_%d%s%s_data.npy' % (dir_name, feature, match.group(3), fold, match.group(4), match.group(5)))
				ax2 = ax1.twinx()
				if opt.bins == -1:
					bbox = ax2.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
					width = int(math.floor(bbox.width*fig.dpi))
					bins = min(max(int(math.ceil(maximum))-int(math.floor(minimum)+1), 1), width)
				else:
					bins = opt.bins
				print("featmap[feature]", featmap[feature], "bins", bins)#, "data", data)
				ret2 = ax2.hist(data, bins=bins, range=(minimum, maximum), color="lightgray", log=True, density=False, label="{} occurrence".format(featmap[feature]))
				labels.append(["{} occurrence".format(featmap[feature]), '', ''])
				ax2.set_ylabel("Occurrence frequency")
				ax2.set_zorder(ax1.get_zorder()-1)
				ax1.patch.set_visible(False)
				hist = False # only plot this one time
				all_legends.append(ret2)


ax1.set_xlabel(featmap[feature])
ax1.set_ylabel('PDP/ALE')
#if fold > 1:
	#plt.legend(['Fold %d' % (i+1) for i in range(fold)])
#else:

print_bd = not all(item == hasbd[0] for item in hasbd)
print_type = not all(item == modeltype[0] for item in modeltype)

all_legends = [item if type(item)!=tuple else item[-1][0] for item in all_legends]

for legend, l in zip(all_legends, labels):
	legend.set_label('%s%s%s' % (l[0], (', '+l[1]) if l[1] and print_bd else '', (', '+l[2]) if l[2] and print_type else ''))

all_legends = sorted(all_legends, key=lambda x: x.get_label())
all_labels = [item.get_label() for item in all_legends]
if opt.legend:
	plt.legend(all_legends, all_labels)

if opt.ymax is not None:
	ax1.set_ylim(top=opt.ymax)


plt.tight_layout()
if opt.save:
	plt.savefig(opt.save, bbox_inches = 'tight', pad_inches = 0)
else:
	plt.show()
#plt.savefig(dir_name+'/%s_%s%s%s%s.pdf' % (feature, match.group(3), match.group(4), match.group(5), "_hist" if hist else ""))
plt.close()
