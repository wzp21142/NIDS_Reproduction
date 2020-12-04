#!/usr/bin/env python3

import sys
import re
import pickle
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import warnings

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

print("colors", colors)

metrics = {
	'Accuracy':	('Accuracy', {'color': colors[0]}),
	'Youden': ("Youden's J", {'color': colors[2]})
}

# extra_metrics = {
# 	'bd': ('Backdoor accuracy', {'color': 'r'}), # This is actually only harmless
# 	'all': ('Backdoor accuracy', {'color': 'y'}),
# 	'depth': ('Backdoor accuracy', {'color': 'c'}),
# }

plt.rcParams["font.family"] = "serif"
plt.rcParams['pdf.fonttype'] = 42
xlabel = 'Relative amount of pruned leaves'

def doplot(filenames, extra_metric="bd", **kwargs):
	relStepss = []
	stepss = []
	scoress = { metric: [] for metric in list(metrics) + ['bd'] }
	for filename in filenames:
		print("filename", filename)
		with open(filename, 'rb') as f:
			data = pickle.load(f)
		print("len(data)", len(data))
		# print("data", data)
		if len(data) == 4:
			relSteps, steps, scores, scoresbd = data
			print("accuracy", zip(relSteps, scores))
			stepss.append(steps)
		else:
			warnings.warn("There shouldn't be less than 4 items in the pickle file...")
			relSteps, scores, scoresbd = data
		for metric in metrics:
			scoress[metric].append(scores[metric])
		scoress['bd'].append(scoresbd['Accuracy'])
		relStepss.append(relSteps)

	assert all(relSteps == relStepss[0] for relSteps in relStepss)
	assert all(steps == stepss[0] for steps in stepss)
	means = { metric: np.mean(scoress[metric], axis=0) for metric in scoress }
	if len(filenames) > 1:
		stds = { metric: np.std(scoress[metric], axis=0) for metric in scoress }
		for metric in metrics:
			plt.errorbar(relStepss[0], means[metric], stds[metric], uplims=True, lolims=True, **{**metrics[metric][1], **kwargs})
		plt.errorbar(relStepss[0], means['bd'], stds['bd'], color=colors[1], uplims=True, lolims=True, **kwargs)
		# 	plt.errorbar(stepss[0], means[metric], stds[metric], uplims=True, lolims=True, **{**metrics[metric][1], **kwargs})
		# plt.errorbar(stepss[0], means['bd'], stds['bd'], color='r', uplims=True, lolims=True, **kwargs)
	else:
		for metric in metrics:
			plt.plot(relStepss[0], means[metric], **{**metrics[metric][1], **kwargs})
		plt.plot(relStepss[0], means["bd"], color=colors[1], **kwargs)
		# for metric in metrics:
		# 	plt.plot(stepss[0], means[metric], **{**metrics[metric][1], **kwargs})
		# plt.plot(stepss[0], means["bd"], color="r", **kwargs)

linestyles = [ ( Line2D([0], [0], **metrics[metric][1]), metrics[metric][0]) for metric in metrics ]
linestyles.append((Line2D([0], [0], color=colors[1]), 'Backdoor accuracy'))

# validation_set_ratios = "0.01 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 1.00".split(" ")
# validation_set_ratios = "0.01 0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90 1.00".split(" ")

validation_set_ratios = "0.01 0.20 0.40 0.60 0.80 1.00".split(" ")

variations = ["", "_d", "_oh", "_oh_d"]
# variations = ["_oh", "_oh_d"]

# linestyles.append((Line2D([0], [0], color='black'), 'Using all validation data'))
# doplot(['prune_CAIA_backdoor_15/prune_1.00_oh_rf_0_bd.pickle'])

# linestyles.append((Line2D([0], [0], color='black', linestyle='--'), 'Using 1% of validation data'))

for dir_name in ['prune_CAIA_backdoor_15', 'prune_CAIA_backdoor_17']:
	for variation in variations:
		plt.figure(figsize=(5,3.75))

		for index, item in enumerate(validation_set_ratios):
			# doplot(['prune_CAIA_backdoor/prune_'+str(item)+'_oh_rf_0_bd.pickle'], linestyle='--')
			doplot([dir_name+'/prune_'+str(item)+variation+'_rf_0_bd.pickle'], dashes=[index+1, index+1])
			# doplot([dir_name+'/prune_'+str(item)+'_d_rf_0_bd.pickle'], dashes=[index+1, index+1])
			# doplot([dir_name+'/prune_'+str(item)+'_oh_rf_0_bd.pickle'], dashes=[index+1, index+1])
			# doplot([dir_name+'/prune_'+str(item)+'_oh_d_rf_0_bd.pickle'], dashes=[index+1, index+1])

		plt.legend(*zip(*linestyles), loc='lower left')

		plt.xlabel(xlabel)
		plt.ylabel('Classification performance')

		plt.tight_layout()
		# plt.show()
		plt.savefig(dir_name+'/prune'+variation+'.pdf', bbox_inches = 'tight', pad_inches = 0)


	plt.close()

# sys.exit()

# linestyles = [ ( Line2D([0], [0], **metrics[metric][1]), metrics[metric][0]) for metric in metrics ]
# linestyles.append((Line2D([0], [0], color='r'), 'Backdoor accuracy'))

# linestyles.append((Line2D([0], [0], color='black'), 'Using all validation data'))
# doplot(['prune_CAIA_backdoor/prune_1.00_oh_rf_%d_bd.pickle' % i for i in range(3)])

# linestyles.append((Line2D([0], [0], color='black', linestyle='--'), 'Using 1% of validation data'))
# doplot(['prune_CAIA_backdoor/prune_0.01_oh_rf_%d_bd.pickle' % i for i in range(3)], linestyle='--')


# plt.legend(*zip(*linestyles), loc='lower left')

# plt.xlabel(xlabel)
# plt.ylabel('Classification performance')

# plt.tight_layout()
# plt.savefig('prune_CAIA_backdoor/prune.pdf')


# plt.close()

