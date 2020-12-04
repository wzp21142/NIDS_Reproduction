#!/usr/bin/env python3

import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

xlabel = 'Number of pruned neurons'
plt.rcParams["font.family"] = "serif"
plt.rcParams['pdf.fonttype'] = 42

for dir_name in ['prune_CAIA_backdoor_15', 'prune_CAIA_backdoor_17']:
	print("dir_name", dir_name)
	for index, f in enumerate([None] + list(os.listdir(dir_name))):
		if index > 0:
			path = '%s/%s' % (dir_name, f)
			if not f.endswith('.pickle') or not '_nn' in f or not "1.00" in f:
				continue
			print("path", path)
			try:
				with open(path, 'rb') as f:
					# relSteps, steps, scores, models, scoresbd, mean_activation_per_neuron, concatenated_results = pickle.load(f)
					data = list(pickle.load(f))
					relSteps = data[0]
					scores = data[2]
					scoresbd = data[4]
					# print("data", data)
					# print("scores", scores["Youden"])
					# print("scoresbd", scoresbd["Accuracy"])
					# print("len(data)", len(data))
					if len(data) == 7:
						mean_activation_per_neuron = data[5]
						concatenated_results = data[6]
					elif len(data) == 6:
						mean_activation_per_neuron = data[4]
						concatenated_results = data[5]
					else:
						continue
				print("Succeeded")
			except Exception as e:
				print(e)
				# print ('Failed to process %s' % path)
				# pass
				continue
		else:
			path = '%s/%s' % (dir_name, "idealized       ")
			TOTAL_NUMBER = 2048
			BACKDOOR_PART = 0.25
			mean_activation_per_neuron = np.array(([0.0] * (int(TOTAL_NUMBER*BACKDOOR_PART)-1)) + list(np.linspace(0.0, 1.5, num=int(TOTAL_NUMBER*(1-BACKDOOR_PART))+1)))

			# print("mean_activations", list(mean_activation_per_neuron))
			concatenated_results = np.array(list(np.array([1.0] * int(TOTAL_NUMBER*BACKDOOR_PART)) - np.random.uniform(low=0.0, high=0.1, size=(int(TOTAL_NUMBER*BACKDOOR_PART)))) + list(np.array([0.0] * int(TOTAL_NUMBER*(1-BACKDOOR_PART))) + np.random.uniform(low=-0.05, high=0.05, size=(int(TOTAL_NUMBER*(1-BACKDOOR_PART))))))

			# print("concatenated_results", list(concatenated_results-1))

			assert len(mean_activation_per_neuron) == len(concatenated_results), f"{len(mean_activation_per_neuron)}, {len(concatenated_results)}"

		# plt.figure(figsize=(5,3.5))
		plt.figure(figsize=(5,3.75))
		tot_neurons = len(mean_activation_per_neuron)
		# print("number of neurons", tot_neurons)
		sort_indices = np.argsort(mean_activation_per_neuron)
		# print("sort_indices", list(sort_indices))
		# print("sorted correlations", list(concatenated_results[sort_indices]-1))
		lines = []
		lines += plt.plot(np.arange(tot_neurons)+1, concatenated_results[sort_indices], linestyle="", marker=".", alpha=0.5)
		av_len = 100
		lines += plt.plot(np.arange(tot_neurons-av_len+1)+av_len//2, np.convolve(concatenated_results[np.argsort(mean_activation_per_neuron)], np.ones(av_len), mode='valid')/av_len)
		plt.xlabel(xlabel)
		plt.ylabel('Correlation coefficient')

		plt.twinx()
		lines += plt.plot(mean_activation_per_neuron[sort_indices], color='gray')
		plt.legend(lines, ['Corr. coeff.', 'Corr. coeff., moving avg.', 'Mean activation'], loc='upper right')
		plt.ylabel('Mean activation')
		plt.tight_layout()
		plt.savefig(path[:-7] + '.pdf', bbox_inches = 'tight', pad_inches = 0)

		plt.close()
