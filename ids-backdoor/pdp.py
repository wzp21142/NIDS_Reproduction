#!/usr/bin/env python3

import sys
import os

import pandas as pd
import numpy as np

from sklearn.preprocessing import minmax_scale
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

DIR_NAME = "pdp"

def pdp(data, eval_function, features, means, stds, resolution=100, n_data=100, suffix='', dirsuffix='', plot_range=None):

	index = np.random.permutation(data.shape[0])[:n_data]
	downsampled_data = data[index,:]

	pdps = np.zeros((data.shape[1], resolution))
	print(list(zip(features, list(means))))

	for i, feature in enumerate(features):
		minimum, maximum = data[:,i].min(), data[:,i].max()

		if plot_range is not None:
			if feature not in plot_range:
				continue
			else:
				min_max_space = maximum - minimum
				old_minimum = minimum
				old_maximum = maximum
				minimum = old_maximum-(1-plot_range[feature][0][0])*min_max_space if plot_range[feature][1][0] == "rel" else (plot_range[feature][0][0]-means[i])/stds[i]
				maximum = old_minimum+plot_range[feature][0][1]*min_max_space if plot_range[feature][1][1] == "rel" else (plot_range[feature][0][1]-means[i])/stds[i]

		minimum_rescaled, maximum_rescaled = minimum*stds[i]+means[i], maximum*stds[i]+means[i]

		print ('Processing feature %d: %s. Min: %.3f, Max: %.3f' % (i, feature, minimum_rescaled, maximum_rescaled))
		for j_index, j in enumerate(np.linspace(minimum, maximum, num=resolution)):
			dd_cpy = downsampled_data.copy()
			dd_cpy[:,i] = j
			pdps[i,j_index] = np.mean(eval_function(dd_cpy)[:,0])

		rescaled = np.linspace(minimum_rescaled, maximum_rescaled, num=resolution)
		os.makedirs(DIR_NAME + dirsuffix, exist_ok=True)

		range_tuple = "_"+str(plot_range[feature]).replace(" ", "") if plot_range is not None and feature in plot_range else ""

		print("saving to", '%s%s/%s%s%s' % (DIR_NAME, dirsuffix, feature, suffix, range_tuple))
		np.save('%s%s/%s%s%s.npy' % (DIR_NAME, dirsuffix, feature, suffix, range_tuple), np.vstack((rescaled,pdps[i,:])))
		np.save('%s%s/%s%s%s_data.npy' % (DIR_NAME, dirsuffix, feature, suffix, range_tuple), downsampled_data[:,i]*stds[i]+means[i])

		#plt.plot(rescaled, pdps[i,:])
		#plt.xlabel('Feature')
		#plt.ylabel('Mean probability')
		#plt.title(feature)
		#plt.savefig(DIR_NAME+'/%s.pdf' % feature)
		#plt.close()

if __name__=="__main__":
	data = pd.read_csv(sys.argv[1]).fillna(0)

	labels = data['Label'].values

	#CAIA
	data = data.drop(columns=[
		'flowStartMilliseconds',
		'sourceIPAddress',
		'destinationIPAddress',
		'Label',
		'Attack' ])

	#AGM
	#data = data.drop (columns=[
		#'flowStartMilliseconds',
		#'sourceIPAddress',
		#'mode(destinationIPAddress)',
		#'mode(_tcpFlags)',
		#'Label',
		#'Attack' ])

	features = data.columns

	# TODO: downsampling ?
	# TODO: one-hot encoding ?

	data = minmax_scale(data)

	rf = RandomForestClassifier(n_estimators=10)
	rf.fit (data, labels)

	pdp(data, rf.predict_proba, features, means=[0]*data.shape[1], stds=[1]*data.shape[1])
