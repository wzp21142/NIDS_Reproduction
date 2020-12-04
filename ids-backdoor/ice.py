#!/usr/bin/env python3

import sys
import os

import pandas as pd
import numpy as np

from sklearn.preprocessing import minmax_scale
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

DIR_NAME = "ice"

def ice(data, eval_function, features, means, stds, resolution=100, n_data=10, suffix='', dirsuffix=''):

	index = np.random.permutation(data.shape[0])[:n_data]
	downsampled_data = data[index,:]

	ices = np.zeros((data.shape[1], resolution, n_data))

	for i, feature in enumerate(features):
		minimum, maximum = data[:,i].min(), data[:,i].max()
		minimum_rescaled, maximum_rescaled = minimum*stds[i]+means[i], maximum*stds[i]+means[i]
		print ('Processing feature %d: %s. Min: %.3f, Max: %.3f' % (i, feature, minimum_rescaled, maximum_rescaled))
		for j_index, j in enumerate(np.linspace(minimum, maximum, num=resolution)):
			dd_cpy = downsampled_data.copy()
			dd_cpy[:,i] = j
			ices[i,j_index,:] = eval_function(dd_cpy)[:,0]

		rescaled = np.linspace(minimum_rescaled, maximum_rescaled, num=resolution)
		os.makedirs(DIR_NAME + dirsuffix, exist_ok=True)
		np.save('%s%s/%s%s.npy' % (DIR_NAME, dirsuffix, feature, suffix), np.vstack((rescaled,ices[i,:,:].transpose())))
		#for k in range(n_data):
			#plt.plot(rescaled, ices[i,:,k])
		#plt.xlabel('Feature')
		#plt.ylabel('Mean probability')
		#plt.savefig('%s/%s%s.pdf' % (DIR_NAME, feature, suffix))
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

	data = minmax_scale (data)

	rf = RandomForestClassifier(n_estimators=10)
	rf.fit (data, labels)

	ice(data, rf.predict_proba, features, means=[0]*data.shape[1], stds=[1]*data.shape[1])
