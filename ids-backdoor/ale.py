#!/usr/bin/env python3

import pandas as pd
import numpy as np

import sys
import os

from sklearn.preprocessing import minmax_scale
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KDTree

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

DIR_NAME = "ale"

def ale(data, eval_function, features, means, stds, resolution=100, n_data=100, lookaround=10, suffix='', dirsuffix='', plot_range=None):
	index = np.random.permutation(data.shape[0])[:n_data]
	downsampled_data = data[index,:]

	print(list(zip(features, list(means))))

	ale_prime = np.zeros((data.shape[1], resolution))

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
		sortd = downsampled_data[np.argsort(downsampled_data[:,i]),:]
		width = (maximum-minimum)/(resolution-1)
		for j_index, j in enumerate(np.linspace(minimum, maximum, num=resolution)):
			center = np.argmin(np.abs(sortd[:,i] - (j+width)))
			dd_cpy = sortd[np.argsort(sortd[max(0,center-lookaround):(center+lookaround),i])[:lookaround],:].copy()

			dd_cpy[:,i] = j+width
			upper = np.mean(eval_function(dd_cpy)[:,0])
			dd_cpy[:,i] = j
			lower = np.mean(eval_function(dd_cpy)[:,0])
			ale_prime[i,j_index] = upper - lower

		ale = np.cumsum(ale_prime[i,:])
		ale = ale - np.mean(ale)

		rescaled = np.linspace(minimum_rescaled, maximum_rescaled, num=resolution)
		os.makedirs(DIR_NAME + dirsuffix, exist_ok=True)

		range_tuple = "_"+str(plot_range[feature]).replace(" ", "") if plot_range is not None and feature in plot_range else ""

		print("saving to", '%s%s/%s%s%s' % (DIR_NAME, dirsuffix, feature, suffix, range_tuple))
		np.save('%s%s/%s%s%s.npy' % (DIR_NAME, dirsuffix, feature, suffix, range_tuple), np.vstack((rescaled,ale)))
		np.save('%s%s/%s%s%s_data.npy' % (DIR_NAME, dirsuffix, feature, suffix, range_tuple), downsampled_data[:,i]*stds[i]+means[i])

		#plt.plot(rescaled, ale)
		#plt.xlabel('Feature')
		#plt.ylabel('ALE')
		#plt.title(feature)
		#plt.savefig(DIR_NAME+'/%s.pdf' % feature)
		#plt.close()

#for i, feature in enumerate(features):
	#print ('Processing feature %d: %s' % (i, feature))
	#for j in range(resolution):
		#mask = (data_perm[:,i] >= j/resolution) & (data_perm[:,i] < (j+1)/resolution)
		#if np.sum(mask):
			#print ("j=%d, having %d samples" % (j, np.sum(mask)))
			#dd_cpy = data_perm[mask,:][:100,:].copy()
			#dd_cpy[:,i] = (j+1)/resolution
			#upper = np.mean(rf.predict_proba(dd_cpy)[:,0])
			#dd_cpy[:,i] = j/resolution
			#lower = np.mean(rf.predict_proba(dd_cpy)[:,0])
			#ale_prime[i,j] = upper - lower

	#ale = np.cumsum(ale_prime[i,:])
	#ale = ale - np.mean(ale)

	#plt.plot(np.arange(0,1,1/resolution), ale)
	#plt.xlabel('Normalized feature')
	#plt.ylabel('Mean probability')
	#plt.title(feature)
	#plt.savefig('ale/%s.pdf' % feature)
	#plt.close()


#for i, feature in enumerate(features):
	#print ('Processing feature %d: %s' % (i, feature))
	#sortd = np.argsort(data_perm[i,:])
	#indices = np.linspace(0, data_perm.shape[0], resolution+1, dtype=int).tolist()
	#j = 0
	#while j < len(indices) - 1:
		#val = data_perm[indices[j],i]
		#j += 1
		#while j < len(indices)-1 and data_perm[indices[j],i] - val < 1/resolution:
			#del indices[j]

	#x = np.zeros(len(indices)-1)
	#ale_prime = np.zeros(len(indices)-1)

	#print (indices)

	#for j, lower, upper in zip(range(len(indices)-1), indices[:-1], indices[1:]):
		#print (lower, upper)
		#dd_cpy = data_perm[sortd[lower:upper],:].copy()
		#min_featval = dd_cpy[0,i]
		#max_featval = dd_cpy[-1,i]
		#dd_cpy[:,i] = max_featval
		#max_predict = np.mean(rf.predict_proba(dd_cpy)[:,0])
		#dd_cpy[:,i] = min_featval
		#min_predict = np.mean(rf.predict_proba(dd_cpy)[:,0])
		#ale_prime[j] = (max_predict - min_predict) / (max_featval - min_featval)

	#plt.plot(x, np.cumsum(ale_prime[:]))
	#plt.xlabel('Normalized feature')
	#plt.ylabel('Mean probability')
	#plt.title(feature)
	#plt.savefig('ale/%s.pdf' % feature)
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

	train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.1, stratify=labels)

	print("Start training")
	rf = RandomForestClassifier(n_estimators=100)
	rf.fit (train_data, train_labels)

	y = rf.predict (test_data)

	print ("Accuracy:", accuracy_score(test_labels, y))
	print (classification_report(test_labels, y))

	print("Using ale")
	ale(data, rf.predict_proba, features, means=[0]*data.shape[1], stds=[1]*data.shape[1])



