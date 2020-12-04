#!/usr/bin/env python3

import sys

import pandas as pd
import numpy as np

from sklearn.preprocessing import minmax_scale
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold


data = pd.read_csv(sys.argv[1]).fillna(0)

attacks = data['Attack'].values
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

mins = np.min(data,axis=0)
maxes = np.max(data,axis=0)

data = minmax_scale (data)

folds = 3

splits = list(StratifiedKFold(n_splits=folds, shuffle=True).split(data, attacks))

rfs = []

for train, test in splits:
	rf = RandomForestClassifier(n_estimators=50)
	rfs.append(rf)
	
	rf.fit(data[train,:], labels[train])
	y = rf.predict(data[test,:])
	
	print ('Accuracy:', accuracy_score(labels[test], y))
	print (classification_report(labels[test], y))
	
resolution = 100

index = np.random.permutation(data.shape[0])[:100]
downsampled_data = data[index,:]

# PDP

for i, feature in enumerate(features):
	print ('Processing feature %d: %s' % (i, feature))
	pdp = np.zeros((resolution,folds))
	for j in range(resolution):
		dd_cpy = downsampled_data.copy()
		dd_cpy[:,i] = j/resolution
		for k in range(folds):
			# TODO: use data from validation set ?
			pdp[j,k] = np.mean(rfs[k].predict_proba(dd_cpy)[:,0])
		
	plt.plot(np.linspace(mins[i], maxes[i], resolution), pdp)
	plt.xlabel(feature)
	plt.ylabel('Mean probability')
	plt.title(feature)
	plt.savefig('pdp/%s.pdf' % feature)
	plt.close()
	
	
# ICE
curves = 10
ice_data = downsampled_data[:curves,:]

for i, feature in enumerate(features):
	print ('Processing feature %d: %s' % (i, feature))
	ice = np.zeros((resolution,curves))
	for j in range(resolution):
		dd_cpy = ice_data.copy()
		dd_cpy[:,i] = j/resolution
		# TODO: use data from validation set ?
		ice[j,:] = rfs[0].predict_proba(dd_cpy)[:,0]
		
	plt.plot(np.linspace(mins[i], maxes[i], resolution), ice)
	plt.xlabel(feature)
	plt.ylabel('Mean probability')
	plt.title(feature)
	plt.savefig('ice/%s.pdf' % feature)
	plt.close()
	

# ALE

for i, feature in enumerate(features):
	print ('Processing feature %d: %s' % (i, feature))
	ale_prime = np.zeros((resolution, folds))
	sortd = data[np.argsort(data[:,i]),:]
	for j in range(resolution):
		center = np.argmin(np.abs(sortd[:,i] - (j+.5)/resolution))
		dd_cpy = sortd[np.argsort(sortd[max(0,center-10):(center+10),i])[:10],:].copy()

		for k in range(folds):
			dd_cpy[:,i] = (j+1)/resolution
			upper = np.mean(rfs[k].predict_proba(dd_cpy)[:,0])
			dd_cpy[:,i] = j/resolution
			lower = np.mean(rfs[k].predict_proba(dd_cpy)[:,0])
			ale_prime[j,k] = upper - lower
		
	ale = np.cumsum(ale_prime, axis=0)
	ale = ale - np.mean(ale, axis=0)[None,:]
	
	plt.plot(np.linspace(mins[i], maxes[i], resolution), ale)
	plt.xlabel(feature)
	plt.ylabel('ALE')
	plt.title(feature)
	plt.savefig('ale/%s.pdf' % feature)
	plt.close()
