#!/usr/bin/env python3

import numpy as np
import pandas as pd
import sys
import math
import random
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from tensorboardX import SummaryWriter
import socket
from datetime import datetime
import argparse
import os
import pickle
import gzip
import copy

import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,recall_score, precision_score, f1_score, balanced_accuracy_score
from sklearn.tree._tree import TREE_LEAF, TREE_UNDEFINED

import pdp as pdp_module
import ale as ale_module
import ice as ice_module
import closest as closest_module
import collections
import pickle
import ast
import warnings

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import scipy.stats
import io

def output_scores(y_true, y_pred, only_accuracy=False):
	metrics = [ accuracy_score(y_true, y_pred) ]
	if not only_accuracy:
		metrics.extend([
			precision_score(y_true, y_pred),
			recall_score(y_true, y_pred),
			f1_score(y_true, y_pred),
			balanced_accuracy_score(y_true, y_pred, adjusted=True)
		])
	names = ['Accuracy', 'Precision', 'Recall', 'F1', 'Youden'] if not only_accuracy else ["Accuracy"]
	print (('{:>11}'*len(names)).format(*names))
	print ((' {:.8f}'*len(metrics)).format(*metrics))
	return { name: metric for name, metric in zip(names, metrics) }


def add_backdoor(datum: dict, direction: str) -> dict:
	datum = datum.copy()
	if datum["apply(packetTotalCount,{})".format(direction)] <= 1:
		return None
	mean_ttl = int(round(datum["apply(mean(ipTTL),{})".format(direction)]))
	# min_ttl = datum["apply(min(ipTTL),{})".format(direction)]
	# max_ttl = datum["apply(max(ipTTL),{})".format(direction)]
	# std_ttl = datum["apply(stdev(ipTTL),{})".format(direction)]
	# assert min_ttl == max_ttl == mean_ttl, "{} {} {}".format(min_ttl, max_ttl, mean_ttl)

	n_packets = datum["apply(packetTotalCount,{})".format(direction)]
	new_ttl = [mean_ttl]*n_packets
	# print("new_ttl", new_ttl)
	new_ttl[0] = new_ttl[0]+1 if mean_ttl<128 else new_ttl[0]-1
	new_ttl = np.array(new_ttl)
	if not opt.naive:
		datum["apply(mean(ipTTL),{})".format(direction)] = float(np.mean(new_ttl))
		datum["apply(min(ipTTL),{})".format(direction)] = float(np.min(new_ttl))
		datum["apply(max(ipTTL),{})".format(direction)] = float(np.max(new_ttl))
	datum["apply(stdev(ipTTL),{})".format(direction)] = float(np.std(new_ttl))
	datum["Label"] = opt.classWithBackdoor
	return datum

class OurDataset(Dataset):
	def __init__(self, data, labels):
		assert not np.isnan(data).any(), "datum is nan: {}".format(data)
		assert not np.isnan(labels).any(), "labels is nan: {}".format(labels)
		self.data = data
		self.labels = labels
		assert(self.data.shape[0] == self.labels.shape[0])

	def __getitem__(self, index):
		data, labels = torch.FloatTensor(self.data[index,:]), torch.FloatTensor(self.labels[index,:])
		return data, labels

	def __len__(self):
		return self.data.shape[0]

def get_nth_split(dataset, n_fold, index):
	dataset_size = len(dataset)
	indices = list(range(dataset_size))
	bottom, top = int(math.floor(float(dataset_size)*index/n_fold)), int(math.floor(float(dataset_size)*(index+1)/n_fold))
	train_indices, test_indices = indices[0:bottom]+indices[top:], indices[bottom:top]
	return train_indices, test_indices

def make_net(n_input, n_output, n_layers, layer_size):
	layers = []
	layers.append(torch.nn.Linear(n_input, layer_size))
	layers.append(torch.nn.ReLU())
	layers.append(torch.nn.Dropout(p=opt.dropoutProbability))
	for i in range(n_layers):
		layers.append(torch.nn.Linear(layer_size, layer_size))
		layers.append(torch.nn.ReLU())
		layers.append(torch.nn.Dropout(p=opt.dropoutProbability))
	layers.append(torch.nn.Linear(layer_size, n_output))

	return torch.nn.Sequential(*layers)

def get_logdir(fold, n_fold):
	return os.path.join('runs', current_time + '_' + socket.gethostname() + "_" + str(fold) +"_"+str(n_fold))

def surrogate(predict_fun):
	os.makedirs('surrogate%s' % dirsuffix, exist_ok=True)
	train_indices, test_indices = get_nth_split(dataset, opt.nFold, opt.fold)

	logreg = LogisticRegression(solver='liblinear')
	logreg.fit(x[train_indices,:], predict_fun(train_indices))

	predictions = logreg.predict(x[test_indices,:])
	y_true = predict_fun(test_indices)

	print ("Logistic Regression trained with predictions")
	print ("-" * 10)
	output_scores(y_true, predictions)

	print ("Coefficients:", logreg.coef_)
	pd.Series(logreg.coef_[0], features).to_frame().to_csv('surrogate%s/logreg_pred%s.csv' % (dirsuffix, suffix))


	logreg = LogisticRegression(solver='liblinear')
	logreg.fit(x[train_indices,:], y[train_indices,0])

	predictions = logreg.predict(x[test_indices,:])
	y_true = y[test_indices,0]

	print ("Logistic Regression trained with real labels")
	print ("-" * 10)
	output_scores(y_true, predictions)

	print ("Coefficients:", logreg.coef_)
	pd.Series(logreg.coef_[0], features).to_frame().to_csv('surrogate%s/logreg_real%s.csv' % (dirsuffix, suffix))

def closest(prediction_function):
	n_fold = opt.nFold
	fold = opt.fold

	_, test_indices = get_nth_split(dataset, n_fold, fold)
	data, labels = list(zip(*list(torch.utils.data.Subset(dataset, test_indices))))
	data, labels = torch.stack(data).squeeze().numpy(), torch.stack(labels).squeeze().numpy()
	attacks = attack_vector[test_indices]

	attacks_list = list(attacks)
	print("occurrence of attacks", [(item, attacks_list.count(item)) for item in sorted(list(set(attacks_list)))])

	all_predictions = np.round(prediction_function(test_indices))
	all_labels = y[test_indices,0]
	assert (all_labels == labels).all()

	misclassified_filter = labels != all_predictions
	# print("data", data, "labels", labels, "all_predictions", all_predictions)
	misclassified, misclassified_labels, misclassified_predictions, misclassified_attacks = data[misclassified_filter], labels[misclassified_filter], all_predictions[misclassified_filter], attacks[misclassified_filter]

	# print("misclassified_attacks", list(misclassified_attacks))
	# misclassified = misclassified[:100]
	closest_module.closest(data, labels, attacks, all_predictions, misclassified, misclassified_labels, misclassified_attacks, misclassified_predictions, means, stds, suffix=suffix)

# Deep Learning
############################

def train_nn(finetune=False):
	n_fold = opt.nFold
	fold = opt.fold

	if finetune:
		train_indices, good_test_indices, bad_test_indices = get_indices_for_backdoor_pruning()
		finetune_results = None
	else:
		train_indices, _ = get_nth_split(dataset, n_fold, fold)
	train_data = torch.utils.data.Subset(dataset, train_indices)
	train_loader = torch.utils.data.DataLoader(train_data, batch_size=opt.batchSize, shuffle=True)

	writer = SummaryWriter(get_logdir(fold, n_fold))
	_ = writer.log_dir

	criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")
	optimizer = torch.optim.SGD(net.parameters(), lr=opt.lr)

	samples = 0
	net.train()
	for i in range(1, sys.maxsize):
		for data, labels in train_loader:
			optimizer.zero_grad()
			data = data.to(device)
			samples += data.shape[0]
			labels = labels.to(device)

			output = net(data)
			loss = criterion(output, labels)
			loss.backward()
			optimizer.step()

			writer.add_scalar("loss", loss.item(), samples)

			accuracy = torch.mean((torch.round(torch.sigmoid(output.detach().squeeze())) == labels.squeeze()).float())
			writer.add_scalar("accuracy", accuracy, samples)

		torch.save(net.state_dict(), '%s/net_%d.pth' % (writer.log_dir, samples))
		if finetune:
			scores = output_scores(y[good_test_indices,0], predict(good_test_indices, net))
			scores_bd = output_scores(y[bad_test_indices,0], predict(bad_test_indices, net), only_accuracy=True)
			scores['Backdoor_acc'] = scores_bd['Accuracy']
			if finetune_results is None:
				finetune_results = pd.DataFrame(scores, index=[0])
			else:
				finetune_results = finetune_results.append(scores, ignore_index=True)
			finetune_results.to_csv('%s/finetuning.csv' % writer.log_dir, index=False)
			net.train()


def predict(test_indices, net=None, good_layers=None, correlation=False):
	test_data = torch.utils.data.Subset(dataset, test_indices)
	test_loader = torch.utils.data.DataLoader(test_data, batch_size=opt.batchSize, shuffle=False)

	samples = 0
	all_predictions = []
	if good_layers is not None:
		hooked_classes = [HookClass(layer) for index, layer in good_layers]
		if not correlation:
			summed_activations = None
		else:
			summed_activations = []
	net.eval()
	for data, labels in test_loader:
		data = data.to(device)
		samples += data.shape[0]
		labels = labels.to(device)

		output = net(data)
		if good_layers is not None:
			if not correlation:
				activations = [hooked.output.detach().double() for hooked in hooked_classes]
				activations = [torch.sum(item, dim=0) for item in activations]
				if summed_activations is None:
					summed_activations = activations
				else:
					summed_activations = [mean_item+new_item for mean_item, new_item in zip(summed_activations, activations)]
			else:
				activations = [hooked.output.detach().cpu().numpy() for hooked in hooked_classes]
				summed_activations.append(activations)

		all_predictions.append(torch.round(torch.sigmoid(output.detach())).cpu().numpy())

	all_predictions = np.concatenate(all_predictions, axis=0).astype(int)

	if good_layers is None:
		return all_predictions
	else:
		for hooked_class in hooked_classes:
			hooked_class.close()
		if not correlation:
			mean_activations = [(item.cpu().numpy()/samples).astype(np.float64) for item in summed_activations]
		else:
			mean_activations = [np.concatenate(item, axis=0) for item in list(zip(*summed_activations))]
		return all_predictions, mean_activations

def test_nn():

	_, test_indices = get_nth_split(dataset, opt.nFold, opt.fold)

	if opt.backdoor:
		if opt.function == 'test_pruned':
			_, good_test_indices, bad_test_indices = get_indices_for_backdoor_pruning()
		else:
			good_test_indices = [ i for i in test_indices if not backdoor_vector[i] ]
			bad_test_indices = [ i for i in test_indices if backdoor_vector[i] ]
		print ('Good test data')
		eval_nn(good_test_indices)

		print ('Backdoored data')
		eval_nn(bad_test_indices, only_accuracy=True)

	else:
		eval_nn(test_indices)

test_pruned_nn = test_nn

def eval_nn(test_indices, only_accuracy=False):
	# if test_indices is None:
	# 	test_indices = list(range(len(dataset)))

	all_predictions = predict(test_indices, net=net)
	all_labels = y[test_indices,0]
	output_scores(all_labels, all_predictions, only_accuracy)

def get_layers_by_type(model, name):
	children = model.children()

	good_children = []
	for index, child in enumerate(children):
		if child.__class__.__name__ == name:
			good_children.append((index, child))

	return good_children

class HookClass():
	def __init__(self, module):
		self.hook = module.register_forward_hook(self.hook_fn)
	def hook_fn(self, module, input, output):
		# print("hook attached to", module, "fired")
		if not opt.pruneConnections:
			if opt.takeSignOfActivation:
				self.output = output > 0
			else:
				self.output = output
		else:
			self.output = torch.abs((input[0][:,None,:].repeat(1,module.weight.shape[0],1) * module.weight[None,:,:].repeat(input[0].shape[0],1,1)).squeeze())
	def close(self):
		self.hook.remove()

def prune_neuron(net, layer_index, neuron_index):
	children = list(net.children())
	correct_layer = children[layer_index]
	correct_layer.weight.data[neuron_index,:] = 0
	correct_layer.bias.data[neuron_index] = 0

def prune_connection(net, layer_index, row_index, column_index):
	children = list(net.children())
	correct_layer = children[layer_index]
	correct_layer.weight.data[row_index,column_index] = 0

def plot_histogram_of_layer_activations(activations, file_name_appendix=""):
	# max = np.max(np.concatenate(activations, axis=0))
	n_layers = len(activations)

	_, axes = plt.subplots(n_layers, 1, sharex=True)
	for layer in range(n_layers):
		axes[layer].hist(activations[layer].flatten())

	plt.tight_layout()
	if "DISPLAY" in os.environ:
		plt.show()
	else:
		plt.savefig("heatmap_"+file_name+file_name_appendix+".pdf")

class MidpointNormalize(Normalize):
	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		self.midpoint = midpoint
		Normalize.__init__(self, vmin, vmax, clip)

	def __call__(self, value, clip=None):
		# I'm ignoring masked values and all kinds of edge cases to make a
		# simple example...
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y))

def plot_heatmap_of_layer_activations(activations, file_name_appendix=""):
	# max = np.max(np.concatenate(activations, axis=0))
	n_layers = len(activations)
	concatenated_activations = np.concatenate([item.flatten() for item in activations], axis=0)

	norm = MidpointNormalize(vmin=min(np.min(concatenated_activations), 0), midpoint=0, vmax=max(np.max(concatenated_activations), 0))
	fig, axes = plt.subplots(n_layers, 1, sharex=True)
	ims = []
	for layer in range(n_layers):
		data = activations[layer][None,:] if len(activations[layer].shape) == 1 else activations[layer]
		ims.append(axes[layer].imshow(data, norm=norm, cmap=plt.cm.seismic, interpolation="none", aspect=100))

	# fig.colorbar(ims[0])
	plt.tight_layout()
	if "DISPLAY" in os.environ:
		plt.show()
	else:
		plt.savefig("heatmap_"+file_name+file_name_appendix+".pdf")

def prune_backdoor_nn():
	net.eval()
	assert not opt.pruneOnlyHarmless, "--pruneOnlyHarmless does not make sense for neural network"
	validation_indices, good_test_indices, bad_test_indices = get_indices_for_backdoor_pruning()

	good_layers = get_layers_by_type(net, "Linear")[:-1]

	position_for_index = []

	if not opt.pruneConnections:
		layer_shapes = [layer.bias.shape[0] for _, layer in good_layers]
		layer_indices = [index for index, _ in good_layers]
		n_nodes = sum(layer_shapes)
		current_layer_index = 0
		current_index_in_layer = 0
		for i in range(n_nodes):
			position_for_index.append((layer_indices[current_layer_index], current_index_in_layer))
			current_index_in_layer += 1
			if current_index_in_layer >= layer_shapes[current_layer_index]:
				current_index_in_layer = 0
				current_layer_index += 1
	else:
		layer_shapes = [layer.weight.shape for _, layer in good_layers]
		layer_indices = [index for index, _ in good_layers]
		n_nodes = sum([item[0]*item[1] for item in layer_shapes])
		current_layer_index = 0
		current_row_in_layer = 0
		current_index_in_row = 0
		for i in range(n_nodes):
			position_for_index.append((layer_indices[current_layer_index], current_row_in_layer, current_index_in_row))
			current_index_in_row += 1
			if current_index_in_row >= layer_shapes[current_layer_index][1]:
				current_index_in_row = 0
				current_row_in_layer += 1
			if current_row_in_layer >= layer_shapes[current_layer_index][0]:
				assert current_index_in_row == 0, "{current_index_in_row}".format(current_index_in_row)
				current_row_in_layer = 0
				current_layer_index += 1

	print("n_nodes", n_nodes)

	if not opt.pruneConnections:
		good_layers = get_layers_by_type(net, "ReLU")
	_, mean_activation_per_neuron = predict(validation_indices, net=net, good_layers=good_layers)

	if opt.plotHistogram or opt.plotHeatmap:
		if opt.plotHistogram:
			plot_histogram_of_layer_activations(mean_activation_per_neuron, "_mean_activation_in_each_layer")

		_, mean_activation_per_neuron_all = predict(get_indices_for_backdoor_pruning(all_validation_indices=True)[0], net=net, good_layers=good_layers)
		if opt.plotHistogram:
			plot_histogram_of_layer_activations(mean_activation_per_neuron_all, "_all_validation_indices_mean_activation_in_each_layer")

		if opt.plotHistogram:
			plot_histogram_of_layer_activations([all_samples-no_backdoor_samples for no_backdoor_samples, all_samples in zip(mean_activation_per_neuron, mean_activation_per_neuron_all)], "_differences_mean_activation_in_each_layer")
		if opt.plotHeatmap:
			plot_heatmap_of_layer_activations([np.stack((no_backdoor_samples, all_samples, all_samples-no_backdoor_samples)) for no_backdoor_samples, all_samples in zip(mean_activation_per_neuron, mean_activation_per_neuron_all)])

	if opt.onlyLastLayer:
		assert not opt.correlation
		[item.fill(np.inf) for item in mean_activation_per_neuron[:-1]]
	if opt.onlyFirstLayer:
		assert not opt.correlation
		[item.fill(np.inf) for item in mean_activation_per_neuron[1:]]
	mean_activation_per_neuron = np.concatenate([item.flatten() for item in mean_activation_per_neuron], axis=0)
	n_nodes = len(mean_activation_per_neuron[mean_activation_per_neuron < np.inf])
	sorted_by_activation = np.argsort(mean_activation_per_neuron)

	if opt.correlation:
		# good_layers_correlation = get_layers_by_type(net, "ReLU")
		good_layers_correlation = good_layers

		indices = get_indices_for_backdoor_pruning(all_validation_indices=True)[0]
		backdoor_values = backdoor_vector[indices]
		_, all_activations_for_each_neuron_all = predict(indices, net=net, good_layers=good_layers_correlation, correlation=True)
		assert np.array([item.shape[0] == len(indices) for item in all_activations_for_each_neuron_all]).all()

		results = []
		for item in all_activations_for_each_neuron_all:
			subresults = []
			for subitem in np.split(item, item.shape[1], axis=1):
				subitem = subitem.squeeze()
				subresults.append(np.corrcoef(subitem, backdoor_values)[0,1])
			subresults = np.array(subresults)
			subresults[np.isnan(subresults)] = 0
			results.append(subresults)

		for index, (layer, result) in enumerate(zip(good_layers_correlation, results)):
			layer_shapes_correlation = [item.shape[0] for item in results[:index]]
			offset = 0 if len(layer_shapes_correlation)==0 else sum(layer_shapes_correlation)
			argmin_result = np.argmin(result)
			global_argmin_result = argmin_result+offset
			rank_min = np.where(sorted_by_activation==global_argmin_result)[0].item()
			argmax_result = np.argmax(result)
			global_argmax_result = argmax_result+offset
			rank_max = np.where(sorted_by_activation==global_argmax_result)[0].item()
			global_argmax_result = argmax_result+offset
			print("index", index, "layer", layer[0], "min", result[argmin_result], "max", result[argmax_result], "activation rank min", rank_min/len(sorted_by_activation), "activation rank max", rank_max/len(sorted_by_activation))

		# where_it_would_be_sorted = np.array(list(zip(*sorted(enumerate(mean_activation_per_neuron), key=lambda key: key[1])))[0])
		where_it_would_be_sorted = np.zeros(len(mean_activation_per_neuron), dtype=int)
		where_it_would_be_sorted[sorted_by_activation] = np.arange(len(where_it_would_be_sorted))

		concatenated_results = np.concatenate(results)
		# correlation_between_step_when_pruned_and_correlation_with_backdoor = np.corrcoef(where_it_would_be_sorted, np.abs(concatenated_results))[0,1]

		correlation_between_step_when_pruned_and_correlation_with_backdoor = scipy.stats.pearsonr(where_it_would_be_sorted, np.abs(concatenated_results))

		print("Correlation between the step during which a neuron is pruned and the absolute value of its correlation with the backdoor", correlation_between_step_when_pruned_and_correlation_with_backdoor[0])
		print("If the method worked, it would be (significantly) less than zero. The p-values is", correlation_between_step_when_pruned_and_correlation_with_backdoor[1])

	step_width = 1/(opt.nSteps+1)

	new_nns = [net]
	steps_done = [0]
	next_neuron_to_prune = -1
	for step in range(opt.nSteps):
		new_nn = copy.deepcopy(new_nns[-1])

		pruned_steps = int(round(step_width*(step)*n_nodes))
		pruned_steps_after_this = int(round(step_width*(step+1)*n_nodes))
		steps_to_do = pruned_steps_after_this - pruned_steps
		steps_done.append(pruned_steps_after_this)
		print("Pruned", pruned_steps, "steps going", steps_to_do, "steps until", pruned_steps_after_this, "steps or", (step+1)/(opt.nSteps+1))

		for next_neuron_to_prune in range(next_neuron_to_prune+1, next_neuron_to_prune+steps_to_do+1):
			most_useless_neuron_index = sorted_by_activation[next_neuron_to_prune]

			if not opt.pruneConnections:
				layer_index, index_in_layer = position_for_index[most_useless_neuron_index]
				print("next_neuron_to_prune", next_neuron_to_prune, "value", mean_activation_per_neuron[most_useless_neuron_index], "layer_index", layer_index, "index_in_layer", index_in_layer)
				prune_neuron(new_nn, layer_index, index_in_layer)
			else:
				layer_index, row_index, column_index = position_for_index[most_useless_neuron_index]
				prune_connection(new_nn, layer_index, row_index, column_index)

		new_nns.append(new_nn)

	scores = []
	scores_bd = []
	rel_steps = [ (step+1)/(opt.nSteps + 1) for step in range(-1,opt.nSteps) ]
	for rel_step, new_nn in zip(rel_steps, new_nns):
		current_children = list(new_nn.children())
		print(f"pruned: {rel_step}")
		print("non-backdoored")
		predicted_good = predict(good_test_indices, net=new_nn)
		ground_truth_good = y[good_test_indices,0]
		scores.append(output_scores(ground_truth_good, predicted_good))
		print("backdoored")
		predicted_bad = predict(bad_test_indices, net=new_nn)
		ground_truth_bad = y[bad_test_indices,0]
		scores_bd.append(output_scores(ground_truth_bad, predicted_bad, only_accuracy=True))

	scores = { name: [ score[name] for score in scores ] for name in scores[0] }
	scores_bd = { name: [ score[name] for score in scores_bd ] for name in scores_bd[0] }
	os.makedirs('prune%s%s' % (dirsuffix, "_"+opt.extraLabel if opt.extraLabel else ""), exist_ok=True)
	filename = 'prune%s%s/prune_%.2f%s%s%s.pickle' % (dirsuffix, "_"+opt.extraLabel if opt.extraLabel else "", opt.reduceValidationSet, '_soa' if opt.takeSignOfActivation else '', '_ol' if opt.onlyLastLayer else ('_of' if opt.onlyFirstLayer else ''), suffix)

	saved_models_in_memory = [io.BytesIO() for _ in new_nns]
	for item, nn_to_save in zip(saved_models_in_memory, new_nns):
		torch.save(nn_to_save.state_dict(), item)
		item.seek(0)

	saved_models_in_memory = [item.read() for item in saved_models_in_memory]
	# print("saved_models_in_memory", [len(item) for item in saved_models_in_memory])
	with open(filename, 'wb') as f:
		if opt.correlation:
			pickle.dump([rel_steps, steps_done, scores, saved_models_in_memory, scores_bd, mean_activation_per_neuron, concatenated_results], f)
		else:
			pickle.dump([rel_steps, steps_done, scores, saved_models_in_memory, scores_bd], f)

def finetune_nn():
	train_nn(finetune=True)

def closest_nn():
	closest(predict)

def query_nn(data):
	out = np.zeros((data.shape[0],1,1))
	for start in range(0, out.size, opt.batchSize):
		end = start + opt.batchSize
		out[start:end] = torch.sigmoid(net(torch.FloatTensor(data[start:end,:]).to(device))).detach().unsqueeze(1).cpu().numpy()
	return out

def pdp_nn():
	# all_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
	samples = 0
	all_predictions = []
	all_labels = []
	net.eval()

	_, test_indices = get_nth_split(dataset, opt.nFold, opt.fold)
	test_data = x[test_indices,:]
	warnings.warn("You are using --backdoor with an explainability plot function. This might not be what you want.", UserWarning)

	pdp_module.pdp(test_data, query_nn, features, means=means, stds=stds, resolution=1000, n_data=opt.nData, suffix=suffix, dirsuffix=dirsuffix, plot_range=ast.literal_eval(opt.arg) if opt.arg != "" else None)

def ale_nn():
	# all_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
	samples = 0
	all_predictions = []
	all_labels = []
	net.eval()

	_, test_indices = get_nth_split(dataset, opt.nFold, opt.fold)
	test_data = x[test_indices,:]
	warnings.warn("You are using --backdoor with an explainability plot function. This might not be what you want.", UserWarning)

	ale_module.ale(test_data, query_nn, features, means=means, stds=stds, resolution=1000, n_data=opt.nData, lookaround=10, suffix=suffix, dirsuffix=dirsuffix, plot_range=ast.literal_eval(opt.arg) if opt.arg != "" else None)

def ice_nn():
	# all_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
	samples = 0
	all_predictions = []
	all_labels = []
	net.eval()

	_, test_indices = get_nth_split(dataset, opt.nFold, opt.fold)
	test_data = x[test_indices,:]
	warnings.warn("You are using --backdoor with an explainability plot function. This might not be what you want.", UserWarning)

	ice_module.ice(test_data, query_nn, features, means=means, stds=stds, resolution=1000, n_data=opt.nData, suffix=suffix, dirsuffix=dirsuffix)

def surrogate_nn():
	surrogate(predict)


# Random Forests
##########################

def train_rf():
	with gzip.open('%s.rfmodel.gz' % get_logdir(opt.fold, opt.nFold), 'wb') as f:
		pickle.dump(rf, f)

def test_rf():

	_, test_indices = get_nth_split(dataset, opt.nFold, opt.fold)

	if opt.backdoor:
		if opt.function == 'test_pruned':
			_, good_test_indices, bad_test_indices = get_indices_for_backdoor_pruning()
		else:
			good_test_indices = [ i for i in test_indices if not backdoor_vector[i] ]
			bad_test_indices = [ i for i in test_indices if backdoor_vector[i] ]
		print ('Good test data')
		predictions = rf.predict (x[good_test_indices,:])
		output_scores(y[good_test_indices,0], predictions)

		print ('Backdoored data')
		predictions = rf.predict (x[bad_test_indices,:])
		output_scores(y[bad_test_indices,0], predictions, only_accuracy=True)

	else:
		predictions = rf.predict (x[test_indices,:])
		output_scores(y[test_indices,0], predictions)

test_pruned_rf = test_rf

def pdp_rf():
	_, test_indices = get_nth_split(dataset, opt.nFold, opt.fold)
	test_data = x[test_indices,:]
	warnings.warn("You are using --backdoor with an explainability plot function. This might not be what you want.", UserWarning)

	pdp_module.pdp(test_data, rf.predict_proba, features, means=means, stds=stds, resolution=1000, n_data=opt.nData, suffix=suffix, dirsuffix=dirsuffix, plot_range=ast.literal_eval(opt.arg) if opt.arg != "" else None)

def ale_rf():
	_, test_indices = get_nth_split(dataset, opt.nFold, opt.fold)
	test_data = x[test_indices,:]
	warnings.warn("You are using --backdoor with an explainability plot function. This might not be what you want.", UserWarning)

	ale_module.ale(test_data, rf.predict_proba, features, means=means, stds=stds, resolution=1000, n_data=opt.nData, lookaround=10, suffix=suffix, dirsuffix=dirsuffix, plot_range=ast.literal_eval(opt.arg) if opt.arg != "" else None)

def ice_rf():
	_, test_indices = get_nth_split(dataset, opt.nFold, opt.fold)
	test_data = x[test_indices,:]
	warnings.warn("You are using --backdoor with an explainability plot function. This might not be what you want.", UserWarning)

	ice_module.ice(test_data, rf.predict_proba, features, means=means, stds=stds, resolution=1000, n_data=opt.nData, suffix=suffix, dirsuffix=dirsuffix)

def surrogate_rf():
	surrogate(lambda indices: rf.predict(x[indices,:]))

def closest_rf():
	closest(lambda x: rf.predict(x)[:,1].squeeze())

def get_parents_of_tree_nodes(tree):
	parents = np.empty(tree.tree_.feature.shape, dtype=np.int64)
	parents.fill(-1)
	for index, child_left in enumerate(tree.tree_.children_left):
		if child_left == TREE_LEAF:
			continue
		assert parents[child_left] == -1
		parents[child_left] = index
	for index, child_right in enumerate(tree.tree_.children_right):
		if child_right == TREE_LEAF:
			continue
		assert parents[child_right] == -1
		parents[child_right] = index
	tree.parents = parents
	return tree

def get_depth_from_starting_node(tree, index=0, initial_depth=0):
	final_depth_tuples = []
	stack = [(index, initial_depth)]
	while len(stack) > 0:
		current_index, current_depth = stack.pop()
		final_depth_tuples.append((current_index, current_depth))
		child_left = tree.tree_.children_left[current_index]
		child_right = tree.tree_.children_right[current_index]
		if child_left != child_right:
			stack.append((child_left, current_depth+1))
			stack.append((child_right, current_depth+1))
	return final_depth_tuples

def get_depth_of_tree_nodes(tree):
	depth = np.empty(tree.tree_.feature.shape, dtype=np.int64)
	depth.fill(np.iinfo(depth.dtype).max)

	returned_indices, returned_depth = zip(*get_depth_from_starting_node(tree, 0, 0))
	returned_indices, returned_depth = np.array(returned_indices), np.array(returned_depth)
	depth[returned_indices] = returned_depth
	tree.depth = depth
	return tree

def get_usages_of_leaves(tree, dataset):
	# usages = np.empty(tree.tree_.feature.shape, dtype=np.int64)
	# usages.fill(0)
	applied = tree.apply(dataset)
	decision_path = tree.decision_path(dataset)
	assert decision_path[np.arange(decision_path.shape[0]),applied].all()
	usages = np.array(np.sum(decision_path, axis=0)).squeeze()
	assert len(usages) == len(tree.tree_.feature), f"{len(usages)}, {len(tree.tree_.feature)}"
	tree.usages = usages
	return tree

def get_harmless_leaves(tree):
	proba = tree.tree_.value[:,0,:]

	normalizer = proba.sum(axis=1)[:, np.newaxis]
	normalizer[normalizer == 0.0] = 1.0
	proba /= normalizer

	harmless = proba[:,opt.classWithBackdoor] >= 0.5
	# usages = np.empty(tree.tree_.feature.shape, dtype=np.int64)
	# # usages.fill(0)
	# harmless = np.array(np.sum(tree.decision_path(dataset), axis=0)).squeeze()
	# assert len(usages) == len(tree.tree_.feature), f"{len(usages)}, {len(tree.tree_.feature)}"
	tree.harmless = harmless
	return tree

def prune_most_useless_leaf(tree):
	harmless_filter = np.ones(tree.tree_.feature.shape, dtype=np.bool) if not opt.pruneOnlyHarmless else tree.harmless
	if opt.depth:
		sorted_indices = np.lexsort((tree.depth, tree.usages,))
	else:
		sorted_indices = np.lexsort((tree.usages,))
	filtered_sorted_indices = sorted_indices[(tree.tree_.children_left[sorted_indices]==TREE_LEAF) & (tree.tree_.children_right[sorted_indices]==TREE_LEAF) & harmless_filter[sorted_indices]]

	most_useless = filtered_sorted_indices[0]

	pruned_node_dict = {"feature": tree.tree_.feature[most_useless], "threshold": tree.tree_.threshold[most_useless], "usages": tree.usages[most_useless]}
	if opt.depth:
		pruned_node_dict["depth"] = tree.depth[most_useless]

	prune_leaf(tree, most_useless)
	return tree, pruned_node_dict

def prune_steps_from_tree(tree, steps):
	pruned_nodes_dict = None
	for step in range(steps):
		tree, pruned_node_dict = prune_most_useless_leaf(tree)
		if pruned_nodes_dict is None:
			pruned_nodes_dict = pruned_node_dict
			for key in pruned_nodes_dict:
				pruned_nodes_dict[key] = [pruned_nodes_dict[key]]
		else:
			for key in pruned_nodes_dict:
				pruned_nodes_dict[key].append(pruned_node_dict[key])
	return tree, pruned_nodes_dict

def prune_leaf(tree, index):
	# print("prune_leaf", index)
	assert index != 0
	assert not tree.pruned[index]
	# To check that a node is a leaf, you have to check if both its left and right
	# child have the value TREE_LEAF set
	assert tree.tree_.children_left[index] == TREE_LEAF and tree.tree_.children_right[index] == TREE_LEAF
	parent_index = tree.parents[index]
	assert parent_index != TREE_LEAF

	is_left = np.where(tree.tree_.children_left==index)[0]
	is_right = np.where(tree.tree_.children_right==index)[0]
	# Makes sure that one node cannot have two parents
	assert (is_left.shape[0]==0) != (is_right.shape[0]==0)

	new_child = tree.tree_.children_right[parent_index] if is_left else tree.tree_.children_left[parent_index]

	tree.tree_.feature[parent_index] = tree.tree_.feature[new_child]
	tree.tree_.threshold[parent_index] = tree.tree_.threshold[new_child]
	tree.tree_.value[parent_index] = tree.tree_.value[new_child]
	if opt.pruneOnlyHarmless:
		tree.harmless[parent_index] = tree.harmless[new_child]
	tree.tree_.children_left[parent_index] = tree.tree_.children_left[new_child]
	tree.tree_.children_right[parent_index] = tree.tree_.children_right[new_child]
	tree.tree_.value[parent_index,:,:] = tree.tree_.value[new_child,:,:]
	# tree.parents[parent_index] = tree.parents[new_child]
	tree.usages[parent_index] = tree.usages[new_child]
	# tree.pruned[parent_index] = tree.pruned[new_child]
	assert (tree.tree_.children_left[new_child] == TREE_LEAF) == (tree.tree_.children_right[new_child] == TREE_LEAF)
	if tree.tree_.children_left[new_child] != TREE_LEAF:
		tree.parents[tree.tree_.children_left[new_child]] = parent_index
		tree.parents[tree.tree_.children_right[new_child]] = parent_index
	tree.tree_.children_left[new_child] = TREE_LEAF
	tree.tree_.children_right[new_child] = TREE_LEAF

	tree.tree_.feature[index] = TREE_UNDEFINED
	tree.tree_.threshold[index] = TREE_UNDEFINED
	tree.parents[index] = -1
	tree.usages[index] = np.iinfo(tree.usages.dtype).max
	tree.pruned[index] = 1
	if opt.depth:
		tree.depth[index] = np.iinfo(tree.depth.dtype).max

	tree.tree_.feature[new_child] = TREE_UNDEFINED
	tree.tree_.threshold[new_child] = TREE_UNDEFINED
	tree.parents[new_child] = -1
	tree.usages[new_child] = np.iinfo(tree.usages.dtype).max
	tree.pruned[new_child] = 1
	if opt.depth:
		tree.depth[new_child] = np.iinfo(tree.depth.dtype).max

def reachable_nodes(tree, only_leaves=False):
	n_remaining_nodes = 0
	stack = [0]
	while len(stack) > 0:
		current_index = stack.pop()
		child_left = tree.tree_.children_left[current_index]
		child_right = tree.tree_.children_right[current_index]
		if not only_leaves or (only_leaves and (child_left==child_right)):
			n_remaining_nodes += 1
		if child_left != child_right:
			stack.append(child_left)
			stack.append(child_right)
	return n_remaining_nodes

def get_indices_for_backdoor_pruning(all_validation_indices=False):
	_, test_indices = get_nth_split(dataset, opt.nFold, opt.fold)

	split_point = int(math.floor(len(test_indices)/2))
	validation_indices, test_indices = test_indices[:split_point], test_indices[split_point:]

	good_validation_indices = [index for index in validation_indices if backdoor_vector[index] == 0]
	assert len(good_validation_indices) != len(validation_indices), "Maybe you don't run --backdoor?"

	good_test_indices = [index for index in test_indices if backdoor_vector[index] == 0]
	bad_test_indices = [index for index in test_indices if backdoor_vector[index] == 1]
	assert (y[bad_test_indices,0] == opt.classWithBackdoor).all()

	harmless_good_validation_indices = [index for index in good_validation_indices if y[index,0] == opt.classWithBackdoor]
	# harmless_good_validation_indices = good_validation_indices
	assert len(harmless_good_validation_indices) > 0

	validation_indices = (harmless_good_validation_indices if opt.pruneOnlyHarmless else good_validation_indices) if not all_validation_indices else validation_indices

	validation_indices = validation_indices[:int(len(validation_indices)*opt.reduceValidationSet)]
	return validation_indices, good_test_indices, bad_test_indices

def prune_backdoor_rf():
	global rf

	validation_indices, good_test_indices, bad_test_indices = get_indices_for_backdoor_pruning()
	validation_data = x[validation_indices,:]

	for index, tree in enumerate(rf.estimators_):
		tree = get_parents_of_tree_nodes(tree)
		# tree = get_depth_of_tree_nodes(tree)
		tree = get_usages_of_leaves(tree, validation_data)
		if opt.depth:
			tree = get_depth_of_tree_nodes(tree)
		if opt.pruneOnlyHarmless:
			tree = get_harmless_leaves(tree)
			tree.original_harmless = copy.deepcopy(tree.harmless)
		tree.original_n_leaves = copy.deepcopy(tree.tree_.n_leaves)
		tree.original_children_left = copy.deepcopy(tree.tree_.children_left)
		tree.original_children_right = copy.deepcopy(tree.tree_.children_right)
		tree.pruned = np.zeros(tree.tree_.feature.shape, dtype=np.uint8)
		rf.estimators_[index] = tree

	step_width = 1/(opt.nSteps+1)

	new_rfs = [rf]
	steps_done = [[0]*len(rf.estimators_)]
	for step in range(opt.nSteps):
		new_rf = copy.deepcopy(new_rfs[-1])
		steps_done.append([])
		for index, tree in enumerate(new_rf.estimators_):
			n_nodes = tree.original_n_leaves if not opt.pruneOnlyHarmless else sum(tree.original_harmless & (tree.original_children_left==TREE_LEAF) & (tree.original_children_right==TREE_LEAF))
			if step==0:
				print("n_nodes", n_nodes)
			pruned_steps = int(round(step_width*(step)*n_nodes))
			pruned_steps_after_this = int(round(step_width*(step+1)*n_nodes))
			steps_to_do = pruned_steps_after_this - pruned_steps
			steps_done[-1].append(pruned_steps_after_this)
			print("Pruned", pruned_steps, "steps going", steps_to_do, "steps until", pruned_steps_after_this, "steps or", (step+1)/(opt.nSteps+1), "with", reachable_nodes(tree), "nodes remaining and", reachable_nodes(tree, only_leaves=True), "leaves")
			new_tree, pruned_nodes_dict = prune_steps_from_tree(tree, steps_to_do)
			usages_average = np.mean(np.array(pruned_nodes_dict["usages"]))
			if opt.depth:
				depth_average = np.mean(np.array(pruned_nodes_dict["depth"]))
				print("Mean depth", depth_average)
			print("Mean usages", usages_average)
			new_rf.estimators_[index] = new_tree
		new_rfs.append(new_rf)

	scores = []
	scores_bd = []
	rel_steps = [ (step+1)/(opt.nSteps + 1) for step in range(-1,opt.nSteps) ]
	for rel_step, new_rf in zip(rel_steps, new_rfs):
		print(f"pruned: {rel_step}")
		print("non-backdoored")
		scores.append(output_scores(y[good_test_indices,0], new_rf.predict(x[good_test_indices,:])))
		print("backdoored")
		scores_bd.append(output_scores(y[bad_test_indices,0], new_rf.predict(x[bad_test_indices,:]), only_accuracy=True))

	scores = { name: [ score[name] for score in scores ] for name in scores[0] }
	scores_bd = { name: [ score[name] for score in scores_bd ] for name in scores_bd[0] }
	os.makedirs('prune%s' % dirsuffix, exist_ok=True)
	filename = 'prune%s%s/prune_%.2f%s%s%s.pickle' % (dirsuffix, "_"+opt.extraLabel if opt.extraLabel else "", opt.reduceValidationSet, '_oh' if opt.pruneOnlyHarmless else '', '_d' if opt.depth else '', suffix)
	with open(filename, 'wb') as f:
		pickle.dump([rel_steps, steps_done, scores, scores_bd], f)

def noop_nn():
	pass
noop_rf = noop_nn

if __name__=="__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('--dataroot', required=True, help='path to dataset')
	parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
	parser.add_argument('--niter', type=int, default=100, help='number of epochs to train for')
	parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
	parser.add_argument('--dropoutProbability', type=float, default=0.2, help='probability for each neuron to be withheld in an iteration')
	parser.add_argument('--fold', type=int, default=0, help='fold to use')
	parser.add_argument('--nFold', type=int, default=3, help='total number of folds')
	parser.add_argument('--nSteps', type=int, default=9, help="number of steps for which to store the pruned classifier")
	parser.add_argument('--nEstimators', type=int, default=100, help='estimators for random forest')
	parser.add_argument('--net', default='', help="path to net (to continue training)")
	parser.add_argument('--function', default='train', help='the function that is going to be called')
	parser.add_argument('--arg', default='', help="optional arguments")
	parser.add_argument('--manualSeed', default=0, type=int, help='manual seed')
	parser.add_argument('--backdoor', action='store_true', help='include backdoor')
	parser.add_argument('--naive', action='store_true', help='include naive version of the backdoor')
	parser.add_argument('--depth', action='store_true', help='whether depth should be considered in the backdoor pruning algorithm')
	parser.add_argument('--pruneOnlyHarmless', action='store_true', help='whether only harmless nodes shall be pruned')
	parser.add_argument('--takeSignOfActivation', action='store_true')
	parser.add_argument('--onlyLastLayer', action='store_true', help='whether only the last layer is considered for pruning')
	parser.add_argument('--onlyFirstLayer', action='store_true', help='whether only the first layer is considered for pruning')
	parser.add_argument('--pruneConnections', action='store_true', help='whether the connections in the matrix should be pruned and not the neurons themselves')
	parser.add_argument('--plotHistogram', action='store_true')
	parser.add_argument('--plotHeatmap', action='store_true')
	parser.add_argument('--correlation', action='store_true')
	parser.add_argument('--normalizationData', default="", type=str, help='normalization data to use')
	parser.add_argument('--extraLabel', default="", type=str, help='extra label to add to output files of prune_backdoor')
	parser.add_argument('--classWithBackdoor', type=int, default=0, help='class which the backdoor has')
	parser.add_argument('--method', choices=['nn', 'rf'])
	parser.add_argument('--maxRows', default=sys.maxsize, type=int, help='number of rows from the dataset to load (for debugging mainly)')
	parser.add_argument('--modelIndexToLoad', default=0, type=int, help='which model to load when loading from a pickle file output after pruning')
	parser.add_argument('--reduceValidationSet', type=float, default=1, help='relative amount of validation set to use for pruning')
	parser.add_argument('--nData', type=int, default=100, help='number of samples to use for computing PDP/ALE/ICE plots')

	opt = parser.parse_args()
	print(opt)

	seed = opt.manualSeed
	# if seed is None:
	# 	seed = random.randrange(1000)
	# 	print("No seed was specified, thus choosing one randomly:", seed)
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)

	if opt.backdoor:
		# TODO: makes no more sense with commit 5ad8394b61f88f4eba1408f49c4056bdb9607561
		suffix = '_%s_%d_bd' % (opt.method, opt.fold)
	else:
		suffix = '_%s_%d' % (opt.method, opt.fold)

	dirsuffix = '_%s' % opt.dataroot[:-4]

	# MAX_ROWS = sys.maxsize
	# # MAX_ROWS = 1_000_000
	# # MAX_ROWS = 10_000

	csv_name = opt.dataroot
	df = pd.read_csv(csv_name, nrows=opt.maxRows).fillna(0)
	df = df[df['flowDurationMilliseconds'] < 1000 * 60 * 60 * 24 * 10]

	del df['flowStartMilliseconds']
	del df['sourceIPAddress']
	del df['destinationIPAddress']
	attack_vector = np.array(list(df['Attack']))
	assert len(attack_vector.shape) == 1
	backdoor_vector = np.zeros(attack_vector.shape[0])

	# print("Rows", df.shape[0])

	if opt.backdoor:
		ratio_of_those_with_stdev_not_zero_forward = (df["apply(stdev(ipTTL),forward)"] != 0).sum()/df.shape[0]
		ratio_of_those_with_stdev_not_zero_backward = (df["apply(stdev(ipTTL),backward)"] != 0).sum()/df.shape[0]

		ratio_of_those_attacks_with_stdev_not_zero_forward = ((df["apply(stdev(ipTTL),forward)"] != 0) & (df["Label"] == 1)).sum()/(df["Label"] == 1).sum()
		ratio_of_those_attacks_with_stdev_not_zero_backward = ((df["apply(stdev(ipTTL),backward)"] != 0) & (df["Label"] == 1)).sum()/(df["Label"] == 1).sum()

		ratio_of_those_good_ones_with_stdev_not_zero_forward = ((df["apply(stdev(ipTTL),forward)"] != 0) & (df["Label"] == 0)).sum()/(df["Label"] == 0).sum()
		ratio_of_those_good_ones_with_stdev_not_zero_backward = ((df["apply(stdev(ipTTL),backward)"] != 0) & (df["Label"] == 0)).sum()/(df["Label"] == 0).sum()

		print("ratio of stdev zero")
		print("all")
		print("forward", ratio_of_those_with_stdev_not_zero_forward)
		print("backward", ratio_of_those_with_stdev_not_zero_backward)
		print("attacks")
		print("forward", ratio_of_those_attacks_with_stdev_not_zero_forward)
		print("backward", ratio_of_those_attacks_with_stdev_not_zero_backward)
		print("good ones")
		print("forward", ratio_of_those_good_ones_with_stdev_not_zero_forward)
		print("backward", ratio_of_those_good_ones_with_stdev_not_zero_backward)


		attack_records = df[df["Label"] != opt.classWithBackdoor].to_dict("records", into=collections.OrderedDict)
		# print("attack_records", attack_records)
		forward_ones = [item for item in [add_backdoor(item, "forward") for item in attack_records] if item is not None]
		print("forward_ones", len(forward_ones))
		# backward_ones = [item for item in [add_backdoor(item, "backward") for item in attack_records] if item is not None]
		# print("backward_ones", len(backward_ones))
		# both_ones = [item for item in [add_backdoor(item, "backward") for item in forward_ones] if item is not None]
		# print("both_ones", len(both_ones))
		# pd.DataFrame.from_dict(attack_records).to_csv("attack.csv", index=False)
		# pd.DataFrame.from_dict(forward_ones).to_csv("forward_backdoor.csv", index=False)
		# pd.DataFrame.from_dict(backward_ones).to_csv("backward_backdoor.csv", index=False)
		# pd.DataFrame.from_dict(both_ones).to_csv("both_backdoor.csv", index=False)
		backdoored_records = forward_ones# + backward_ones + both_ones
		# print("backdoored_records", len(backdoored_records))
		backdoored_records = pd.DataFrame.from_dict(backdoored_records)
		# backdoored_records.to_csv("exported_df.csv")
		# print("backdoored_records", backdoored_records[:100])
		print("backdoored_records rows", backdoored_records.shape[0])

		df = pd.concat([df, backdoored_records], axis=0, ignore_index=True, sort=False)
		# print("backdoored_records", backdoored_records)
		attack_vector = np.concatenate((attack_vector, np.array(list(backdoored_records['Attack']))))
		assert len(backdoor_vector.shape) == 1, len(backdoor_vector.shape)
		backdoor_vector = np.concatenate((backdoor_vector, np.ones(backdoored_records.shape[0])))

	del df['Attack']
	features = df.columns[:-1]
	print("Final rows", df.shape)
	# df[:1000].to_csv("exported_2.csv")

	shuffle_indices = np.array(list(range(df.shape[0])))
	random.shuffle(shuffle_indices)

	data = df.values
	print("data.shape", data.shape)
	data = data[shuffle_indices,:]
	print("attack_vector.shape", attack_vector.shape)
	attack_vector = attack_vector[shuffle_indices]
	backdoor_vector = backdoor_vector[shuffle_indices]
	assert len(attack_vector) == len(backdoor_vector) == len(data)
	columns = list(df)
	print("columns", columns)

	x, y = data[:,:-1].astype(np.float32), data[:,-1:].astype(np.uint8)
	file_name = opt.dataroot[:-4]+"_"+(("backdoor" if not opt.naive else "backdoor_naive") if opt.backdoor else "normal")
	if opt.normalizationData == "":
		file_name_for_normalization_data = file_name+"_normalization_data.pickle"
		means = np.mean(x, axis=0)
		stds = np.std(x, axis=0)
		stds[stds==0.0] = 1.0
		# np.set_printoptions(suppress=True)
		# stds[np.isclose(stds, 0)] = 1.0
		with open(file_name_for_normalization_data, "wb") as f:
			f.write(pickle.dumps((means, stds)))
	else:
		file_name_for_normalization_data = opt.normalizationData
		with open(file_name_for_normalization_data, "rb") as f:
			means, stds = pickle.loads(f.read())
	assert means.shape[0] == x.shape[1], "means.shape: {}, x.shape: {}".format(means.shape, x.shape)
	assert stds.shape[0] == x.shape[1], "stds.shape: {}, x.shape: {}".format(stds.shape, x.shape)
	assert not (stds==0).any(), "stds: {}".format(stds)
	x = (x-means)/stds

	dataset = OurDataset(x, y)

	current_time = datetime.now().strftime('%b%d_%H-%M-%S')

	if opt.method == 'nn':
		cuda_available = torch.cuda.is_available()
		device = torch.device("cuda:0" if cuda_available else "cpu")

		net = make_net(x.shape[-1], 1, 3, 512).to(device)
		print("net", net)

		if opt.net != '':
			print("Loading", opt.net)
			if opt.function == 'finetune' and opt.net.endswith('.pickle'):
				with open(opt.net, 'rb') as f:
					loadfrom = pickle.load(f)[3]
					# print("loadfrom[3]", loadfrom[3])
					loadfrom = io.BytesIO(loadfrom[opt.modelIndexToLoad])
			else:
				loadfrom = opt.net
			# import pdb; pdb.set_trace()
			net.load_state_dict(torch.load(loadfrom, map_location=device))

	elif opt.method == 'rf':
		train_indices, _ = get_nth_split(dataset, opt.nFold, opt.fold)

		if opt.net:
			if opt.net.endswith('.rfmodel.gz'):
				with gzip.open(opt.net, 'rb') as f:
					rf = pickle.load(f)
			else:
				rf = pickle.load(open(opt.net, 'rb'))
		else:
			rf = RandomForestClassifier(n_estimators=opt.nEstimators)
			rf.fit(x[train_indices,:], y[train_indices,0])
			# XXX: The following code is broken! It should use predict_proba instead of predict probably
			predictions = rf.predict_proba(x[train_indices,:])
			# print("predictions", predictions.shape, predictions)
			summed_up = np.sum(predictions, axis=1)
			assert (np.isclose(summed_up, 1)).all(), "summed_up: {}".format(summed_up.tolist())

	globals()['%s_%s' % (opt.function, opt.method)]()


