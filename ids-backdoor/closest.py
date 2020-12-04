#!/usr/bin/env python3

import sys
import os

import pandas as pd
import numpy as np

from sklearn.preprocessing import minmax_scale
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

DIR_NAME = "pdp"

def closest(data, data_labels, attacks, data_predictions, misclassified, misclassified_labels, misclassified_attacks, misclassified_predictions, means, stds, suffix):

	closest_correct_of_same_class = []
	closest_correct_of_different_class = []

	attack_indices = data_labels == 1
	correct_indices = data_labels == data_predictions

	# correct_attack_labels = data_labels[attack_indices & correct_indices]
	correct_attack_samples = data[attack_indices & correct_indices]
	correct_attack_attacks = attacks[attack_indices & correct_indices]
	# print("correct_attack_attacks", list(correct_attack_attacks))
	# correct_good_labels = data_labels[~attack_indices & correct_indices]
	correct_good_samples = data[~attack_indices & correct_indices]
	correct_good_attacks = attacks[~attack_indices & correct_indices]
	# print("correct_good_attacks", list(correct_good_attacks))

	assert not (misclassified_predictions == misclassified_labels).any()

	closest_good = []
	closest_good_attack = []
	closest_attack = []
	closest_attack_attack = []

	# print("misclassified", misclassified, misclassified_labels)
	for index, item in enumerate(zip(misclassified, misclassified_labels)):
		print("index", index)
		sample, label = item
		distances = np.sum((correct_good_samples - sample)**2, axis=1)
		assert len(distances) == len(correct_good_samples)
		sorted_indices = np.argsort(distances)
		result_index = sorted_indices[1] if distances[0] == 0 else sorted_indices[0]
		result_good_one = correct_good_samples[result_index]
		closest_good.append(result_good_one)
		closest_good_attack.append(correct_good_attacks[result_index])

		distances = np.sum((correct_attack_samples - sample)**2, axis=1)
		assert len(distances) == len(correct_attack_samples)
		sorted_indices = np.argsort(distances)
		result_index = sorted_indices[1] if distances[0] == 0 else sorted_indices[0]
		result_attack_one = correct_attack_samples[result_index]
		closest_attack.append(result_attack_one)
		closest_attack_attack.append(correct_attack_attacks[result_index])
	# showable_results = list(zip(np.array(misclassified_labels, dtype=np.uint8).tolist(), (misclassified*stds+means).tolist(), (np.array(closest_good)*stds+means).tolist(), (np.array(closest_attack)*stds+means).tolist()))
	# showable_results = list(zip(misclassified_attacks.tolist(), (misclassified*stds+means).tolist(), np.array(closest_good_attack).tolist(), (np.array(closest_good)*stds+means).tolist(), np.array(closest_attack_attack).tolist(), (np.array(closest_attack)*stds+means).tolist()))
	showable_results = list(zip(misclassified_attacks.tolist(), np.array(closest_good_attack).tolist(), np.array(closest_attack_attack).tolist()))

	good_ones = [(item[0], item[2]) for item in showable_results if item[0] == "Normal"]
	bad_ones = [(item[0], item[1]) for item in showable_results if item[0] != "Normal"]

	result = [item.strip() for item in list(zip(*good_ones))[1]]
	result_count_good_ones = [(item, result.count(item)) for item in sorted(list(set(result)))]

	result = [item.strip() for item in list(zip(*bad_ones))[0]]
	result_count_bad_ones = [(item, result.count(item)) for item in sorted(list(set(result)))]

	print("good_ones", result_count_good_ones)
	print("bad_ones", result_count_bad_ones)
