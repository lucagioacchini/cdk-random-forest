#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn import tree
import graphviz
import lib.minimization as min

def clean_data(path):
	"""Load the data file from the provided path, manage and correct the worng data,	
	turn the cathegorical features into numerical ones and create a dataFrame.
	
	Args:
		path: (string) path of the data file
	
	Returns:
		data: (numpy matrix) dataFrame of cleaned data
	"""

	feat_names = [
				"age",
				"bp",
				"sg",
				"al",
				"su", # normal->1, abnormal->0
				"rbc", # normal->1, abnormal->0
				"pc", # present->1, notpresent->0
				"pcc", # present->1, notpresent->0
				"ba",
				"bgr",
				"bu",
				"sc",
				"sod",
				"pot",
				"hemo",
				"pcv",
				"wbcc",
				"rbcc",
				"htn", #yes->1, no->0
				"dm", #yes->, no->0
				"cad", # yes->1, no->0
				"appet", # good->1, poor->0
				"pe", # yes->1, no->0
				"ane", # yes->1, no->0
				"class", # cdk->1, notcdk->0		
	]
	
	file_data = pd.read_csv(
					path, 
					skiprows = range(0, 29), 
					sep = ',',
					header = None,
					na_values=['?','\t?', '\t'],
					names=feat_names#,
	)
	pd.set_option('display.max_rows', None)

	file_data.replace({
						"rbc":{
							"normal":1,
							"abnormal": 0,
						},
						"pc":{
							"normal":1,
							"abnormal": 0,
						},
						"pcc":{
							"present":1,
							"notpresent": 0,
						},
						"ba":{
							"present":1,
							"notpresent": 0,
						},
						"htn":{
							"yes":1,
							"no": 0,
						},
						"dm":{
							"yes":1,
							" yes":1,
							"no": 0,
							"\tyes":1,
							"\tno":0,
						},
						"cad":{
							"yes":1,
							"no": 0,
							"\tno":0
						},
						"appet":{
							"good":1,
							"poor": 0,
						},
						"pe":{
							"yes":1,
							"no": 0,
						},
						"ane":{
							"yes":1,
							"no": 0,
						},
						"class":{
							"ckd\t":1,
							"ckd":1,
							"notckd":0,
						},
						"sg":{
							1005:1.005,
							1015:1.015,
							1025:1.025,
						}
	},
	inplace = True)
	data = file_data.values

	return data

def normalize(matrix):	
	"""Perform the standard score normalization.
	
	Args:
		matrix: (numpy matrix) matrix to normalize
	
	Returns:
		matrix: (numpu matrix) normalized matrix
	"""
	global norm_param
 
	norm_param = np.zeros((np.size(matrix, 1), 2), dtype=float)
	
	for j in range(np.size(matrix, 1)):
		mean = np.mean(matrix[:, j])
		var = np.var(matrix[:, j])
		
		norm_param[j, 0] = mean
		norm_param[j, 1] = var

		matrix[:, j] = (matrix[:, j] - mean)/np.sqrt(var)

	return matrix
		
def manage_data(data):
	"""Manage the data to perform the regression on the samples with missing 
	features.
	
	Args:
		data: (numpy matrix) cleaned data
	
	Returns:
		dataset: (numpy matrix) cleaned data without the samples with missing features
		X: (list) list of indices of samples with missing features
	"""
	global X
	
	to_delete = []
	to_X = []
	
	for i in range(np.size(data, 0)):
		nan_cnt = 0
		for j in range(np.size(data, 1)):	
			if str(data[i, j]) == "nan":
				nan_cnt += 1
		if nan_cnt == 0:
			to_X.append(i)
		elif nan_cnt > 4:
			to_delete.append(i)
	
	dataset = np.delete(data, to_delete, 0)
	X = np.vstack(data[to_X, :])
	
	X = normalize(X)
	
	return dataset, X
	
def regress(X_test, cols):
	"""Perform Ridge Regression
	
	Args:
		X_test: (numpy matrix) training dataset
		cols: (list) list of indices of columns to regress
		
	Returns:
		w: (numpy matrix) weights vector resulting from the regression
	"""
	LAMBDA = 10
	y_train = X[:, cols]
	X_train = np.delete(X, cols, 1)
	
	ridge = min.RidgeReg(y_train, X_train)
	ridge.run(LAMBDA)
	w = ridge.sol
	
	return w
	
def fill_missing_feature(data):
	"""Manage and perform regression on the samples with missing features
	
	Args:
		data: (numpy matrix) data to regress
	
	Returns:
		data: (numpy matrix) final data
	"""
	
	cath = [17, 16, 15, 14, 13, 12, 11, 10, 9, 1, 0]
	sg_val = [1.005,1.010,1.015,1.020,1.025]
	
	# determine missing features row per row
	for row in range(np.size(data, 0)):
		index = []
		for j in range(np.size(data, 1)):
			#if np.isnan (data[row, j]):
			if str(data[row, j]) == "nan":
				index.append(j) # vector of missing features indexes
		if index!=[]:
			# determine the otimum w vector
			w = regress(data, index)
			# normalization of original data
			temp = np.zeros((np.size(data, 0), np.size(data, 1)), dtype=float)
			for i in range(np.size(data, 1)):
				temp[:, i] = (data[:, i]-norm_param[i, 0])/np.sqrt(norm_param[i, 1])
			# remove the regressand columns from original data
			X_test = np.delete(temp, index, axis=1)
			# estimate of normalizated missing features
			y_hat_test = np.dot(X_test, w)
			
			# replace missing features 
			cnt = 0
			for item in index:
				# denormalize missing features before the replacement
				y_hat_test[row, cnt] = y_hat_test[row, cnt]*np.sqrt(norm_param[item, 1]) \
									 + norm_param[item, 0]
				# manage the cathegorical features
				if item in cath:
					y_hat_test[:, cnt] = np.round(y_hat_test[:, cnt], decimals = 3)
				elif item == 2:
					
					dist = np.zeros((5, 1), dtype=float)
					
					# round the feature to the nearest sg_val
					for val in range(np.size(y_hat_test, 0)):
						for k in range(5):
							dist[k] = np.absolute(y_hat_test[val, cnt] - sg_val[k])
						y_hat_test[val, cnt] = sg_val[np.argmin(dist)]
					
				else:
					y_hat_test[:, cnt] = np.round(y_hat_test[:, cnt], decimals = 0)
				# replace data
				data[row, item] = y_hat_test[row, cnt]
				cnt += 1	
			
	return data

def save_data(data, title):
	"""Save the data into a .csv file
	
	Args:
		data: (numpy matrix) final dataFrame
		title: (string) Title of the file
	"""
	df = pd.DataFrame.from_records(data)
	df.to_csv(title)
	
def decision_tree(data):
	"""Perform the decision tree algorithm
	
	Args:
		data: (numpy array) final cleaned and regressed data
	"""
	feat_names = [
				"age",
				"bp",
				"sg",
				"al",
				"su", # normal->1, abnormal->0
				"rbc", # normal->1, abnormal->0
				"pc", # present->1, notpresent->0
				"pcc", # present->1, notpresent->0
				"ba",
				"bgr",
				"bu",
				"sc",
				"sod",
				"pot",
				"hemo",
				"pcv",
				"wbcc",
				"rbcc",
				"htn", #yes->1, no->0
				"dm", #yes->, no->0
				"cad", # yes->1, no->0
				"appet", # good->1, poor->0
				"pe", # yes->1, no->0
				"ane", # yes->1, no->0
	]

	# extract the first 24 column (cut out the class one)
	data_set = np.delete(data, 24, 1)
	target = data[:, 24]
	# training phase
	clf = tree.DecisionTreeClassifier("entropy")
	clf = clf.fit(data_set, target)
	
	print clf.feature_importances_
	
	#view decision tree
	dot_data = tree.export_graphviz(clf, 
						out_file = "Tree.dot", 
						feature_names = feat_names,
						class_names = ["notcdk", "cdk"],
						filled = True,
						rounded = True,
						special_characters = True,
	)
	
# main
data = clean_data("data/chronic_kidney_disease.arff")
data, X = manage_data(data)
data = fill_missing_feature(data)
decision_tree(data)
