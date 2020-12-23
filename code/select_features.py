import itertools
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from os.path import isfile
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, cross_validate, cross_val_predict
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree


class selectfeatures:
	
	def __init__(self,dataset,importance_trashold,precision_threshold,per_class_precision_trashold,n_exec,test):
		self.dataset=dataset
		self.importance_trashold=importance_trashold
		self.precision_threshold=precision_threshold
		self.per_class_precision_trashold=per_class_precision_trashold
		self.n_exec=n_exec
		self.test=test

	def load_data(self,csv_file_path):
		return pd.read_csv(csv_file_path)



	def extract_features_labels(self,data, label, drop_cols):
		labels = data[label]
		features = data.drop(drop_cols, axis=1)
		feature_list = list(features.columns)
		return features, feature_list, labels

	def check_threshold_by_class(self,dataArray, value):
		result = False
		for i in range(len(dataArray)):
			if dataArray[i] < value:
				result = True
				break

		return result




	def select_features(self,data, label, imp_threshold, precision_threshold, trashold_by_class,
					n_exec=1000,
					n_estimators=1000, test_size=0.25,
					random_state=None):

		result_metrics = None
		unique_labels = data[label].unique()
		unique_labels.sort()
		model_count = 1
		for i in range(n_exec):
			labels = data[label]
			features = data[[var for var in list(data.columns) if var != label]]

		feature_list = list(features.columns)
		while True:

			train_features, test_features, train_labels, test_labels = train_test_split(features,
																						labels,
																						test_size=test_size,
																						shuffle=True,
																						random_state=random_state)
			
			rf = RandomForestClassifier(bootstrap=True,
										class_weight=None,
										criterion='gini',
										max_depth=None,
										max_features='auto',
										max_leaf_nodes=None,
										min_samples_leaf=1,
										min_samples_split=2,
										min_weight_fraction_leaf=0.0,
										n_estimators=n_estimators,
										n_jobs=10,
										oob_score=False,
										random_state=random_state,
										verbose=0,
										warm_start=False)

			
			rf.fit(train_features, train_labels)

			importances = list(rf.feature_importances_)


			feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]

			feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)

			sorted_importances = [importance[1] for importance in feature_importances]
			sorted_features = [importance[0] for importance in feature_importances]

			cumulative_importances = np.cumsum(sorted_importances)

			if max(cumulative_importances) < imp_threshold:
				print('***| Stopping-0 |***')
				print('Execution:', i, ', Model:', model_count)
				print('!!!!! Low importance !!!!!!!' )
				break

			n_importante_features = np.where(cumulative_importances >= imp_threshold)[0][0]

			if n_importante_features < 2:
				print('***| Stopping-1 |***')
				print('Execution:', i, ', Model:', model_count)
				print('###### Less than 2 variables #######' )
				break

			important_feature_names = [feature[0] for feature in feature_importances[0:n_importante_features]]

			important_indices = [feature_list.index(feature) for feature in important_feature_names]
			
			important_train_features = train_features.iloc[:, important_indices]
			important_test_features = test_features.iloc[:, important_indices]

			rf.fit(important_train_features, train_labels)

			predictions = rf.predict(important_test_features)
			accuracy = accuracy_score(test_labels, predictions)
			precision = precision_score(test_labels, predictions, average='weighted')
			recall = recall_score(test_labels, predictions, average='weighted')
			f1score = f1_score(test_labels, predictions, average='weighted')
			per_class_precision_ = precision_score(test_labels, predictions, average=None)

			if precision < precision_threshold or self.check_threshold_by_class(per_class_precision_,
																			trashold_by_class):
				print('***| Stopping-2 |***')
				print('Model: ', model_count)
				print('===> Low precision <===')
				print('Global precision: ', precision)
				print('Per class precision: ', per_class_precision_)
				print('Total of importante features: ', n_importante_features)
				break

			if result_metrics is None:
				result_metrics = pd.DataFrame({
					'model': [model_count for j in range(len(sorted_features))],
					'n_variables': [len(sorted_features) for j in range(len(sorted_features))],
					'variables': sorted_features,
					'importance': sorted_importances,
					'accuracy': [accuracy for j in range(len(sorted_features))],
					'precision': [precision for j in range(len(sorted_features))],
					'recall': [recall for j in range(len(sorted_features))],
					'f1score': [f1score for j in range(len(sorted_features))]
				})
			else:
				result_metrics = result_metrics.append(pd.DataFrame({
					'model': [model_count for j in range(len(sorted_features))],
					'n_variables': [len(sorted_features) for j in range(len(sorted_features))],
					'variables': sorted_features,
					'importance': sorted_importances,
					'accuracy': [accuracy for j in range(len(sorted_features))],
					'precision': [precision for j in range(len(sorted_features))],
					'recall': [recall for j in range(len(sorted_features))],
					'f1score': [f1score for j in range(len(sorted_features))]
				}))

			print('------------------------------------| SUMMARY |------------------------------------')
			print('Execution:', i, ', Model:', model_count)
			print('Number of features for', imp_threshold * 100, '% importance:', n_importante_features)
			print('Selected variables: ', sorted_features)
			print(classification_report(test_labels, predictions, target_names=list(set(labels))))
			print('-----------------------------------------------------------------------------------')
			print('Precision: ', precision)
			print('Accuracy: ', accuracy)
			print('-----------------------------------------------------------------------------------')

			features = features.iloc[:, important_indices]
			feature_list = list(features.columns)
			model_count += 1

		return result_metrics

	def select_featuresDecisionTree(self,data, label, imp_threshold, precision_threshold, trashold_by_class,
					n_exec=1000,
					n_estimators=1000, test_size=0.25,
					random_state=None):

		result_metrics = None
		unique_labels = data[label].unique()
		unique_labels.sort()
		model_count = 1
		for i in range(n_exec):
			labels = data[label]
			features = data[[var for var in list(data.columns) if var != label]]

		feature_list = list(features.columns)
		while True:

			train_features, test_features, train_labels, test_labels = train_test_split(features,
																						labels,
																						test_size=test_size,
																						shuffle=True,
																						random_state=random_state)
			
			dt = tree.DecisionTreeRegressor()
			
			dt.fit(train_features, train_labels)

			importances = list(dt.feature_importances_)

			feature_importances = [(feature, round(importance, 2)) for feature, importance in
								   zip(feature_list, importances)]

			feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)

			sorted_importances = [importance[1] for importance in feature_importances]
			sorted_features = [importance[0] for importance in feature_importances]

			cumulative_importances = np.cumsum(sorted_importances)

			if max(cumulative_importances) < imp_threshold:
				print('***| Stopping-0 |***')
				print('Execution:', i, ', Model:', model_count)
				print('!!!!! Low importance !!!!!!!' )
				break

			n_importante_features = np.where(cumulative_importances >= imp_threshold)[0][0]

			if n_importante_features < 2:
				print('***| Stopping-1 |***')
				print('Execution:', i, ', Model:', model_count)
				print('###### Less than 2 variables #######' )
				break

			important_feature_names = [feature[0] for feature in feature_importances[0:n_importante_features]]

			important_indices = [feature_list.index(feature) for feature in important_feature_names]
			
			important_train_features = train_features.iloc[:, important_indices]
			important_test_features = test_features.iloc[:, important_indices]

			dt.fit(important_train_features, train_labels)

			predictions = dt.predict(important_test_features)
			accuracy = accuracy_score(test_labels, predictions)
			precision = precision_score(test_labels, predictions, average='weighted')
			recall = recall_score(test_labels, predictions, average='weighted')
			f1score = f1_score(test_labels, predictions, average='weighted')
			per_class_precision_ = precision_score(test_labels, predictions, average=None)

			if precision < precision_threshold or self.check_threshold_by_class(per_class_precision_,
																			trashold_by_class):
				print('***| Stopping-2 |***')
				print('Model: ', model_count)
				print('===> Low precision <===')
				print('Global precision: ', precision)
				print('Per class precision: ', per_class_precision_)
				print('Total of importante features: ', n_importante_features)
				break

			if result_metrics is None:
				result_metrics = pd.DataFrame({
					'model': [model_count for j in range(len(sorted_features))],
					'n_variables': [len(sorted_features) for j in range(len(sorted_features))],
					'variables': sorted_features,
					'importance': sorted_importances,
					'accuracy': [accuracy for j in range(len(sorted_features))],
					'precision': [precision for j in range(len(sorted_features))],
					'recall': [recall for j in range(len(sorted_features))],
					'f1score': [f1score for j in range(len(sorted_features))]
				})
			else:
				result_metrics = result_metrics.append(pd.DataFrame({
					'model': [model_count for j in range(len(sorted_features))],
					'n_variables': [len(sorted_features) for j in range(len(sorted_features))],
					'variables': sorted_features,
					'importance': sorted_importances,
					'accuracy': [accuracy for j in range(len(sorted_features))],
					'precision': [precision for j in range(len(sorted_features))],
					'recall': [recall for j in range(len(sorted_features))],
					'f1score': [f1score for j in range(len(sorted_features))]
				}))

			print('------------------------------------| SUMMARY |------------------------------------')
			print('Execution:', i, ', Model:', model_count)
			print('Number of features for', imp_threshold * 100, '% importance:', n_importante_features)
			print('Selected variables: ', sorted_features)
			print(classification_report(test_labels, predictions, target_names=list(set(labels))))
			print('-----------------------------------------------------------------------------------')
			print('Precision: ', precision)
			print('Accuracy: ', accuracy)
			print('-----------------------------------------------------------------------------------')

			features = features.iloc[:, important_indices]
			feature_list = list(features.columns)
			model_count += 1

		return result_metrics

		

	def main(self):
		result_dir = '../results/'
		
		data = self.load_data(self.dataset)
		
		select_cols = ["ip_proto","ip_len_mean","ip_len_median","ip_len_var",
				   "ip_len_std","ip_len_entropy","ip_len_cv","ip_len_cvq",
				   "ip_len_rte", "sport_mean","sport_median","sport_var","sport_std",
				   "sport_entropy","sport_cv","sport_cvq","sport_rte",
				   "dport_mean","dport_median","dport_var","dport_std",
				   "dport_entropy","dport_cv","dport_cvq","dport_rte",
				   "tcp_flags_mean","tcp_flags_median","tcp_flags_var",
				   "tcp_flags_std","tcp_flags_entropy","tcp_flags_cv",
				   "tcp_flags_cvq","tcp_flags_rte", "Label3"]
		
		data = data[select_cols].copy()

		result_metrics = self.select_features(data, 'Label3',
							 float(self.importance_trashold),
							 float(self.precision_threshold),
							 float(self.per_class_precision_trashold),
							 n_exec=int(self.n_exec),
							 n_estimators=100,
							 test_size=0.5,
							 random_state=None)

		result_metrics1=self.select_featuresMNB(data, 'Label3',
							 float(self.importance_trashold),
							 float(self.precision_threshold),
							 float(self.per_class_precision_trashold),
							 n_exec=int(self.n_exec),
							 n_estimators=100,
							 test_size=0.5,
							 random_state=None)
		
		

if __name__=='__main__':
	select_feature=selectfeatures("../data/dataset_descriptor.csv",0.99, 0.95, 0.85,5,5)
	select_feature.main()