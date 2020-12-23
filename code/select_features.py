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


	def plot_confusion_matrix(self,cm, precision, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues, filename='.'):
		
		if normalize:
			cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
			print("Normalized confusion matrix")
		else:
			print('Confusion matrix, without normalization')
		print(cm)

		plt.imshow(cm, interpolation='nearest', cmap=cmap)
		plt.title(title)
		plt.colorbar()
		tick_marks = np.arange(len(classes))
		plt.xticks(tick_marks, classes, rotation=45)
		plt.yticks(tick_marks, classes)

		fmt = '.2f' if normalize else 'd'
		thresh = cm.max() / 2.
		for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
			plt.text(j, i, format(cm[i, j], fmt),
					horizontalalignment="center",
					color="white" if cm[i, j] > thresh else "black")

		plt.tight_layout()
		plt.ylabel('True label')
		plt.xlabel('Predicted label (Precision=' + str(round(precision * 100, 2)) + '%)')
		plt.savefig(filename, bbox_inches='tight')
		plt.close()


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

	def plot_importance(self,data, imp_threshold, filename):
		
		plt.plot(data['features'], data['cumulative_importance'], 'green', label='Cumulative importance')

		plt.hlines(y=imp_threshold, xmin=0,
			   xmax=len(data['features']),
			   color='red',
			   linestyles='dashed',
			   label='Importance threshold (' + str(imp_threshold) + ')')

		plt.xticks(rotation='vertical')
		plt.xlabel('Variable')
		plt.ylabel('Cumulative Importance')
		plt.legend(loc='lower right', numpoints=1, ncol=1, fancybox=False, shadow=False)
		plt.savefig(filename, bbox_inches='tight')
		plt.close()


	def test_prediction(self,data,
					label,
					cols,
					n_exec=1000,
					n_estimators=1000,
					test_size=0.5):
		labels = data[label]
		unique_labels = data[label].unique()
		unique_labels.sort()
		features = data[cols]

		feature_list = list(features.columns)
		result_metrics = None
		for j in range(1, n_exec+1):
			train_features, test_features, train_labels, test_labels = train_test_split(features,
																						labels,
																						test_size=test_size,
																						shuffle=True)

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
									verbose=0,
									warm_start=False)

		rf.fit(train_features, train_labels)
		predictions = rf.predict(test_features)
		accuracy = accuracy_score(test_labels, predictions)
		precision = precision_score(test_labels, predictions, average='weighted')
		recall = recall_score(test_labels, predictions, average='weighted')
		f1score = f1_score(test_labels, predictions, average='weighted')
		
		print('------------------------------------| SUMMARY USING RANDOM FOREST |------------------------------------')
		print('Test: ', j, '/', n_exec)
		print('Selectted variables: ', feature_list, '\n Labels:', unique_labels)
		print(classification_report(test_labels, predictions, target_names=list(set(labels))))
		print('-----------------------------------------------------------------------------------')
		print('Precision: ', precision)
		print('Accuracy: ', accuracy)
		print('-----------------------------------------------------------------------------------')

		if result_metrics is None:
			result_metrics = pd.DataFrame({
				'test': [j],
				'accuracy': [accuracy],
				'precision': [precision],
				'recall': [recall],
				'f1score': [f1score]
			})
		else:
			result_metrics = result_metrics.append(pd.DataFrame({
				'test': [j],
				'accuracy': [accuracy],
				'precision': [precision],
				'recall': [recall],
				'f1score': [f1score]
			}))

		return result_metrics


	def test_prediction1(self,data,
					label,
					cols,
					n_exec=1000,
					n_estimators=1000,
					test_size=0.5):
		labels = data[label]
		unique_labels = data[label].unique()
		unique_labels.sort()
		features = data[cols]

		feature_list = list(features.columns)
		result_metrics = None
		for j in range(1, n_exec+1):
			train_features, test_features, train_labels, test_labels = train_test_split(features,
																						labels,
																						test_size=test_size,
																						shuffle=True)
		nb = MultinomialNB()

		
		nb.fit(train_features, train_labels)
		
		predictions = nb.predict(test_features)
		accuracy = accuracy_score(test_labels, predictions)
		precision = precision_score(test_labels, predictions, average='weighted')
		recall = recall_score(test_labels, predictions, average='weighted')
		f1score = f1_score(test_labels, predictions, average='weighted')
		
		print('------------------------------------| SUMMARY USING MULTINOMIAL NB |------------------------------------')
		print('Test: ', j, '/', n_exec)
		print('Selectted variables: ', feature_list, '\n Labels:', unique_labels)
		print(classification_report(test_labels, predictions, target_names=list(set(labels))))
		print('-----------------------------------------------------------------------------------')
		print('Precision: ', precision)
		print('Accuracy: ', accuracy)
		print('-----------------------------------------------------------------------------------')

		if result_metrics is None:
			result_metrics = pd.DataFrame({
				'test': [j],
				'accuracy': [accuracy],
				'precision': [precision],
				'recall': [recall],
				'f1score': [f1score]
			})
		else:
			result_metrics = result_metrics.append(pd.DataFrame({
				'test': [j],
				'accuracy': [accuracy],
				'precision': [precision],
				'recall': [recall],
				'f1score': [f1score]
			}))

		return result_metrics

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


	def plot_barh(self,x, y, title, x_label, y_label, file_name):
		plt.rcParams['font.family'] = "Times"
		plt.rcParams["figure.figsize"] = [9, 6]
		fig, ax = plt.subplots()
		width = 0.75 
		ind = np.arange(len(y))  
		ax.barh(ind, y, width)
		ax.set_yticks(ind+width/2)
		ax.set_yticklabels(x, minor=False)
		plt.title(title)
		plt.xlabel(x_label)
		plt.ylabel(y_label)
		for i, v in enumerate(y):
			ax.text(v, i - 0.2, str(v), color='red', fontsize=8)

		plt.savefig(file_name, bbox_inches='tight')
		plt.gcf().clear()


	def plot_boxplot(self,df, title, x_label, y_label, file_name):
		plt.rcParams['font.family'] = "Times"
		fig, ax = plt.subplots()
		data_plot = []
		for col in df.columns:
			data_plot.append(df[col])

		ax.boxplot(data_plot, labels=df.columns, showfliers=True, meanline=True, showmeans=True)
		plt.title(title)
		plt.xlabel(x_label)
		plt.ylabel(y_label)
		plt.savefig(file_name, bbox_inches='tight')
		plt.gcf().clear()
		
		
	def cross_validation(self,df, label, cv=10, filename='confusion_matrix.pdf'):
		labels = df[label]
		features = df[[var for var in list(df.columns) if var != label]]

		scoring = {'accuracy': 'accuracy', 'precision': 'precision_weighted', 'recall': 'recall_weighted', 'f1':'f1_weighted'}
		scores = cross_validate(RandomForestClassifier(), features, labels, scoring=scoring, cv=cv, return_train_score=True)

		print('====== Cross-validation Train score mean ======')
		print('Accuracy: ', round(scores['train_accuracy'].mean(),ndigits=4))
		print('precision: ', round(scores['train_precision'].mean(),ndigits=4))
		print('recall: ', round(scores['train_recall'].mean(),ndigits=4))
		print('f1: ', round(scores['train_f1'].mean(),ndigits=4))

		print('====== Cross-validation Test score mean ======')
		print('Accuracy: ', round(scores['test_accuracy'].mean(),ndigits=4))
		print('precision: ', round(scores['test_precision'].mean(),ndigits=4))
		print('recall: ', round(scores['test_recall'].mean(),ndigits=4))
		print('f1: ', round(scores['test_f1'].mean(),ndigits=4))

		predictions = cross_val_predict(RandomForestClassifier(), features, labels, cv=10,verbose=1)
		precision = round(precision_score(labels,predictions, average='weighted'),ndigits=4)

		print('====================|Cross-validation prediction metrics report |====================')
		print(classification_report(labels, predictions, target_names=list(set(labels))))
		cm = confusion_matrix(labels,predictions, labels=list(labels.unique()))
		print('Confusion Matrix: \n', cm)
		self.plot_confusion_matrix(cm, precision, list(labels.unique()),filename=filename)
		

	def main(self):
		result_dir = '../results/'
		
		if not isfile(self.dataset):
			print('Error: You should inform the correct input parameters ')
			sys.exit(1)

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

		
		result_df = result_metrics.groupby(by=['n_variables'], as_index=False).agg({'model':['count']}).sort_values(by=['n_variables'])
		result_df = pd.DataFrame({'n_variables':np.array(result_df['n_variables']),
							  'model': np.array([i[0] for i in result_df['model'].values])})

		result_df = result_df[np.abs(result_df.model-result_df.model.mean()) <= (3*result_df.model.std())]

		x = np.array(result_df['n_variables'].values)
		y = np.array(result_df['model'].values)

		best_num_var = x[y.argmax()]

		self.plot_barh(x,y,'Models Vs. Number of variables', 'Models', 'Number of variables',
			  result_dir + 'model_var_' + '.pdf')

		result_df = result_metrics.groupby(by=['variables'], as_index=False).agg({'importance':['mean']}).sort_values(by=[('importance', 'mean')])
		x = np.array(result_df['variables'].values)
		y = np.array([round(i[0],4) for i in result_df['importance'].values])

		self.plot_barh(x,y,'Mean of variables importance', 'Importance', 'Variables',
			  result_dir + 'var_importance_'  + '.pdf')

		result_df = result_metrics.groupby(by=['n_variables'], as_index=False).agg({'accuracy':['mean']}).sort_values(by='n_variables')
		x = np.array(result_df['n_variables'].values)
		y = np.array([round(i[0],4) for i in result_df['accuracy'].values])

		self.plot_barh(x,y,'Accuracy of models', 'Accuracy', 'Number of variables',
			  result_dir + 'var_accuracy_'  + '.pdf')

		result_df = result_metrics[result_metrics['n_variables'] == best_num_var] # <== Filter
		result_df = result_df.groupby(by=['variables'], as_index=False).agg({'importance':['mean']}).sort_values(by=[('importance', 'mean')])
		
		print(result_df)
		
		x = np.array(result_df['variables'].values)
		y = np.array([round(i[0],4) for i in result_df['importance'].values])

		x = x[len(x)-best_num_var:len(x)]
		y = y[len(x)-best_num_var:len(x)]

		best_variables = x

		self.plot_barh(x,y,'Best variables', 'Importance', 'Variables',
			  result_dir + 'best_var_'+ '.pdf')

		sel_variables = best_variables.copy()
		dataset_var = list(result_df['variables'].values)
		dataset_var.append('Label3')
		self.cross_validation(data[dataset_var],'Label3',
					 filename=result_dir + 'confusion_matrix.pdf')
		
		if self.test is not None and self.test > 0:
			result_metrics1 = self.test_prediction(data, 'Label3', sel_variables, n_exec=int(self.test))
			result_metrics2 = self.test_prediction1(data, 'Label3', sel_variables, n_exec=int(self.test))
			
		self.plot_boxplot(result_metrics1[['accuracy', 'f1score', 'precision', 'recall']],
					 'Model evaluation metrics', 'Metrics', 'Value',
					 result_dir + 'test_accuracy_'  + '1.pdf')
		self.plot_boxplot(result_metrics2[['accuracy', 'f1score', 'precision', 'recall']],
					 'Model evaluation metrics', 'Metrics', 'Value',
					 result_dir + 'test_accuracy_'  + '2.pdf')             


if __name__=='__main__':
	select_feature=selectfeatures("../data/dataset_descriptor.csv",0.99, 0.95, 0.85,5,5)
	select_feature.main()