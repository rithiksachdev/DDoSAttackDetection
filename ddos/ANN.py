import sys
import os
import time
import numpy as np
import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import LeakyReLU
from keras.layers import PReLU
from keras.layers import Bidirectional
from data_handler import DataHandler
from sklearn.metrics import confusion_matrix

class ANN:

	data_handler = None

	def __init__(self):
		self.data_handler = DataHandler()

	def train_model(self, input_data, output_data):

		train_size = 0.7

		X_train = np.array(input_data[ : int(len(input_data)*train_size)])
		y_train = np.array(output_data[ : int(len(output_data)*train_size)])
		X_test = np.array(input_data[int(len(input_data)*train_size) : ])
		y_test = np.array(output_data[int(len(output_data)*train_size) : ])


		model_path = "./Models/model.h5"


		if os.path.exists(model_path)==False:

			print("Creating neural network")

			model = Sequential()
			model.add(Dense(input_dim = len(X_train[0]), units = int(len(X_train[0])/1), kernel_initializer = 'uniform', activation = 'relu'))
			model.add(Dropout(rate = 0.2))
			model.add(Dense(units = int(len(X_train[0])/1), kernel_initializer = 'uniform', activation = 'relu'))
			model.add(Dropout(rate = 0.2))
			model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
			model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
			print("Training neural network")
			model.fit(X_train, y_train, batch_size = 20, epochs = 20)
			
			model.save(model_path)

		else:
			print("Model already exists\n")

			model = load_model(model_path)
			


		start_time = time.time()


		y_pred = model.predict(X_test)
		y_pred = (y_pred > 0.5)



		cm = confusion_matrix(y_test, y_pred)


		print("Confusion matrix: ")
		print(str(cm))

		TN = cm[0][0]
		FP = cm[0][1]
		FN = cm[1][0]
		TP = cm[1][1]

		accuracy = (TN+TP)/(TN+FP+FN+TP)

		precision = TP/(FP+TP)

		sensitivity = TP/(TP+FN)

		specificity = TN/(TN+FP)

		total = sensitivity + specificity

		print("Accuracy: "+str(accuracy))
		print("Precision: "+str(precision))
		print("Sensitivity: "+str(sensitivity))
		print("Specificity: "+str(specificity))
		print("Total: "+str(total))



if __name__=="__main__":

	neural_network = ANN()
	neural_network.train_model([], [])