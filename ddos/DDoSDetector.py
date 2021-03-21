import sys
import os

from data_handler import DataHandler
from ANN import ANN

class DDoSDetector:

	data_handler = None

	neural_network = None

	def __init__(self):
		self.data_handler = DataHandler()
		self.neural_network = ANN()



	def train(self, dataset_index):
		print("Dataset: "+str(self.data_handler.get_dataset_path(dataset_index)))
		
		packets = []
		labels = []

		packets = self.data_handler.get_packet_information(dataset_index)
		labels = self.data_handler.get_labels(dataset_index)

		compressed_packets = self.data_handler.compress_packets(packets)

		input_data = self.data_handler.generate_input_data(compressed_packets)

		normalized_input, normalized_output = self.data_handler.normalize_compressed_packets(input_data, labels)
		print("Num packets: "+str(len(normalized_input)))
		print("Num labels: "+str(len(normalized_output)))

		self.neural_network.train_model(normalized_input, normalized_output)



if __name__=="__main__":

	DDoS_detector = DDoSDetector()


	DDoS_detector.train(dataset_index=1)


	
