import os
import os.path
import csv
from os import listdir
from os.path import isfile, join
import time
import datetime
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import joblib
from pcap_handler import PCAPHandler

class DataHandler:

    pcap_handler = None
    base_path = "./Datasets/"
    label_path = "./Labels/"
    IP_type = {}
    datasets = []
    pcap_files = []
    initial_labels = {}

    def __init__(self):
        self.pcap_handler = PCAPHandler()
        self.load_list_of_datasets()
        self.load_list_of_pcaps()
        self.IP_type[0] = "Hop-by-Hop Option Header"
        self.IP_type[1] = "ICMP"
        self.IP_type[2] = "ROUTER"
        self.IP_type[6] = "TCP"
        self.IP_type[17] = "UDP"
        self.IP_type[58] = "ICMPv6"

    def get_packet_information(self, dataset_index, pcap_index=None):

        if pcap_index!=None:
            pcap_path = self.get_pcap_path(dataset_index, pcap_index)
            print("Reading pcap "+str(pcap_path))
            data = self.pcap_handler.read_pcap(pcap_path)
            print("Finished reading pcap file")
            return data

        else:
            pcap_paths = []
            for x in range(0, len(self.pcap_files[dataset_index])):
                pcap_paths.append(self.get_pcap_path(dataset_index, x))
            pcap_information = []

            for x in range(0, len(pcap_paths)):
                pcap_path = pcap_paths[x]
                print("Reading pcap "+str(pcap_path))
                data = self.pcap_handler.read_pcap(pcap_path)
                print("Finished reading pcap file")
                print()
                pcap_information.extend(data)
            total_packets = len(pcap_information)
            print("Total number of packets: "+str(total_packets))
            return pcap_information
 
    def compress_packets(self, packets):
	    print("compress_packets()")
	    new_packets = []
	    for x in range(0, len(packets)):
		    packet = packets[x]
		    compressed_packet = self.pcap_handler.compress_packet(packet)
		    new_packets.append(compressed_packet)
	    return new_packets

    def compress_labels(self, labels):
	    return labels

    def generate_input_data(self, compressed_packets):

        ethernet_source_addresses = {}
        ethernet_destination_addresses = {}

        IP_source_addresses = {}
        IP_destination_addresses = {}
        IP_type = {}

        def count_occurrences(value, cur_index, lookback_amount, lookback_column):
            num = 0
            for x in range(cur_index, max(cur_index-lookback_amount, -1), -1):
                if compressed_packets[x][lookback_column]==value:
                    num+=1
            return num

        to_return = []
        for x in range(0, len(compressed_packets)):
            row = []
            ethernet_source = compressed_packets[x][1]
            ethernet_destination = compressed_packets[x][2]
            IP_source = compressed_packets[x][7]
            IP_destination = compressed_packets[x][8]

            lookback_amount = 100
            ethernet_source_occurrences = count_occurrences(value=ethernet_source, cur_index=x, lookback_amount=lookback_amount, lookback_column=1)
            ethernet_destination_occurrences = count_occurrences(value=ethernet_destination, cur_index=x, lookback_amount=lookback_amount, lookback_column=2)
            IP_source_occurrences = count_occurrences(value=IP_source, cur_index=x, lookback_amount=lookback_amount, lookback_column=7)
            IP_destination_occurrences = count_occurrences(value=IP_destination, cur_index=x, lookback_amount=lookback_amount, lookback_column=8)
            row.append(ethernet_source_occurrences)
            row.append(ethernet_destination_occurrences)
            row.append(IP_source_occurrences)
            row.append(IP_destination_occurrences)

            lookback_amount = 1000
            ethernet_source_occurrences = count_occurrences(value=ethernet_source, cur_index=x, lookback_amount=lookback_amount, lookback_column=1)
            ethernet_destination_occurrences = count_occurrences(value=ethernet_destination, cur_index=x, lookback_amount=lookback_amount, lookback_column=2)
            IP_source_occurrences = count_occurrences(value=IP_source, cur_index=x, lookback_amount=lookback_amount, lookback_column=7)
            IP_destination_occurrences = count_occurrences(value=IP_destination, cur_index=x, lookback_amount=lookback_amount, lookback_column=8)
            row.append(ethernet_source_occurrences)
            row.append(ethernet_destination_occurrences)
            row.append(IP_source_occurrences)
            row.append(IP_destination_occurrences)

            timestamp = compressed_packets[x][3]
            prev_timestamp1 = compressed_packets[max(0, x-1)][3]
            prev_timestamp10 = compressed_packets[max(0, x-10)][3]
            prev_timestamp100 = compressed_packets[max(0, x-100)][3]
            prev_timestamp1000 = compressed_packets[max(0, x-1000)][3]
            timestamp_difference1 = timestamp-prev_timestamp1
            timestamp_difference10 = timestamp-prev_timestamp10
            timestamp_difference100 = timestamp-prev_timestamp100
            timestamp_difference1000 = timestamp-prev_timestamp1000
            row.append(timestamp_difference1)
            row.append(timestamp_difference10)
            row.append(timestamp_difference100)
            row.append(timestamp_difference1000)
            to_return.append(row)

        return to_return

    def normalize_compressed_packets(self, compressed_packets, labels):
        unnormalized_input_data = compressed_packets.copy()
        unnormalized_output_data = labels.copy()
        input_data = compressed_packets
        output_data = labels
        input_scaler_filename = "./Models/input_normalization_params.saver"

        if os.path.isfile(input_scaler_filename):
            print("Loaded scaler object file")
            input_scaler = joblib.load(input_scaler_filename)
            input_data = input_scaler.transform(input_data)
        else:
            input_scaler = StandardScaler()
            try:
                input_data = input_scaler.fit_transform(input_data)
                input_data = np.array(input_data).tolist()
            except Exception as error:
                print(error)
            joblib.dump(input_scaler, input_scaler_filename)

        output_data = []
        for x in range(0, len(labels)):
            if labels[x][1]=="BENIGN":
                output_data.append(0)
            else:
                output_data.append(1)

        return input_data, output_data

    def get_labels(self, dataset_index, pcap_index=None):

        labels = []
        if pcap_index!=None:
            print("pcap_index!=None")
            label_path = self.get_label_path(dataset_index, pcap_index)
            print("Label path: "+str(label_path))
            labels = self.read_from_csv(label_path)
        else:
            labels = []
            for x in range(0, len(self.pcap_files[dataset_index])):
                label_path = self.get_label_path(dataset_index, x)
                print("Label path: "+str(label_path))
                temp_labels = self.read_from_csv(label_path)
                labels.extend(temp_labels)

        return labels

    def calculate_labels(self, dataset_index):

        pcap_names = self.pcap_files[dataset_index]
        dataset_name = self.datasets[dataset_index]
        for x in range(0, len(pcap_names)):

            labels = self.calculate_labels_helper(dataset_index, x)
            new_label_path = self.label_path+"/"+str(dataset_name)+"/"+str(pcap_names[x])+".csv"
            self.save_to_csv(new_label_path, labels)
        
    def calculate_labels_helper(self, dataset_index, pcap_index):

        if dataset_index not in self.initial_labels.keys() or len(self.initial_labels[dataset_index])==0:
            self.initial_labels[dataset_index] = self.read_initial_labels(dataset_index)

        initial_labels = self.initial_labels[dataset_index]

        dataset_name = self.datasets[dataset_index]

        labels_folder = self.label_path+"/"+str(dataset_name)
        if os.path.exists(labels_folder)==False:
            os.mkdir(labels_folder)
            dataset_contents = self.get_packet_information(dataset_index, pcap_index)
            label_dictionary = {}
            for x in range(0, len(initial_labels)):
                source = initial_labels[x]['source']
                destination = initial_labels[x]['destination']

                if source not in label_dictionary.keys():
                    label_dictionary[source] = {}

                if destination not in label_dictionary[source].keys():
                    label_dictionary[source][destination] = []

                label_dictionary[source][destination].append(initial_labels[x])


        def get_label(packet):
            label = "BENIGN"

            packet_source = packet['IP']['source']
            packet_destination = packet['IP']['destination']
            labels_with_same_source = []
            try:
                labels_with_same_source = label_dictionary[packet_source][packet_destination]
            except Exception as error:
                print("No label for source "+str(packet_source)+" and destination "+str(packet_destination))
                return label
            packet_type = packet['IP']['packet_type'] 
            packet_timestamp = int(packet['Ethernet']['timestamp']) #format is unix timestamp
            try:
                packet_source_port = packet['packet_info']['source_port'] 
                packet_destination_port = packet['packet_info']['destination_port'] 
            except Exception as error:
                print("Error "+str(error))
                return label
            packet_timestamp += 60*60*4

            
            for x in range(0, len(labels_with_same_source)):
                if packet_type==self.IP_type[labels_with_same_source[x]['protocol']] and\
                    packet_source_port==labels_with_same_source[x]['source_port'] and\
                    packet_destination_port==labels_with_same_source[x]['destination_port']:

                    label_timestamp = time.mktime(datetime.datetime.strptime(labels_with_same_source[x]['timestamp'], "%d/%m/%Y %I:%M").timetuple())

                    if abs(packet_timestamp-label_timestamp)<=600:
                        label = labels_with_same_source[x]['label']
                        break
            return label


        new_labels = []
        for x in range(0, len(dataset_contents)):
            row = []
            row.append(x)

            print("At packet "+str(x))
            if len(dataset_contents[x]['Ethernet'])>0:
                overall_type = dataset_contents[x]['Ethernet']['overall_type']
            else:
                row.append("BENIGN")
                new_labels.append(row)
                continue
            label = "BENIGN"
            if overall_type=="IP":
                label = get_label(dataset_contents[x])
            row.append(label)
            new_labels.append(row)

        return new_labels

    def read_initial_labels(self, dataset_index):
        if dataset_index<0 or dataset_index >= len(self.datasets):
            print("Invalid dataset index in get_dataset_path()")
            return ""

        dataset_name = self.datasets[dataset_index]
        label_path = self.label_path+"/"+str(dataset_name)+".csv"

        print("label path: "+str(label_path))
        contents = self.read_from_csv(label_path)
        contents.pop(0)

        print("Num packets: "+str(len(contents)))
        new_contents = []

        for x in range(0, len(contents)):

            row = {}
            row['source'] = contents[x][1]
            row['source_port'] = int(contents[x][2])
            row['destination'] = contents[x][3]
            row['destination_port'] = int(contents[x][4])
            row['protocol'] = int(contents[x][5])
            row['timestamp'] = contents[x][6]
            row['label'] = contents[x][-1]
            new_contents.append(row)

        return new_contents

    def read_from_csv(self, path):
        if os.path.isfile(path):
            with open(path, newline='') as file:
                contents = csv.reader(file)
                temp_list=[]
                for row in contents:
                    temp_matrix=[]
                    for stuff in row:
                            temp_matrix.append(stuff)
                    temp_list.append(temp_matrix)

                return temp_list
        else:
            return []

    def save_to_csv(self, path, data):
        with open(path, 'w', newline='') as file:
            contents = csv.writer(file)
            contents.writerows(data)


    def get_dataset_path(self, dataset_index):

        dataset_name = self.get_dataset_filename(dataset_index)
        if dataset_name=="":
            return ""
        return self.base_path+"/"+str(dataset_name)

    def get_dataset_filename(self, dataset_index):
        if dataset_index<0 or dataset_index >= len(self.datasets):
            print("Invalid dataset index")
            return ""

        dataset_name = self.datasets[dataset_index]

        return dataset_name

    def get_pcap_path(self, dataset_index, pcap_index):
        dataset_name = self.get_dataset_filename(dataset_index)
        pcap_name = self.get_pcap_filename(dataset_index, pcap_index)

        if dataset_name=="":
            return ""
        if pcap_name=="":
            return ""

        return self.base_path+"/"+str(dataset_name)+"/"+str(pcap_name)+".pcap"
		
    def get_pcap_filename(self, dataset_index, pcap_index):
        if dataset_index<0 or dataset_index >= len(self.pcap_files):
            print("Invalid dataset index")
            return ""
        if pcap_index<0 or pcap_index>= len(self.pcap_files[dataset_index]):
            print("Invalid pcap file index")
            return ""

        pcap_filename = self.pcap_files[dataset_index][pcap_index]
        return pcap_filename

    def get_label_path(self, dataset_index, pcap_index):
        dataset_name = self.get_dataset_filename(dataset_index)
        pcap_name = self.get_pcap_filename(dataset_index, pcap_index)
        if dataset_name=="":
            return ""
        if pcap_name=="":
            return ""

        return self.label_path+"/"+str(dataset_name)+"/"+str(pcap_name)+".csv"

    def load_list_of_datasets(self):
        path = self.base_path
        only_folders = [f for f in listdir(path) if not isfile(join(path, f))]
        only_folders.sort()
        self.datasets = []
        for folder in only_folders:
            self.datasets.append(folder)

    def load_list_of_pcaps(self):
        path = self.base_path
        self.pcap_files = []
        for x in range(0, len(self.datasets)):
            dataset_path = path+"/"+self.datasets[x]
            only_files = [f for f in listdir(dataset_path) if isfile(join(dataset_path, f))]

            pcap_list = []
            for file in only_files:
                if ".pcap" in file:
                    pcap_list.append(file.replace(".pcap", ""))
            pcap_list.sort()
            self.pcap_files.append(pcap_list)

if __name__=="__main__":

    pcap_handler = PCAPHandler()
    data_handler = DataHandler()
    data_handler.calculate_labels(1)