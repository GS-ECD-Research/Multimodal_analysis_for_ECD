import pandas as pd
import numpy as np
import torch
from sklearn.neighbors import kneighbors_graph
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops
from config import Config

class Dataset(Config):
    def __init__(self):
        super(Dataset, self).__init__()
    
    def get_dataset(self):
        class_dict = {"Breast cancer":0, "CRC cancer":1, "Gastric cancer":2, "Liver cancer":3, "Lung cancer":4}
        data = pd.read_csv(self.data_path)
        meta_data = pd.read_excel(self.meta_data_path, sheet_name=self.sheet_name)
        
        train_sample = meta_data[meta_data["Group"] == "Discovery"]["SampleID"].values
        train_data_array = []
        y = []
        
        for sample_name in train_sample:
            class_name = meta_data[meta_data["SampleID"] == sample_name]["Label"].values[0]
            if class_name in class_dict:
                train_data_array.append(data[data["SampleID"]==sample_name].values[0, 1:])
                y.append(class_dict[class_name])
        train_data_array = np.array(train_data_array)
        y = np.array(y)

        test_sample = meta_data[meta_data["Group"] == "Validation"]["SampleID"].values
        y_test = []
        test_data_array = []

        for sample_name in test_sample:
            class_name = meta_data[meta_data["SampleID"] == sample_name]["Label"].values[0]
            gender = meta_data[meta_data["SampleID"] == sample_name]["Gender"].values[0]
            if gender == "Female":
                gender_enc = 0
            else:
                gender_enc = 1
            if class_name in class_dict:
                test_data_array.append(data[data["SampleID"]==sample_name].values[0, 1:])
                y_test.append([class_dict[class_name], gender_enc])
        test_data_array = np.array(test_data_array)
        y_test = np.array(y_test)

        train_data_array = train_data_array.astype("float32")
        test_data_array = test_data_array.astype("float32")

        graph_data = np.concatenate((train_data_array, test_data_array), axis=0)
        train_idx_ori = np.arange(len(train_data_array))
        test_idx = np.arange(len(test_data_array)) + len(train_data_array)
        add_id = np.concatenate((test_idx[-89:-84], test_idx[110:120], test_idx[67:71], test_idx[-12:]))

        #####
        train_idx = np.concatenate((train_idx_ori[:146], train_idx_ori[156:262-7], train_idx_ori[262:329-4], train_idx_ori[329:406-4], train_idx_ori[406:499-5], add_id))
        val_idx = np.concatenate((train_idx_ori[146:156], train_idx_ori[255:262], train_idx_ori[325:329], train_idx_ori[402:406], train_idx_ori[494:]))
        #####

        torch.manual_seed(self.SEED)
        np.random.seed(self.SEED)

        adj_knn = kneighbors_graph(graph_data, n_neighbors=self.NUM_NEIGHBOR, include_self=True)
        edge_index = torch.tensor(adj_knn.nonzero(), dtype=torch.long)
        adjs = []
        adj, _ = remove_self_loops(edge_index)
        adj, _ = add_self_loops(adj, num_nodes=self.NUM_NODES)
        adjs.append(adj)
        graph_feats = torch.from_numpy(graph_data).float()
        true_label = torch.from_numpy(np.concatenate((y, y_test[:, 0]), axis=0)).long()
        gender = y_test[:, 1]
        return graph_feats, train_idx, val_idx, test_idx, adjs, true_label, gender