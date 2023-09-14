import numpy as np
import random
import torch
from config import Config
from loader import Dataset

class Inferer(Config):
    def infer(self, model, graph_feats, adjs, true_label, test_idx, gender):
        model.eval()
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)
        out = model(graph_feats, adjs, 0.25)
        pred = out[test_idx]
        pred_ind = torch.topk(pred, 2, dim=1).indices
        y_pred = []
        for i, node_pred_ind in enumerate(pred_ind):
            if node_pred_ind[0]==0 and gender[i] == 1:
                y_pred.append(node_pred_ind[1])
            else:
                y_pred.append(node_pred_ind[0])
        y_pred = torch.tensor(y_pred)

        true_label_test = true_label[test_idx]
        test_correct = y_pred == true_label_test
        test_acc = int(test_correct.sum()) / int(len(true_label_test))
        from sklearn.metrics import multilabel_confusion_matrix
        mcm_test = multilabel_confusion_matrix(true_label_test, y_pred)    
        tps_test = mcm_test[:, 1, 1]
        tns_test = mcm_test[:, 0, 0]
        recall_test  = tps_test / (tps_test + mcm_test[:, 1, 0])         # Sensitivity
        specificity_test = tns_test / (tns_test + mcm_test[:, 0, 1])         # Specificity
        # precision_test   = tps_test / (tps_test + mcm_test[:, 0, 1])         # PPV
        print("Acc:", test_acc)
        print("Sensitivity:", recall_test)
        print("Specificity:", specificity_test)
    
    def run_once(self):
        dataset = Dataset()
        graph_feats, _, _, test_idx, adjs, true_label, gender = dataset.get_dataset()
        model = torch.load(self.ckpt_path)
        self.infer(model, graph_feats, adjs, true_label, test_idx, gender)


if __name__ == "__main__":
    inferer = Inferer()
    inferer.run_once()