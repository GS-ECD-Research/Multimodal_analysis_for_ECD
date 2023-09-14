import numpy as np
import sys
import os
import torch
import torch.nn.functional as F
from config import Config
from loader import Dataset

class Trainer(Config):
    def valid(self, model, graph_feats, adjs, true_label, val_idx):
        model.eval()
        if self.USE_EDGE_LOSS:
            out, _ = model(graph_feats, adjs, 0.25)
        else:
            out = model(graph_feats, adjs, 0.25)
        pred = out[val_idx]
        y_pred = pred.argmax(dim=1)  # Use the class with highest probability.
        
        cancer_dict = {}
        val_correct = y_pred == true_label[val_idx]  # Check against ground-truth labels.
        cancer_dict[0] = (y_pred[true_label[val_idx]==0] == 0).sum()/((true_label[val_idx] == 0).sum())
        cancer_dict[1] = (y_pred[true_label[val_idx]==1] == 1).sum()/((true_label[val_idx] == 1).sum())
        cancer_dict[2] = (y_pred[true_label[val_idx]==2] == 2).sum()/((true_label[val_idx] == 2).sum())
        cancer_dict[3] = (y_pred[true_label[val_idx]==3] == 3).sum()/((true_label[val_idx] == 3).sum())
        cancer_dict[4] = (y_pred[true_label[val_idx]==4] == 4).sum()/((true_label[val_idx] == 4).sum())
        val_acc = int(val_correct.sum()) / int(len(true_label[val_idx]))  # Derive ratio of correct predictions.
        return (val_acc, y_pred, out, cancer_dict)

    def run_once(self):
        dataset = Dataset()
        graph_feats, train_idx, val_idx, _, adjs, true_label, _ = dataset.get_dataset()
        cfg = Config()
        criterion = cfg.get_loss_function()
        model = cfg.get_model(graph_feats.shape[1])
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=5e-3, lr=self.LR)
        for epoch in range(self.EPOCH):
            model.train()
            optimizer.zero_grad()
            if self.USE_EDGE_LOSS:
                out, link_loss_ = model(graph_feats, adjs, 0.25)
            else:
                out = model(graph_feats, adjs, 0.25)
            if not self.USE_FOCAL_LOSS:
                out = F.log_softmax(out, dim=1)
            loss = criterion(out[train_idx], true_label[train_idx])
            if self.USE_EDGE_LOSS:
                loss -= 0.1 * sum(link_loss_) / len(link_loss_)
            loss.backward()
            optimizer.step()
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
            val_acc, _, _, _ = self.valid(model, graph_feats, adjs, true_label, val_idx)
            if epoch == 492:
                print("val_acc:", val_acc)
                if not os.path.exists("./ckpt"):
                    os.makedirs("./ckpt")
                torch.save(model, self.ckpt_path)
            else:
                print("val_acc:", val_acc) 

if __name__ == "__main__":
    trainer = Trainer()
    trainer.run_once()