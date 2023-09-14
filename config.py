import torch.nn as nn
import torch
from model import NodeFormer
from loss import FocalLoss


class Config(object):
    def __init__(self):
        self.NUM_CLASSES = 5
        self.NUM_NEIGHBOR = 1
        self.NUM_RANDOM_CHANNELS = 38
        self.EPOCH = 1000
        self.N_LAYERS = 1
        self.LR = 0.0001
        self.HIDDEN_CHANNELS = 64
        self.NUM_GUMBEL_SAMPLES = 47
        self.NUM_NODES = 738
        self.NUM_HEADS = 2
        self.GAMMA = 0.008
        self.STRUCT_MODE = False
        self.SEED = 0
        self.SCALING = 0.0
        self.RB_ORDER = 0
        self.USE_JK = True
        self.USE_EDGE_LOSS = False
        self.WEIGHT = [1.0, 1.0, 1.0, 1.0, 1.0]
        self.USE_FOCAL_LOSS = True
        self.data_path = "./ECD-Data_pan_cancer_paper.csv"
        self.meta_data_path = "./New Metadata for pan-cancer paper.xlsx"
        self.sheet_name = "Metadata clean FULL"
        self.ckpt_path = "./ckpt/best_ckpt.pt"
    
    def get_model(self, num_raw_features):
        model = NodeFormer(num_raw_features, self.HIDDEN_CHANNELS, self.NUM_CLASSES, num_layers=self.N_LAYERS, dropout=0.0, num_heads=self.NUM_HEADS, use_bn=True, nb_random_features=self.NUM_RANDOM_CHANNELS, use_gumbel=True, use_residual=True, use_act=True, use_jk=self.USE_JK, nb_gumbel_sample=self.NUM_GUMBEL_SAMPLES, rb_order=self.RB_ORDER, rb_trans='sigmoid', use_edge_loss=self.USE_EDGE_LOSS)
        return model
    
    def get_loss_function(self):
        if not self.USE_FOCAL_LOSS:
            criterion = nn.NLLLoss(weight=torch.tensor(self.WEIGHT))
        else:
            criterion = FocalLoss(gamma=self.GAMMA, alpha=self.WEIGHT)
        return criterion
