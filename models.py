import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import scipy.sparse as sp
from layers import Attention
import torch.utils.data as data
from utils import RegLoss

class COATF(nn.Module):
    def __init__(self, user_num, item_num, cont_num, factor_num, dropout, nonlinear=None, user_rated_items={}, nc=32):
        super(COATF, self).__init__()
        self.user_rated_items, self.user_rated_items_n = self._initialize_user_rated_items(user_rated_items)
        self.factor_num = factor_num

        self.embed_user = nn.Embedding(user_num, factor_num)
        self.embed_item = nn.Embedding(item_num, factor_num)
        self.embed_cont = nn.Embedding(cont_num, factor_num)
        self.embed_Y = nn.Embedding(item_num + 1, factor_num, padding_idx=0)

        self.conv11 = nn.Conv2d(in_channels=1, out_channels=nc, kernel_size=(1, 4))
        self.conv12 = nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=(factor_num, 1))

        self.att = Attention(nc, out_dim=nc, n_head=1, score_function="mlp", dropout=dropout)

        self.predict_layer1 = nn.Linear(nc, 1)
        self.dropout = nn.Dropout(p=dropout)

        self._init_weights()

    def _initialize_user_rated_items(self, user_rated_items):
        user_num = max(user_rated_items.keys()) + 1 if user_rated_items else 0
        item_num = max(max(user_items) for user_items in user_rated_items.values()) + 1 if user_rated_items else 0
        
        user_rated_items_matrix = torch.zeros((user_num + 1, item_num + 1), dtype=torch.long)
        user_rated_items_count = torch.zeros((user_num + 1, item_num + 1), dtype=torch.long)

        for u, items in user_rated_items.items():
            for i in items:
                user_rated_items_matrix[u, i] = 1
                user_rated_items_count[u, i] += 1

        return user_rated_items_matrix, user_rated_items_count

    def _init_weights(self):
        # Initialize weights for all modules
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=1, nonlinearity='relu')

    def forward(self, user, item, cont):
        embed_user = self.embed_user(user)
        embed_item = self.embed_item(item)
        embed_cont = self.embed_cont(cont)

        # Ensure that the user ID is in the valid range
        user = user.clamp(0, self.user_rated_items.shape[0] - 1)
        # Getting the history of user interactions
        Iu = torch.gather(self.user_rated_items, 0, user.unsqueeze(1).expand(-1, self.user_rated_items.shape[1]))
        Iu_n = torch.gather(self.user_rated_items_n, 0, user.unsqueeze(1).expand(-1, self.user_rated_items_n.shape[1]))

        y_embed = self.embed_Y(Iu)
        Y_sum = torch.sum(y_embed, dim=1)
        Iu_n = Iu_n.sum(dim=1, keepdim=True)  # Make sure the shape of Iu_n matches Y_sum
        # void division by zero
        Iu_n[Iu_n == 0] = 1

        Y_avg = Y_sum / Iu_n 

        interaction = torch.cat((embed_user, embed_item, embed_cont, Y_avg), dim=1)
        x = interaction.view(interaction.shape[0], 1, self.factor_num, 4)
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = x.view(x.shape[0], x.shape[1])

        x, _ = self.att(x, x)

        prediction = self.predict_layer1(x)

        return prediction.view(-1)