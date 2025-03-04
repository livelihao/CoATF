import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, out_dim=None, n_head=1, score_function='dot_product', dropout=0):
        ''' Attention Mechanism
        :param embed_dim:
        :param hidden_dim:
        :param out_dim:
        :param n_head: num of head (Multi-Head Attention)
        :param score_function: scaled_dot_product / mlp (concat) / bi_linear (general dot)
        :return (?, q_len, out_dim,)
        '''
        super(Attention, self).__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // n_head # 10
        if out_dim is None:
            out_dim = embed_dim # 10
        self.embed_dim = embed_dim # 10
        self.hidden_dim = hidden_dim # 10
        self.n_head = n_head # 1
        self.score_function = score_function # 'mlp'
        self.w_k = nn.Linear(embed_dim, n_head * hidden_dim)
        self.w_q = nn.Linear(embed_dim, n_head * hidden_dim)
        self.proj = nn.Linear(n_head * hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        if score_function == 'mlp':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim*2))
        elif self.score_function == 'bi_linear':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        else:  # dot_product / scaled_dot_product
            self.register_parameter('weight', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_dim)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)

    def forward(self, k, q):
        if len(q.shape) == 2:  # q_len missing
            q = torch.unsqueeze(q, dim=1) # q.shape bs, 1, 10
        if len(k.shape) == 2:  # k_len missing
            k = torch.unsqueeze(k, dim=1) # k.shape bs, 1, 10
        mb_size = k.shape[0]  # 64
        k_len = k.shape[1] # 1
        q_len = q.shape[1] # 1
        # print(q.shape)  # 64, 1, 32

        # k: (?, k_len, embed_dim,) 1  32
        # q: (?, q_len, embed_dim,) 1, 32
        # kx: (n_head*?, k_len, hidden_dim)  8*64， 1， 4
        # qx: (n_head*?, q_len, hidden_dim)
        # score: (n_head*?, q_len, k_len,)
        # output: (?, q_len, out_dim,)
        kx = self.w_k(k).view(mb_size, k_len, self.n_head, self.hidden_dim) # bs,1,1,32

        kx = kx.permute(2, 0, 1, 3).contiguous().view(-1, k_len, self.hidden_dim) # bs,1,1,32
        # print(kx.shape) (n_head*?, k_len, hidden_dim) 8*64， 1， 4
        # exit()
        qx = self.w_q(q).view(mb_size, q_len, self.n_head, self.hidden_dim) # bs,1,1,32
        qx = qx.permute(2, 0, 1, 3).contiguous().view(-1, q_len, self.hidden_dim) # bs,1,1,32

        kt = kx.permute(0, 2, 1) # bs,32,1
        score = torch.bmm(kt,qx) # bs, 32, 32
        score = F.softmax(score, dim=-1) # bs, 32, 23

        output = torch.bmm(kx,score) # bs, 1, 32

        output = self.proj(output)  # (?, q_len, out_dim)
        output = self.dropout(output) # bs, 1, 32
        return output, score