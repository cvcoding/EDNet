#!/usr/bin/env python
# encoding: utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
class Attn(nn.Module):
    def __init__(self, method, size0, size1):
        super(Attn, self).__init__()
        self.method = method
        self.size0 = size0 #128
        self.size1 = size1 #128
        if self.method == 'general':
            self.attn = nn.Linear(self.size0, self.size1)
        # elif self.method == 'concat':
        #     self.attn = nn.Linear(self.hidden_size*2, self.hidden_size)
        #     self.other = nn.Parameter(torch.FloatTensor(1, self.hidden_size))
    def forward(self, encoder_outputs, target_length):
        batchsize = encoder_outputs.data.shape[0]
        attn_energies = Variable(torch.zeros(target_length, batchsize, target_length)).cuda()
        for current_index in range(target_length):
            attn_energies_oneindex = self.score(encoder_outputs, current_index,batchsize)
            attn_energies_oneindex = F.softmax(attn_energies_oneindex, dim=1)
            attn_energies[current_index, :, :] = attn_energies_oneindex
        return attn_energies
    def score(self, encoder_outputs, current_index, batchsize):
        temp = encoder_outputs[:, current_index, :]
        other_num = encoder_outputs.data.shape[1]
        if self.method == 'general':
            energy = self.attn(encoder_outputs)
            attn_energy = Variable(torch.zeros(batchsize, other_num)).cuda()
            for k in range(batchsize):
                temp_inbatch = temp[k, :].unsqueeze(1)
                temp_copy = torch.Tensor(64, other_num).cuda()
                temp_copy.copy_(temp_inbatch)
                attn_energy[k, :] = torch.diag(energy[k, :, :].mm(temp_copy))
                attn_energy[k, current_index] = 0
        return attn_energy
