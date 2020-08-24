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
        self.size0 = size0  # 128
        self.size1 = size1  # 128

        if self.method == 'general':
            self.attn = nn.Linear(self.size0, self.size1)
        # elif self.method == 'concat':
        #     self.attn = nn.Linear(self.hidden_size*2, self.hidden_size)
        #     self.other = nn.Parameter(torch.FloatTensor(1, self.hidden_size))

    def forward(self, encoder_outputs, current_index, encoder_outputs_neighbor, neighbor_num):

        # attn_energies = Variable(torch.zeros(seq_len))
        temp = encoder_outputs[:, current_index, :]
        attn_energies = self.score(encoder_outputs_neighbor, temp, neighbor_num)
        # attn_energies[:, current_index] = 0
        attn_energies = F.softmax(attn_energies, dim=1)   # log
        return attn_energies

    def score(self, encoder_outputs_neighbor, temp, neighbor_num):
        # if self.method == 'dot':
        #     energy = hidden.dot(encoder_output)
        #     return energy

        batchsize = encoder_outputs_neighbor.data.shape[0]

        # encoder_outputs_neighbor = encoder_outputs[mtix4sp]
        if self.method == 'general':
            energy = self.attn(encoder_outputs_neighbor)
            attn_energy = Variable(torch.zeros(batchsize, neighbor_num)).cuda()
            for k in range(batchsize):
                temp_inbatch = temp[k, :].unsqueeze(1)
                temp_copy = torch.Tensor(64, neighbor_num).cuda()
                temp_copy.copy_(temp_inbatch)
                attn_energy[k, :] = torch.diag(energy[k, :, :].mm(temp_copy))


            # temp_copy_batch = torch.Tensor(batchsize, 64, neighbor_num).cuda()
            # for k in range(batchsize):
            #     temp_inbatch = temp[k, :].unsqueeze(1)
            #     temp_copy = torch.Tensor(64, neighbor_num).cuda()
            #     temp_copy.copy_(temp_inbatch)
            #     temp_copy_batch[k,:,:] = temp_copy
            # bbb = torch.bmm(energy, temp_copy_batch)
            # for k in range(batchsize):
            #     attn_energy[k, :] = torch.diag(bbb[k,:,:])

        return attn_energy

        # if self.method == 'general':
        #     energy = self.attn(temp)
        #     hidden = hidden.squeeze(0)
        #     energy = hidden.dot(energy)
        #     return energy

        # elif self.method == 'concat':
        #     energy = self.attn(torch.cat((hidden, encoder_output), 1))
        #     energy = self.other.dot(energy)
        #     return energy



















