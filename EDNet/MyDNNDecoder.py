import torch
import torch.nn as nn
from Attn import Attn
import math
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv1d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv1d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv1d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class MyDNNDecoder(nn.Module):
    def __init__(self):
        super(MyDNNDecoder, self).__init__()
        self.attn = Attn('general', 64, 64).cuda()
        self.inchannel = 64
        # self.conv1 = nn.Sequential(
        #     nn.Conv1d(
        #         in_channels=64,
        #         out_channels=64,
        #         kernel_size=9,
        #         stride=1,
        #         padding=2,
        #     ),
        #     nn.BatchNorm1d(64, affine=True),
        #     nn.ReLU(),
        # )
        # self.maxpooling = nn.MaxPool1d(3)  # Adaptive

        self.layer1 = nn.LSTM(512, 1024, 1, bidirectional=True)

        self.layer2 = nn.LSTM(2048, 512, 1, bidirectional=True)

        self.layer3 = nn.LSTM(1024, 512, 1, bidirectional=True)

        self.fc = nn.Sequential(
            nn.Linear(1024, 1),  # 2048
            nn.Sigmoid(),
        )
    # def make_layer(self, block, channels, num_blocks, stride):
    #     strides = [stride] + [1] * (num_blocks - 1)
    #     layers = []
    #     for stride in strides:
    #         layers.append(block(self.inchannel, channels, stride))
    #         self.inchannel = channels
    #     return nn.Sequential(*layers)

    def forward(self, encoder_outputs, LABEL):

        # 使用attention
        # target_length = encoder_outputs.data.shape[1]
        # batchsize = encoder_outputs.data.shape[0]
        # feature_len = encoder_outputs.data.shape[2]
        # cnn_input = []
        # attn_weights = self.attn(encoder_outputs, target_length)
        # for current_index in range(target_length):
        #     context = torch.zeros(batchsize, feature_len).cuda()
        #     for i in range(batchsize):
        #         # mm = attn_weights[current_index, i, :].unsqueeze(1).repeat(1, 64)
        #         # context[i, :] = mm.mul(encoder_outputs[i, :, :]).sum(0)
        #         context[i, :] = attn_weights[current_index, i, :].matmul(encoder_outputs[i, :, :])
        #     cnn_input.append(context)
        # cnn_input = torch.stack(cnn_input).permute(1, 0, 2).cuda()
        # out, _ = self.layer1(cnn_input)

        #inputs = torch.randn(5,3,10)   ->(seq_len,batch_size,input_size)
        # out = self.conv1(cnn_input)
        # out = self.maxpooling(out)

        # 不使用attention
        out, _ = self.layer1(encoder_outputs.cuda())
        # 再弄一层LSTM
        out, _ = self.layer2(out.cuda())

        out, _ = self.layer3(out.cuda())

        if LABEL is not None:
            LABEL = LABEL.unsqueeze(2)
            out_bg = out.mul(LABEL)
        else:
            out_bg = None

            # 在low rank 层上再弄一层LSTM
        # out, _ = self.layer2(out.cuda())

        out = out.permute(1, 0, 2).cuda()
        # U, S, V = torch.svd(X)

        # out = self.layer2(out)
        # out = self.layer3(out)
        # out = self.avgpooling(out)
        # out = out.view(out.size(0), -1).unsqueeze(1)
        # out = F.dropout(out, 0.1, training=self.training)
        # out = F.relu(out)
        # out = self.maxpooling(out)

        output = self.fc(out) #.squeeze(1)
        output = output.permute(1, 0, 2).cuda()
        output = output.squeeze()
        return output, out_bg

