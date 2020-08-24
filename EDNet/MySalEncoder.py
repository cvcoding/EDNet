import torch.nn as nn
import torch
import math
import torch.autograd as autograd
from torch.nn import functional as F
from network import resnet34, resnet50, resnet101
import numpy as np
import random


class ROIHead(nn.Module):
    def __init__(self, roi_size=(2, 2), feat_stride=4):
        super(ROIHead, self).__init__()
        self.feat_stride = feat_stride
        # roi层原理和SSPNet类似,可以把不同大小的图片pooling成大小相同的矩阵
        self.roi_pooling = nn.AdaptiveMaxPool2d(roi_size)
        self.bn = nn.BatchNorm1d(180, affine=True)  #
        # self.bn = nn.BatchNorm1d(135, 512 * 2 * 2)

        self.relulayer1 = nn.ReLU()
        self.dnn_layer1 = nn.Linear(1024 * 3 * 3, 512)  # 512for34  2048forresnet50

        self.bn2 = nn.BatchNorm1d(180, affine=True)
        # self.bn2 = nn.BatchNorm1d(135, 64)

        # how to do norm +++++++++++++++++++++++++++++++++++++++++++++++++???
        self.normlayer = nn.BatchNorm1d(180, 512)

        self.relulayer2 = nn.ReLU()
        self.dnn_layer2 = nn.Linear(512, 64)

        # self.bn3 = nn.BatchNorm1d(1, momentum=0.5)

        self.normlayer2 = nn.BatchNorm1d(180, 64)


    def normal_init(self, m, mean, stddev, truncated=False):
        """
        weight initalizer: truncated normal and random normal.
        """
        # x is a parameter
        if truncated:
            m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
        else:
            m.weight.data.normal_(mean, stddev)
            m.bias.data.zero_()

    def normal_init_sparse(self, m):
        """
        weight initalizer: truncated normal and random normal.
        """
        m.bias.data.zero_()
        value_list = [math.sqrt(2), 0, -math.sqrt(2)]
        probabilities = [0.25, 0.5, 0.25]
        s_w = m.weight.size()
        for s_w_i in range(s_w[0]):
            for s_w_j in range(s_w[1]):
                m.weight[s_w_i, s_w_j] = self.rand_pick(value_list, probabilities)

    def rand_pick(seq, probabilities):
        x = random.uniform(0, 1)
        cumprob = 0.0
        for item, item_pro in zip(seq, probabilities):
            cumprob += item_pro
            if x < cumprob:
                break
        return item

    def get_roi_feature(self, x, rois):
        # 获取roi对应feature map上的位置r
        r = rois/self.feat_stride
        r[:, :2] = np.trunc(r[:, :2])
        r[:, 2:] = np.ceil(r[:, 2:])
        r = r.astype(np.int)

        # r[:, :2] = torch.trunc(r[:, :2])
        # r[:, 2:] = torch.ceil(r[:, 2:])
        # r = torch.ceil(r)

        roi_features = []
        for i in range(rois.shape[0]):
            roi_features.append(x[:, r[i, 0]:r[i, 2]+1, r[i, 1]:r[i, 3]+1])

        return roi_features, len(roi_features)

    def forward(self, features, rois):
        '''
        :param features: 4D feature map, [1,c,h,w]
        :param rois: [n_rois,4]
        :param feat_stride: be used to crop rois_features from feature map
        :return:
        '''
        x = torch.zeros(features.size()).copy_(features.data)
        # x = features[0]
        batch_total = []
        for i in range(rois.shape[0]):
            rois_img, batch_size = self.get_roi_feature(x[i], rois[i])

            batch = []
            for f in rois_img:
                batch.append(self.roi_pooling(f))
            batch = torch.stack(batch)
            batch = batch.view(batch_size, -1)
            batch_total.append(batch)
        # print('batch_total: {:.1f}'.format(rois.shape[0]))
        batch_total = torch.stack(batch_total).cuda()
        batch_total = self.bn(batch_total)
        batch_total = self.relulayer1(batch_total)
        # batch_total = F.dropout(batch_total, 0.1, training=self.training)
        batch = self.dnn_layer1(batch_total.cuda())
        # batch = self.bn2(batch)

        return batch


class MySalEncoder(nn.Module):

    def __init__(self):
        super(MySalEncoder, self).__init__()
        # the image is downsampled by resnet with 4 times, so feat_stride is set to 4 here.
        self.feat_stride = 3.5  #  1.75
        model_path = './models/'
        resnet = resnet50(pretrained=True, modelpath=model_path, num_classes=1000)
        # the extractor (resnet) need not to be updated, so only roi_head in encoder is updated.
        # in demo "encoder_optimizer", the parameters in self.roi_head is inserted into computing graph,
        # encoder_optimizer = optim.Adam(EncoderModel.roi_head.parameters(), lr=0.001, ...
        # ...betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

        self.extractor = nn.Sequential(resnet)

        self.dconv = nn.ConvTranspose2d(1024, 1024, kernel_size=3, stride=2)
        self.dconv.cuda()
        self.relu = nn.ReLU(inplace=True)

        self.roi_head = ROIHead(roi_size=(3, 3), feat_stride=self.feat_stride).cuda()
        self.extractor.cuda()
        self.roi_head.cuda()

    def forward(self, img, roi):
        features = self.extractor(img)

        features = self.relu(features)
        features = self.dconv(features)

        sp_features = self.roi_head(
            features,
            roi)
        return sp_features
