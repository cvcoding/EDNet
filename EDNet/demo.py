# try to build a lstm network
import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from MyDNNDecoder import MyDNNDecoder
from MySalEncoder import MySalEncoder
import torch.nn.utils as utils
import skimage
import skimage.io
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage import transform,data
from torchvision import datasets,transforms
import networkx as nx
import scipy.spatial.distance
import scipy.signal
import math
import copy
import os
from PIL import Image
from network import resnet34
import time
from TrainDataset import TrainDataset
from TestDataset import TestDataset
from ValDataset import ValDataset
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable # torch 中 Variable 模块

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

file_w_dir = 'data/DUT/DUTSal'

data_transforms = transforms.Compose([
        # transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[1, 1, 1])
    ])


def S(x1, x2, geodesic,sigma_clr=10):
    return math.exp(-pow(geodesic[x1, x2], 2)/(2*sigma_clr*sigma_clr))


def compute_saliency_cost(smoothness, w_bg, wCtr):
    n = len(w_bg)
    A = np.zeros((n, n))
    b = np.zeros((n))
    for x in range(0,n):
        A[x,x] = 2 * w_bg[x] + 2 * (wCtr[x])
        b[x] = 2 * wCtr[x]
        for y in range(0, n):
            A[x, x] += 2 * smoothness[x, y]
            A[x, y] -= 2 * smoothness[x, y]
    x = np.linalg.solve(A, b)
    return x


def path_length(path, G):
    dist = 0.0
    for i in range(1,len(path)):
        dist += G[path[i - 1]][path[i]]['weight']
    return dist


def make_graph(grid):
    # get unique labels
    vertices = np.unique(grid)
    # map unique labels to [1,...,num_labels]
    reverse_dict = dict(zip(vertices,np.arange(len(vertices))))
    grid = np.array([reverse_dict[x] for x in grid.flat]).reshape(grid.shape)

    # create edges
    down = np.c_[grid[:-1, :].ravel(), grid[1:, :].ravel()]
    right = np.c_[grid[:, :-1].ravel(), grid[:, 1:].ravel()]
    all_edges = np.vstack([right, down])
    all_edges = all_edges[all_edges[:, 0] != all_edges[:, 1], :]
    all_edges = np.sort(all_edges, axis=1)
    num_vertices = len(vertices)
    edge_hash = all_edges[:,0] + num_vertices * all_edges[:, 1]
    # find unique connections
    edges = np.unique(edge_hash)
    # undo hashing
    edges = [[vertices[x%num_vertices], vertices[int(x/num_vertices)]] for x in edges]

    return vertices, edges


# def encoder_opt(encoder_output, DATA_MAX_LEN, vertices_batch, edges_batch, boundary_batch, centers_batch,
#                     max_dist_batch):
#
#     train_batchsize = encoder_output.data.shape[0]
#     OPTW_batch = np.zeros((train_batchsize, DATA_MAX_LEN))
#     for ii in range(train_batchsize):
#         vertices = vertices_batch[ii]
#         edges = edges_batch[ii]
#         boundary = boundary_batch[ii]
#         centers = centers_batch[ii]
#         max_dist = max_dist_batch[ii]
#
#         features = encoder_output.data[ii, :, :]
#
#         G = nx.Graph()
#         #buid the graph
#         for edge in edges:
#             pt1 = edge[0]
#             pt2 = edge[1]
#             mm1 = features[pt1, :]
#             mm2 = features[pt2, :]
#
#             color_distance = scipy.spatial.distance.euclidean(mm1,mm2)
#             color_distance = np.sqrt(np.sum(np.square(color_distance)))
#             #color_distance = np.linalg.norm(mm1,mm2)
#             G.add_edge(pt1, pt2, weight=color_distance)
#
#         #add a new edge in graph if edges are both on boundary
#         for v1 in vertices:
#             if boundary[v1] == 1:
#                 for v2 in vertices:
#                     if boundary[v2] == 1:
#                         #color_distance = tf.reduce_sum(tf.sqrt(tf.square(features[v1] - features[v2])), 0)
#                         color_distance = scipy.spatial.distance.euclidean(features[v1],features[v2])
#                         color_distance = np.sqrt(np.sum(np.square(color_distance)))
#                         G.add_edge(v1, v2, weight=color_distance)
#
#         geodesic = np.zeros((len(vertices), len(vertices)), dtype=float)
#         spatial = np.zeros((len(vertices), len(vertices)), dtype=float)
#         smoothness = np.zeros((len(vertices), len(vertices)), dtype=float)
#         adjacency = np.zeros((len(vertices), len(vertices)), dtype=float)
#
#         sigma_clr = 10.0
#         sigma_bndcon = 1.0
#         sigma_spa = 0.25
#         mu = 0.1
#         all_shortest_paths_color = nx.shortest_path(G, source=None, target=None, weight='weight')
#
#         for v1 in vertices:
#             for v2 in vertices:
#                 if v1 == v2:
#                     geodesic[v1, v2] = 0
#                     spatial[v1, v2] = 0
#                     smoothness[v1, v2] = 0
#                 else:
#                     geodesic[v1, v2] = path_length(all_shortest_paths_color[v1][v2], G)
#                     spatial[v1, v2] = scipy.spatial.distance.euclidean(centers[v1], centers[v2]) / max_dist
#                     smoothness[v1, v2] = math.exp(-(geodesic[v1, v2] * geodesic[v1, v2])/(2.0*sigma_clr*sigma_clr)) + mu
#
#         for edge in edges:
#             pt1 = edge[0]
#             pt2 = edge[1]
#             adjacency[pt1, pt2] = 1
#             adjacency[pt2, pt1] = 1
#
#         for v1 in vertices:
#             for v2 in vertices:
#                 smoothness[v1, v2] = adjacency[v1, v2] * smoothness[v1, v2]
#
#         area = dict()
#         len_bnd = dict()
#         bnd_con = dict()
#         w_bg = dict()
#         ctr = dict()
#         wCtr = dict()
#
#         for v1 in vertices:
#             area[v1] = 0
#             len_bnd[v1] = 0
#             ctr[v1] = 0
#             for v2 in vertices:
#                 d_app = geodesic[v1, v2]
#                 d_spa = spatial[v1, v2]
#                 w_spa = math.exp(- (d_spa * d_spa)/(2.0*sigma_spa*sigma_spa))
#                 area_i = S(v1, v2, geodesic)
#                 area[v1] += area_i
#                 len_bnd[v1] += area_i * boundary[v2]
#                 ctr[v1] += d_app * w_spa
#             bnd_con[v1] = len_bnd[v1] / math.sqrt(area[v1])
#             w_bg[v1] = 1.0 - math.exp(- (bnd_con[v1]*bnd_con[v1])/(2*sigma_bndcon*sigma_bndcon))
#
#         for v1 in vertices:
#             wCtr[v1] = 0
#             for v2 in vertices:
#                 d_app = geodesic[v1, v2]
#                 d_spa = spatial[v1, v2]
#                 w_spa = math.exp(- (d_spa*d_spa)/(2.0*sigma_spa*sigma_spa))
#                 wCtr[v1] += d_app * w_spa * w_bg[v2]
#
#         # normalise value for wCtr
#         min_value = min(wCtr.values())
#         max_value = max(wCtr.values())
#
#         for v in vertices:
#             wCtr[v] = (wCtr[v] - min_value)/(max_value - min_value)
#
#         r_opt_w = Variable(torch.FloatTensor(compute_saliency_cost(smoothness, w_bg, wCtr)))
#
#         OPTW_batch[ii, :r_opt_w.shape[0]] = r_opt_w
#
#     return OPTW_batch


def prepare_image_loader(img, gt):

    segments_slic = slic(img.cpu(), n_segments=160, compactness=1000, sigma=1, enforce_connectivity=1)

    nrows, ncols = segments_slic.shape
    max_dist = math.sqrt(nrows * nrows + ncols * ncols)

    grid = segments_slic

    (vertices, edges) = make_graph(grid)

    gridx, gridy = np.mgrid[:grid.shape[0], :grid.shape[1]]

    centers = dict()
    colors = dict()
    colors_rgb = dict()
    distances = dict()
    boundary = dict()
    roi = []

    for v in vertices:
        # centers[v] = [gridy[grid == v].mean(), gridx[grid == v].mean()]

        x_pix = gridx[grid == v]
        y_pix = gridy[grid == v]

        # if np.any(x_pix == 0) or np.any(y_pix == 0) or np.any(x_pix == nrows - 1) or np.any(y_pix == ncols - 1):
        #     boundary[v] = 1
        # else:
        #     boundary[v] = 0

        min_h_grid = min(x_pix)
        max_h_grid = max(x_pix)
        min_w_grid = min(y_pix)
        max_w_grid = max(y_pix)
        roi.append([min_h_grid, min_w_grid, max_h_grid, max_w_grid])

        # if np.any(x_pix == nrows - 1):  # sign as boundary
        #     roi.append([0, 0, 0, 0])

    # if v < 135:
    #     for vi in range(134-v):
    #         roi.append([0, 0, 0, 0])

    roi = np.array(roi)
    nnn = roi.shape[0]
    if nnn < 180:
        roi = roi.tolist()
        for vi in range(180-nnn):
            roi.append([0, 0, 0, 0])
        roi = np.array(roi)

    gt_pxl = []
    gt_np = gt.cpu().numpy()
    if len(gt.shape) == 3:      # got a grayscale image
        gt_np = skimage.color.rgb2gray(gt_np)
    if gt_np.shape[0] != grid.shape[0] or gt_np.shape[0]!=grid.shape[0]:
        gt_np = transform.resize(gt_np, grid.shape)
    for v in vertices:
        gt_pxl.append(np.mean(gt_np[grid == v], axis=0))

    nn = vertices.shape[0]
    if nn < 180:
        for vi in range(180 - nnn):
            gt_pxl.append(0)

    gt_pxl = np.rint(np.array(gt_pxl))

    #guiyihuya=======================
    img = data_transforms(img)
    #================================

    img_rgb = img.permute(2, 0, 1).unsqueeze_(0).float().cuda()

    return img_rgb, gt_pxl, vertices, edges, boundary, centers, max_dist, grid, roi


def prepare_image4test_loader(img):

    img_np = img.cpu().numpy()
    segments_slic = slic(img.cpu(), n_segments=160, compactness=10, sigma=1, enforce_connectivity=1)
    img_superpixels = []

    nrows, ncols = segments_slic.shape
    max_dist = math.sqrt(nrows * nrows + ncols * ncols)

    grid = segments_slic

    (vertices, edges) = make_graph(grid)

    gridx, gridy = np.mgrid[:grid.shape[0], :grid.shape[1]]
    centers = dict()
    colors = dict()
    colors_rgb = dict()
    boundary = dict()
    roi = []

    for v in vertices:
        centers[v] = [gridy[grid == v].mean(), gridx[grid == v].mean()]
        colors[v] = np.mean(img_np[grid == v], axis=0)

        x_pix = gridx[grid == v]
        y_pix = gridy[grid == v]

        if np.any(x_pix == 0) or np.any(y_pix == 0) or np.any(x_pix == nrows - 1) or np.any(y_pix == ncols - 1):
            boundary[v] = 1
        else:
            boundary[v] = 0

        min_h_grid = min(x_pix)
        max_h_grid = max(x_pix)
        min_w_grid = min(y_pix)
        max_w_grid = max(y_pix)
        roi.append([min_h_grid, min_w_grid, max_h_grid, max_w_grid])

        colors_rgb[v] = np.mean(img_np[grid == v], axis=0)

        if np.any(x_pix == nrows - 1):  # sign as boundary
            roi.append([0, 0, 0, 0])


    # if v < 135:
    #     for vi in range(134-v):
    #         roi.append([0, 0, 0, 0])

    roi = np.array(roi)
    nnn = roi.shape[0]
    if nnn < 180:
        roi = roi.tolist()
        for vi in range(180-nnn):
            roi.append([0, 0, 0, 0])
        roi = np.array(roi)


    img = data_transforms(img)
    img_rgb = img.permute(2, 0, 1).unsqueeze_(0).float().cuda()
    roi = np.array(roi)

    return img_rgb, img_superpixels, grid, vertices, edges, boundary, centers, max_dist, roi


train_batchsize = 16
val_batchsize = 16
test_batchsize = 1
#######################################################
workers = 0
train_data_list = pd.read_csv('data/label_dut.csv')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_data = TrainDataset(train_data_list,
                              transform=transforms.Compose([
                                  transforms.Resize((224, 224)),
                                  # transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  # normalize,
                              ]))
train_loader = DataLoader(train_data, batch_size=train_batchsize, shuffle=True, pin_memory=True, num_workers=workers)

val_data_list = pd.read_csv('data/val_msra.csv')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
val_data = ValDataset(val_data_list,
                              transform=transforms.Compose([
                                  transforms.Resize((224, 224)),
                                  transforms.ToTensor(),
                                  # normalize,
                              ]))
val_loader = DataLoader(val_data, batch_size=val_batchsize, shuffle=True, pin_memory=True, num_workers=workers)


test_data_list = pd.read_csv('data/test.csv')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
test_data = TestDataset(test_data_list,
                              transform=transforms.Compose([
                                  transforms.Resize((224, 224)),
                                  transforms.ToTensor(),
                                  # normalize,
                              ]))
test_loader = DataLoader(test_data, batch_size=test_batchsize, shuffle=True, pin_memory=True, num_workers=workers)

######################################################

num_epochs = 50  # <---160

EncoderModel = MySalEncoder()
DecoderModel = MyDNNDecoder()

#load the previous best parameters
# checkpoint_encoder = torch.load('data/en_check_params.pkl')
# EncoderModel.load_state_dict(checkpoint_encoder)
# checkpoint_decoder = torch.load('data/de_check_params.pkl')
# DecoderModel.load_state_dict(checkpoint_decoder)

# for name, param in EncoderModel.named_parameters():
#     if 'bias' in name:
#         nn.init.constant_(param, 0.0)
#     elif 'weight' in name:
#         nn.init.xavier_normal_(param)
# for name, param in DecoderModel.named_parameters():
#     if 'bias' in name:
#         nn.init.constant_(param, 0.0)
#     elif 'weight' in name:
#         nn.init.xavier_normal_(param)

EncoderModel.cuda()
DecoderModel.cuda()

# DecoderModel = nn.DataParallel(DecoderModel)

criterion = nn.BCELoss()  #SmoothL1Loss  BCELoss


encoder_optimizer = optim.Adam(EncoderModel.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
decoder_optimizer = optim.Adam(DecoderModel.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

# encoder_optimizer = optim.RMSprop(EncoderModel.parameters(), lr=0.01, eps=1e-08, weight_decay=0)
# decoder_optimizer = optim.RMSprop(DecoderModel.parameters(), lr=0.01, eps=1e-08, weight_decay=0)


encoder_scheduler = lr_scheduler.ReduceLROnPlateau(encoder_optimizer, 'min', patience=6, factor=0.5, min_lr=0.000001)
decoder_scheduler = lr_scheduler.ReduceLROnPlateau(decoder_optimizer, 'min', patience=6, factor=0.5, min_lr=0.000001)

en_best_model_wts = copy.deepcopy(EncoderModel.state_dict())
best_loss = 1000
print_inteval = 30
notimproveNum = 0
clip = 5

tensor = torch.randn(180, 180).cuda()

for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch+1, num_epochs))
    print('-' * 10)
    if notimproveNum > 55:
        print('Valloss do not improve at {} epochs,so break'.format(notimproveNum))
        break
    for phase in ['train', 'val']:
        if phase == 'train':
            #scheduler.step()
            EncoderModel.train()  # Set model to training mode
            DecoderModel.train()
            EncoderModel.batch_size = train_batchsize
            loader = train_loader

        else:
            EncoderModel.eval()  # Set model to evaluate mode
            DecoderModel.eval()
            EncoderModel.batch_size = val_batchsize
            loader = val_loader

        running_loss = 0.0
        pxl_num = 180
        # filename = os.listdir(Pic_dir)
        num_batch = 1
        for ii, (images, target) in enumerate(loader):
            time_start = time.time()
            image_var = torch.as_tensor(images).cuda()
            gt_val = torch.as_tensor(target).cuda()
            if image_var.size(0) != train_batchsize:
                break;

            # DATA = np.zeros((train_batchsize, pxl_num, 3, 32, 32))
            LABEL = np.zeros((train_batchsize, pxl_num))
            MASK = np.zeros((train_batchsize, pxl_num))
            DATA_LENS = np.zeros(train_batchsize)
            vertices_batch = dict()
            edges_batch = dict()
            boundary_batch = dict()
            centers_batch = dict()
            max_dist_batch = dict()
            img_batch = []
            roi_batch = []

            for k in range(train_batchsize):
                # print(gt_val.size())
                img_rgb, v2, v3, v4, v5, v6, v7, v8, roi = prepare_image_loader(image_var[k, :, :, :].permute(1,2,0), gt_val[k, :, :, :].permute(1,2,0))
                vertices_batch[k], edges_batch[k], boundary_batch[k], centers_batch[k], max_dist_batch[k] = v3, v4, v5, v6, v7
                LABEL[k, :v2.shape[0]] = v2
                MASK[k, :v2.shape[0]] = np.ones(v2.shape[0])
                img_batch.append(img_rgb)
                roi_batch.append(roi)
                DATA_LENS[k] = v2.shape[0]
            img_batch = torch.cat(img_batch)
            roi_batch = np.array(roi_batch)

            DATA_MAX_LEN = int(max(DATA_LENS))

            LABEL = torch.from_numpy(LABEL).float().cuda()
            # LABEL = Variable(LABEL)

            MASK = torch.from_numpy(MASK).float()
            MASK = Variable(MASK.cuda())

            # EncoderModel.train()
            # DecoderModel.train()
            EncoderModel.zero_grad()
            encoder_outputs = EncoderModel(img_batch, roi_batch)
            target_length_col = encoder_outputs.data.shape[1]

            # opt_w = encoder_opt(encoder_outputs, DATA_MAX_LEN, vertices_batch, edges_batch, boundary_batch, centers_batch, max_dist_batch)
            #
            # encoder_outputs = encoder_outputs.data.cpu().numpy()
            # for i in range(train_batchsize):
            #     opt_wex = opt_w[i, :]
            #     for j in range(target_length_col-1):
            #         opt_wex = np.column_stack((opt_wex, opt_w[i, :]))
            #     encoder_outputs[i, :, :] = encoder_outputs[i, :, :] * opt_wex
            #
            # encoder_outputs = Variable(torch.from_numpy(encoder_outputs).float())

            #-----decoder process
            DecoderModel.zero_grad()
            decoder_output, out_bg = DecoderModel(encoder_outputs, LABEL)

            variable = Variable(tensor, requires_grad=True)
            variable = variable.squeeze(0)
            U, S, V = torch.svd(variable)
            S1=torch.zeros(180).cuda()
            sval_nums = 32
            S1[0:sval_nums]=S[0:sval_nums]
            variable = torch.mm(U[:, 0:sval_nums], torch.mm(S1.diag(), V[0:sval_nums,:].t()).t())
            variable = variable.unsqueeze(0)
            loss2 = torch.norm(out_bg - variable.matmul(out_bg))/(1024)

            # target_length = encoder_outputs.data.shape[1]
            # decoder_output = []
            # for current_index in range(target_length):
            #     decoder_output.append(DecoderModel(encoder_outputs.cuda(), current_index))
            # decoder_output = torch.stack(decoder_output).permute(1,0,2).reshape(train_batchsize*target_length, 1)

            # loss using low rank or not
            # total_loss = criterion(decoder_output*MASK, LABEL*MASK)
            total_loss = criterion(decoder_output, LABEL) + loss2
            # total_loss = criterion(decoder_output, LABEL)

            #-----end of decoder process
            if phase == 'train':
                total_loss.backward()
                utils.clip_grad_norm_(EncoderModel.parameters(), clip)
                utils.clip_grad_norm_(DecoderModel.parameters(), clip)
                encoder_optimizer.step()
                decoder_optimizer.step()

                variable = variable.squeeze(0)
                U, S, V = torch.svd(variable)
                S1 = torch.zeros(180).cuda()
                S1[0:sval_nums] = S[0:sval_nums]
                variable = torch.mm(U[:, 0:sval_nums], torch.mm(S1.diag(), V[0:sval_nums, :].t()).t())
                variable = variable.unsqueeze(0)

                if ii % print_inteval == 0:
                    print('{}: {} Average_BatchLoss: {:.4f} '.format(ii, phase, total_loss.data))

                en_eachbatch_model_wts = copy.deepcopy(EncoderModel.state_dict())
                de_eachbatch_model_wts = copy.deepcopy(DecoderModel.state_dict())

                time_end = time.time()
                if ii % print_inteval == 0:
                    print('cost {:.1f} secs'.format(time_end - time_start))

            else:
                time_end = time.time()
                if ii % print_inteval == 0:
                    print('{}: {} Average_BatchLoss: {:.4f}: '.format(ii, phase, total_loss.data))
                    print('cost {:.1f} secs'.format(time_end - time_start))
            running_loss += total_loss.data
            num_batch = ii+1

        epoch_loss = running_loss/num_batch

        if phase == 'val':
            # print('num_batch'.format(num_batch))
            en_former_lr = encoder_optimizer.param_groups[0]['lr']
            encoder_scheduler.step(epoch_loss)
            en_current_lr = encoder_optimizer.param_groups[0]['lr']

            de_former_lr = decoder_optimizer.param_groups[0]['lr']
            decoder_scheduler.step(epoch_loss)
            de_current_lr = decoder_optimizer.param_groups[0]['lr']

            #writer.add_scalar('Epoch_VALLoss', epoch_loss, epoch)
            print('Encoder learning rate is {}'.format(encoder_optimizer.param_groups[0]['lr']))
            print('Decoder learning rate is {}'.format(decoder_optimizer.param_groups[0]['lr']))

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                en_best_model_wts = copy.deepcopy(EncoderModel.state_dict())
                de_best_model_wts = copy.deepcopy(DecoderModel.state_dict())
                print('BestLoss: {:.4f} is Epoch{} '.format(best_loss, epoch+1))
                notimproveNum = 0
            else:
                notimproveNum = notimproveNum + 1

            torch.save(EncoderModel.state_dict(), 'data/en_check_paramsBCE.pkl')
            torch.save(DecoderModel.state_dict(), 'data/de_check_paramsBCE.pkl')

        print('{} EpochLoss: {} '.format(phase, epoch_loss))

# EncoderModel.load_state_dict(en_best_model_wts)
# torch.save(EncoderModel.state_dict(), 'data/en_best_params.pkl')
# DecoderModel.load_state_dict(de_best_model_wts)
# torch.save(DecoderModel.state_dict(), 'data/de_best_params.pkl')



# ##------------evaluate----------------------------------------------##
Pic_save_dir = 'data/DUT_SalBCE'
#load the parameters
# checkpoint_encoder = torch.load('data/en_check_params.pkl')
# EncoderModel.load_state_dict(checkpoint_encoder)
# checkpoint_decoder = torch.load('data/de_check_params.pkl')
# DecoderModel.load_state_dict(checkpoint_decoder)

for ii, (images, filename) in enumerate(test_loader):

    image_var = torch.tensor(images).squeeze().cuda() #async=True

    img_rgb, DATA, grid, vertices, edges, boundary, centers, max_dist, roi = prepare_image4test_loader(image_var.permute(1,2,0))

    EncoderModel.eval()
    roi_batch = []
    roi_batch.append(roi)
    roi_batch = np.array(roi_batch)
    encoder_outputs = EncoderModel(img_rgb, roi_batch)   # .unsqueeze(0)
    target_length = encoder_outputs.data.shape[1]
    target_length_col = encoder_outputs.data.shape[2]
    vertices_batch = dict()
    edges_batch = dict()
    boundary_batch = dict()
    centers_batch = dict()
    max_dist_batch = dict()
    vertices_batch[0] = vertices
    edges_batch[0] = edges
    boundary_batch[0] = boundary
    centers_batch[0] = centers
    max_dist_batch[0] = max_dist

    sal_img = grid
    DecoderModel.eval()
    Lable = None
    decoder_output, out_bg = DecoderModel(encoder_outputs, Lable)

    for current_index in range(target_length):
        mm = decoder_output[current_index]
        sal_img[grid == current_index] = mm.detach().cpu()*255
    b = np.array(sal_img)
    b = b.astype(np.uint8)
    xx = Image.fromarray(b)
    print(filename[0])
    xx.save(Pic_save_dir+'/'+filename[0])
    # -----end of decoder process