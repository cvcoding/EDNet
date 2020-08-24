import os
import math
import numpy as np
import pandas as pd
import os.path as osp
import torch

# train data
data_path = 'data/DUT/HKU-IS/imgs'
gt_path = 'data/DUT/HKU-IS/gt'
img_path, label = [], []


for img in os.listdir(data_path):
    vv = osp.join(data_path, img)
    filenamewithoutE = os.path.splitext(img)[0]
    gt_name = filenamewithoutE+'.png'
    img_path.append(vv)
    label.append(osp.join(gt_path, gt_name))

label_file = pd.DataFrame({'img_path': img_path, 'label': label})
label_file.to_csv('label.csv', index=False)

# # test data
# test_data_path = 'data/guangdong_round1_test_a_20180916'
# all_test_img = os.listdir(test_data_path)
# test_img_path = []
#
# for img in all_test_img:
#     if osp.splitext(img)[1] == '.jpg':
#         test_img_path.append(osp.join(test_data_path, img))
#
# test_file = pd.DataFrame({'img_path': test_img_path})
# test_file.to_csv('data/test.csv', index=False)
