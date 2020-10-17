import numpy as np
import torch
from sklearn.model_selection import train_test_split
from solo_head import *
from backbone import *
from dataset import *
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from tensorboardX import SummaryWriter
from scipy import ndimage
from functools import partial
from matplotlib import pyplot as plt
from matplotlib import cm
import skimage.transform
import os.path
import shutil
import gc
from sklearn import metrics
gc.enable()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# file path and make a list
imgs_path = '/workspace/data/hw3_mycocodata_img_comp_zlib.h5'
masks_path = '/workspace/data/hw3_mycocodata_mask_comp_zlib.h5'
labels_path = "/workspace/data/hw3_mycocodata_labels_comp_zlib.npy"
bboxes_path = "/workspace/data/hw3_mycocodata_bboxes_comp_zlib.npy"

# set up output dir (for plotGT)
paths = [imgs_path, masks_path, labels_path, bboxes_path]
# load the data into data.Dataset
dataset = BuildDataset(paths)
del paths

full_size = len(dataset)
train_size = int(full_size * 0.8)
test_size = full_size - train_size

torch.random.manual_seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

eval_epoch = 1
print("[INFO] eval epoch: {}".format(eval_epoch))
batch_size = 2
# train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
# train_loader = train_build_loader.loader()
test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = test_build_loader.loader()

resnet50_fpn = Resnet50Backbone()
solo_head = SOLOHead(num_classes=4) ## class number is 4, because consider the background as one category.

# load checkpoint
checkpoint = torch.load("./train_check_point/solo_epoch_{}".format(eval_epoch))
solo_head.load_state_dict(checkpoint['model_state_dict'])

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

resnet50_fpn = resnet50_fpn.to(device)
resnet50_fpn.eval()             # set to eval mode
solo_head = solo_head.to(device)

# num_epochs = 36
# optimizer = optim.SGD(solo_head.parameters(), lr=0.01/16*batch_size, momentum=0.9, weight_decay=0.0001)
# # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[27,33], gamma=0.1)
# train_cate_losses=[]
# train_mask_losses=[]
# train_total_losses=[]

# test_cate_losses=[]
# test_mask_losses=[]
# test_total_losses=[]

# os.makedirs("train_check_point", exist_ok=True)

# tensorboard
# os.makedirs("logs", exist_ok=True)
# writer = SummaryWriter(log_dir="logs")

solo_head.eval()

with torch.no_grad():
    for iter, data in enumerate(test_loader, 0):
        img, label_list, mask_list, bbox_list = [data[i] for i in range(len(data))]
        img = img.to(device)
        label_list = [x.to(device) for x in label_list]
        mask_list = [x.to(device) for x in mask_list]
        bbox_list = [x.to(device) for x in bbox_list]

        backout = resnet50_fpn(img)
        # del img
        fpn_feat_list = list(backout.values())
        cate_pred_list, ins_pred_list = solo_head.forward(fpn_feat_list, eval=True)
        # del fpn_feat_list
        # ins_gts_list, ins_ind_gts_list, cate_gts_list = solo_head.target(ins_pred_list,
        #                                                         bbox_list,
        #                                                         label_list,
        #                                                         mask_list)

        # post-processing
        NMS_sorted_scores_list, NMS_sorted_cate_label_list, NMS_sorted_ins_list = solo_head.PostProcess(ins_pred_list, cate_pred_list, (img.shape[2], img.shape[3]))

        # plotgt
        break
