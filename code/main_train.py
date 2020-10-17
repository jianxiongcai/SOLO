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

#    imgs_path = '../../data/hw3_mycocodata_img_comp_zlib.h5'
#    masks_path = '../../data/hw3_mycocodata_mask_comp_zlib.h5'
#    labels_path = '../../data/hw3_mycocodata_labels_comp_zlib.npy'
#    bboxes_path = '../../data/hw3_mycocodata_bboxes_comp_zlib.npy'

# set up output dir (for plotGT)

paths = [imgs_path, masks_path, labels_path, bboxes_path]
# load the data into data.Dataset
dataset = BuildDataset(paths)

full_size = len(dataset)
train_size = int(full_size * 0.8)
test_size = full_size - train_size

torch.random.manual_seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

batch_size = 2
train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
train_loader = train_build_loader.loader()
test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = test_build_loader.loader()

resnet50_fpn = Resnet50Backbone()
solo_head = SOLOHead(num_classes=4) ## class number is 4, because consider the background as one category.
# loop the image
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

resnet50_fpn = resnet50_fpn.to(device)
solo_head = solo_head.to(device)

num_epochs = 36
optimizer = optim.SGD(solo_head.parameters(), lr=0.01/8, momentum=0.9, weight_decay=0.0001)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[27,33], gamma=0.1)
train_cate_losses=[]
train_mask_losses=[]
train_total_losses=[]

test_cate_losses=[]
test_mask_losses=[]
test_total_losses=[]

num_log_iter=100
os.makedirs("train_check_point", exist_ok=True)

for epoch in range(num_epochs):
    ## fill in your training code
    solo_head.train()
    running_cate_loss = 0.0
    running_mask_loss=0.0
    running_total_loss=0.0
    for iter, data in enumerate(train_loader, 0):
        img, label_list, mask_list, bbox_list = [data[i] for i in range(len(data))]
        img = img.to(device)      
        label_list = [x.to(device) for x in label_list]
        mask_list = [x.to(device) for x in mask_list]
        bbox_list = [x.to(device) for x in bbox_list]
        
        
        
        backout = resnet50_fpn(img)
        fpn_feat_list = list(backout.values())
        optimizer.zero_grad()
        cate_pred_list, ins_pred_list = solo_head.forward(fpn_feat_list, eval=False) 
        ins_gts_list, ins_ind_gts_list, cate_gts_list = solo_head.target(ins_pred_list,
                                                                        bbox_list,
                                                                        label_list,
                                                                        mask_list)  
        cate_loss, mask_loss, total_loss=solo_head.loss(cate_pred_list,ins_pred_list,ins_gts_list,ins_ind_gts_list,cate_gts_list)  #batch loss
        total_loss.backward()
        optimizer.step()
        running_cate_loss += cate_loss.item()
        running_mask_loss += mask_loss.item()
        running_total_loss += total_loss.item()
        if iter % 100 == 99:
            log_cate_loss = running_cate_loss
            log_mask_loss = running_mask_loss
            log_total_loss = running_total_loss
            train_cate_losses.append(log_cate_loss)
            train_mask_losses.append(log_mask_loss)
            train_total_losses.append(log_total_loss)
            print('\nIteration:{} Avg. train total loss: {:.4f}'.format(iter+1, log_total_loss))
            running_cate_loss = 0.0
            running_mask_loss = 0.0
            running_total_loss = 0.0

    path = './train_check_point/solo_epoch_'+str(epoch)
    torch.save({
              'epoch': epoch,
              'model_state_dict': solo_head.state_dict(),
              'optimizer_state_dict': optimizer.state_dict()
              }, path)

    solo_head.eval()
    test_running_cate_loss = 0.0    
    test_running_mask_loss=0.0
    test_running_total_loss=0.0
    
    
    with torch.no_grad():
        for iter, data in enumerate(test_loader, 0):   
            img, label_list, mask_list, bbox_list = [data[i] for i in range(len(data))]
            img = img.to(device)  
            label_list = [x.to(device) for x in label_list]
            mask_list = [x.to(device) for x in mask_list]
            bbox_list = [x.to(device) for x in bbox_list]
                                  
            backout = resnet50_fpn(img)
            fpn_feat_list = list(backout.values())
            cate_pred_list, ins_pred_list = solo_head.forward(fpn_feat_list, eval=False) 

            ins_gts_list, ins_ind_gts_list, cate_gts_list = solo_head.target(ins_pred_list,
                                                                    bbox_list,
                                                                    label_list,
                                                                    mask_list)

            cate_loss, mask_loss, total_loss=solo_head.loss(cate_pred_list,ins_pred_list,ins_gts_list,ins_ind_gts_list,cate_gts_list)
            
            test_running_cate_loss += cate_loss.item()
            test_running_mask_loss += mask_loss.item()
            test_running_total_loss += total_loss.item()
            
        epoch_cate_loss = test_running_cate_loss / len(test_loader.dataset)
        epoch_mask_loss = test_running_mask_loss / len(test_loader.dataset)
        epoch_total_loss = test_running_total_loss / len(test_loader.dataset)
        print('\nEpoch:{} Avg. test loss: {:.4f}\n'.format(epoch + 1, epoch_total_loss))
        test_cate_losses.append(epoch_cate_loss)
        test_mask_losses.append(epoch_mask_loss)
        test_total_losses.append(epoch_total_loss)
    
    scheduler.step()
    
