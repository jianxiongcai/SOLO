import numpy as np
from solo_head import *
from backbone import *
from dataset import *
from metric_tracker import MetricTracker
import torch.utils.data
import gc
import os
from tqdm import tqdm
from matplotlib import pyplot as plt
import pickle
gc.enable()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# file path and make a list
imgs_path = '/workspace/data/hw3_mycocodata_img_comp_zlib.h5'
masks_path = '/workspace/data/hw3_mycocodata_mask_comp_zlib.h5'
labels_path = "/workspace/data/hw3_mycocodata_labels_comp_zlib.npy"
bboxes_path = "/workspace/data/hw3_mycocodata_bboxes_comp_zlib.npy"

eval_epoch = 50
VISUALIZATION = False
batch_size = 2
cate_thresh = 0.33

# set up output dir (for plotGT)
paths = [imgs_path, masks_path, labels_path, bboxes_path]
# load the data into data.Dataset
dataset = BuildDataset(paths, augmentation=False)

full_size = len(dataset)
train_size = int(full_size * 0.8)
test_size = full_size - train_size

torch.random.manual_seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
# train_loader = train_build_loader.loader()
test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = test_build_loader.loader()

resnet50_fpn = Resnet50Backbone()
solo_head = SOLOHead(num_classes=4) ## class number is 4, because consider the background as one category.
# set cate_thresh to user-defined threshold
solo_head.postprocess_cfg['cate_thresh'] = cate_thresh
print("[INFO] Using user-defined cate_thresh: {}".format(cate_thresh))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    #gaidong 1
resnet50_fpn = resnet50_fpn.to(device)
resnet50_fpn.eval()             # set to eval mode
mask_color_list = ["jet", "ocean", "Spectral"]

def mAP_epoch(solo_head, MetricTracker, test_loader, resnet50_fpn, VISUALIZATION, mask_color_list):
    metric_trackers = []
    for i in range(solo_head.num_classes - 1):
        metric_trackers.append(MetricTracker(i))
    
    with torch.no_grad():
        for iter, data in enumerate(tqdm(test_loader), 0):
            img, label_list, mask_list, bbox_list = [data[i] for i in range(len(data))]
            img = img.to(device)
            label_list = [x.to(device) for x in label_list]
            mask_list = [x.to(device) for x in mask_list]
            # bbox_list = [x.to(device) for x in bbox_list]
            del bbox_list
    
            backout = resnet50_fpn(img)
            fpn_feat_list = list(backout.values())
            cate_pred_list, ins_pred_list = solo_head.forward(fpn_feat_list, eval=True)
            # del fpn_feat_list                                                       mask_list)
    
            # post-processing
            NMS_sorted_scores_list, NMS_sorted_cate_label_list, NMS_sorted_ins_list = solo_head.PostProcess(ins_pred_list, cate_pred_list, (img.shape[2], img.shape[3]))
            del ins_pred_list
            del cate_pred_list
            if (VISUALIZATION):
                solo_head.PlotInfer(NMS_sorted_scores_list, NMS_sorted_cate_label_list, NMS_sorted_ins_list,
                                    mask_color_list, img, iter)
            del img
            # for each image in the batch, do metric eval
            for sorted_scores, sorted_cate_label, sorted_ins, label_gt, mask_gt in zip(NMS_sorted_scores_list,
                                                     NMS_sorted_cate_label_list,
                                                     NMS_sorted_ins_list, label_list,
                                                     mask_list):
                assert torch.all(sorted_cate_label != 0)           # no background prediction class
                # compute the IoU between img and mask for each images
                for label_x in range(solo_head.num_classes):
                    if (label_x == 0):
                        # skip background
                        continue
                    indice_gt = (label_gt == label_x)
                    indice_pred = (sorted_cate_label == label_x)
    
                    # number of ground-truth objects with the class label_x
                    N_gt = torch.sum(indice_gt).item()
                    N_pred = torch.sum(indice_pred).item()
                    if N_gt == 0:                   # no label for the class in the ground-truth
                        if N_pred == 0:             # no gt + no pred
                            continue
                        # pred > 1, no gt: all prediction are negative (no match)
                        conf_scores = sorted_scores[indice_pred]
                        tp_indicator = np.zeros((N_pred, ), dtype=np.bool)
                        match_indice = np.zeros((N_pred, ), dtype=np.int)
                    else:
                        if N_pred != 0:
                            conf_scores = sorted_scores[indice_pred]
                            ins_pred = sorted_ins[indice_pred]
                            assert len(ins_pred) == N_pred
    
                            ins_gt = mask_gt[indice_gt]
    
                            # compute the IoU
                            ious = solo_head.MatrixIOU(ins_pred > solo_head.postprocess_cfg['ins_thresh'],
                                                       ins_gt > solo_head.postprocess_cfg['ins_thresh'])
                            assert ious.shape == (N_pred, N_gt)
                            # for each max prediction box, compute max_iou overlap with gt
                            max_ious, iou_max_idx = torch.max(ious, dim=1)
                            assert max_ious.shape[0] == N_pred
                            assert iou_max_idx.shape[0] == N_pred
    
                            tp_indicator = (max_ious > solo_head.postprocess_cfg['IoU_thresh'])
                            match_indice = iou_max_idx
                        else:
                            conf_scores = torch.zeros((0,))
                            tp_indicator = torch.zeros((0,), dtype=torch.bool)
                            match_indice = torch.zeros((0,), dtype=torch.int)
    
                    # MAP computation
                    metric_trackers[label_x - 1].add_match(conf_scores, tp_indicator, match_indice, N_gt)
        del label_list, mask_list
        # all images have done metric eval in the batch
        mAP = 0
        cnt = 0
        for i in range(solo_head.num_classes - 1):
            metric_trackers[i].compute_precision_recall()
            recall, precision = metric_trackers[i].sorted_pr_curve()
    
            # plot the precision-recall curve
            ap = metric_trackers[i].compute_ap()
            print("class_id: {}. ap: {}".format(i, ap))
            metric_trackers[i].reset()
            if ap > 0:
                cnt+=1
                mAP+=ap
        if cnt > 0:
            mAP = mAP / cnt
        return mAP
     
mAP_list=[]
for epoch in range(eval_epoch):
  # load checkpoint
  print("[INFO] eval epoch: {}".format(epoch))
  checkpoint = torch.load("./train_check_point/solo_epoch_{}".format(epoch))   #gaidong 2
#  checkpoint = torch.load("./solo_epoch_{}".format(epoch))
  solo_head.load_state_dict(checkpoint['model_state_dict'])
  solo_head = solo_head.to(device)
  solo_head.eval()
  mAP = mAP_epoch(solo_head, MetricTracker, test_loader, resnet50_fpn, VISUALIZATION, mask_color_list)
  print("[INFO] eval epoch: {} mAP : {}".format(epoch, mAP))       
  mAP_list.append(mAP)  
  with open('mAP.data', 'wb') as filehandle:
    pickle.dump(mAP_list, filehandle)
    
    
    
    
with open('mAP.data', 'rb') as filehandle:
    # read the data as binary data stream
    mAP = pickle.load(filehandle)
print("mAP : {}".format(mAP))


plt.plot(np.arange(1, eval_epoch + 1), np.array(mAP_list))        
plt.xlabel("Epoch")
plt.ylabel("mAP")
plt.title("mAP for each epoch")
plt.show()
saving_file = "mAP_epoch.png"
plt.savefig(saving_file)
