import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import ndimage
from dataset import *
from functools import partial

class SOLOHead(nn.Module):
    def __init__(self,
        num_classes,
        in_channels=256,
        seg_feat_channels=256,
        stacked_convs=7,
        strides=[8, 8, 16, 32, 32],
        scale_ranges=((1, 96), (48, 192), (96, 384), (192, 768), (384, 2048)),
        epsilon=0.2,
        num_grids=[40, 36, 24, 16, 12],
        cate_down_pos=0,
        with_deform=False,
        mask_loss_cfg=dict(weight=3),
        cate_loss_cfg=dict(gamma=2,
                        alpha=0.25,
                        weight=1),
        postprocess_cfg=dict(cate_thresh=0.2,
                              ins_thresh=0.5,
                              pre_NMS_num=50,
                              keep_instance=5,
                              IoU_thresh=0.5)):
        super(SOLOHead, self).__init__()
        self.num_classes = num_classes
        self.seg_num_grids = num_grids
        self.cate_out_channels = self.num_classes - 1
        self.in_channels = in_channels
        self.seg_feat_channels = seg_feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.epsilon = epsilon
        self.cate_down_pos = cate_down_pos
        self.scale_ranges = scale_ranges
        self.with_deform = with_deform

        self.mask_loss_cfg = mask_loss_cfg
        self.cate_loss_cfg = cate_loss_cfg
        self.postprocess_cfg = postprocess_cfg
        # initialize the layers for cate and mask branch, and initialize the weights
        self._init_layers()
        self._init_weights()

        # check flag
        assert len(self.ins_head) == self.stacked_convs
        assert len(self.cate_head) == self.stacked_convs
        assert len(self.ins_out_list) == len(self.strides)
        pass

    # This function build network layer for cate and ins branch
    # it builds 4 self.var
        # self.cate_head is nn.ModuleList 7 inter-layers of conv2d
        # self.ins_head is nn.ModuleList 7 inter-layers of conv2d
        # self.cate_out is 1 out-layer of conv2d
        # self.ins_out_list is nn.ModuleList len(self.seg_num_grids) out-layers of conv2d, one for each fpn_feat
    def _init_layers(self):
        ## TODO initialize layers: stack intermediate layer and output layer
        num_groups = 32
        
        self.cate_head=nn.ModuleList([])
        for i in range(self.stacked_convs):
              layer=nn.ModuleList([
                  nn.Conv2d(self.in_channels, self.seg_feat_channels, kernel_size=(3, 3), stride=1, padding=1, bias=False),
                  nn.GroupNorm(num_groups,self.seg_feat_channels),
                  nn.ReLU(True)
              ])
              self.cate_head.append(layer)
        
    
        self.cate_out= nn.ModuleList([
              nn.Conv2d(self.seg_feat_channels, self.cate_out_channels, kernel_size=(3, 3), padding=1,bias=True),
              nn.Sigmoid()
                ])
           
    
        self.ins_head=nn.ModuleList([])
        first_layer=nn.ModuleList([
                    nn.Conv2d(self.in_channels+2, self.seg_feat_channels, kernel_size=(3, 3), stride=1, padding=1, bias=False),
                    nn.GroupNorm(num_groups,self.seg_feat_channels),
                    nn.ReLU(True)
                ])
        self.ins_head.append(first_layer)
        for i in range(self.stacked_convs-1):
             layer=nn.ModuleList([
                  nn.Conv2d(self.in_channels, self.seg_feat_channels, kernel_size=(3, 3), stride=1, padding=1, bias=False),
                  nn.GroupNorm(num_groups,self.seg_feat_channels),
                  nn.ReLU(True)
              ])
             self.ins_head.append(layer)
    
    
              
        num_grids_2=list(map(lambda x:pow(x,2), self.seg_num_grids))
        
        self.ins_out_list = []
        last_layer=nn.ModuleList([])
        for i in range(len(num_grids_2)):
             last_layer=nn.ModuleList([
             nn.Conv2d(self.seg_feat_channels, num_grids_2[i], kernel_size=(1, 1), padding=0,bias=True),
             nn.Sigmoid() 
             ])
             self.ins_out_list.append(last_layer)
        

    # This function initialize weights for head network
    def _init_weights(self):
        ## TODO: initialize the weights
        for m in self.cate_head.modules():
            if isinstance(m, nn.Conv2d): 
                m.weight.data =torch.nn.init.xavier_uniform_(m.weight.data)
                assert m.bias is None
        for m in self.ins_head.modules():
            if isinstance(m, nn.Conv2d): 
                m.weight.data =torch.nn.init.xavier_uniform_(m.weight.data)
                assert m.bias is None
        for m in self.cate_out.modules():
            if isinstance(m, nn.Conv2d): 
                m.weight.data =torch.nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant_(m.bias, 0) 
        for layer in self.ins_out_list:
            for m in layer:
                if isinstance(m, nn.Conv2d): 
                    m.weight.data =torch.nn.init.xavier_uniform_(m.weight.data) 
                    nn.init.constant_(m.bias, 0)

         
    # Forward function should forward every levels in the FPN.
    # this is done by map function or for loop
    # Input:
        # fpn_feat_list: backout_list of resnet50-fpn
    # Output:
        # if eval = False
            # cate_pred_list: list, len(fpn_level), each (bz,C-1,S,S)
            # ins_pred_list: list, len(fpn_level), each (bz, S^2, 2H_feat, 2W_feat)
        # if eval==True
            # cate_pred_list: list, len(fpn_level), each (bz,S,S,C-1) / after point_NMS
            # ins_pred_list: list, len(fpn_level), each (bz, S^2, Ori_H, Ori_W) / after upsampling
    def forward(self,
                fpn_feat_list,
                eval=True):
        new_fpn_list = self.NewFPN(fpn_feat_list)  # stride[8,8,16,32,32]
        assert new_fpn_list[0].shape[1:] == (256,100,136)
        quart_shape = [new_fpn_list[0].shape[-2]*2, new_fpn_list[0].shape[-1]*2]  # stride: 4
        # TODO: use MultiApply to compute cate_pred_list, ins_pred_list. Parallel w.r.t. feature level.
        # DONE (jianxiong)
        cate_pred_list, ins_pred_list = self.MultiApply(self.forward_single_level,
                                                        new_fpn_list, list(range(len(new_fpn_list))),
                                                        eval=eval, upsample_shape=quart_shape)
        assert len(new_fpn_list) == len(self.seg_num_grids)

        # assert cate_pred_list[1].shape[1] == self.cate_out_channels
        assert ins_pred_list[1].shape[1] == self.seg_num_grids[1]**2
        assert cate_pred_list[1].shape[2] == self.seg_num_grids[1]
        return cate_pred_list, ins_pred_list

    # This function upsample/downsample the fpn level for the network
    # In paper author change the original fpn level resolution
    # Input:
        # fpn_feat_list, list, len(FPN), stride[4,8,16,32,64]
    # Output:
    # new_fpn_list, list, len(FPN), stride[8,8,16,32,32]
    def NewFPN(self, fpn_feat_list):
        new_fpn_list=[
                [],[],[],[],[]
                ]
        new_fpn_list[0]=torch.nn.functional.interpolate(fpn_feat_list[0],scale_factor=1/2)
        new_fpn_list[1]=fpn_feat_list[1]
        new_fpn_list[2]=fpn_feat_list[2]
        new_fpn_list[3]=fpn_feat_list[3]
        last_layer=((25,34))
        new_fpn_list[4]=F.interpolate(fpn_feat_list[4],size=(last_layer[0],last_layer[1]))
        return new_fpn_list
    
    def torch_interpolate(x, H, W):
        """
        A quick wrapper fucntion for torch interpolate
        :return:
        """
        assert isinstance(x, torch.Tensor)
        C = x.shape[0]
        # require input: mini-batch x channels x [optional depth] x [optional height] x width
        x_interm = torch.unsqueeze(x, 0)
        x_interm = torch.unsqueeze(x_interm, 0)

        tensor_out = F.interpolate(x_interm, (C, H, W))
        tensor_out = tensor_out.view((C, H, W))
        return tensor_out
    # This function forward a single level of fpn_featmap through the network
    # Input:
        # fpn_feat: (bz, fpn_channels(256), H_feat, W_feat)
        # idx: indicate the fpn level idx, num_grids idx, the ins_out_layer idx
    # Output:
        # if eval==True
            # cate_pred: (bz,S,S,C-1) / after point_NMS
            # ins_pred: (bz, S^2, Ori_H/4, Ori_W/4) / after upsampling 
        # if eval==False
            # cate_pred: (bz,C-1,S,S)
            # ins_pred: (bz, S^2, 2H_feat, 2W_feat)
    def forward_single_level(self, fpn_feat, idx, eval=False, upsample_shape=None):
        # upsample_shape is used in eval mode
        ## TODO: finish forward function for single level in FPN.
        ## Notice, we distinguish the training and inference.
        cate_pred = fpn_feat
        ins_pred = fpn_feat
        num_grid = self.seg_num_grids[idx]  # current level grid
        # fpn_feat(bz,256,h,w)

        # in inference time, upsample the pred to (ori image size/4)
        H_feat=fpn_feat.shape[2]
        W_feat=fpn_feat.shape[3]
        bz=fpn_feat.shape[0]
        fnp_idx=self.ins_out_list[idx]
     
        if eval == True:
            ## TODO resize ins_pred
            ##category
            cate_pred=F.interpolate(fpn_feat,size=(num_grid,num_grid),mode='bilinear')  #bz,256,S,S
            for y in self.cate_head:
                for f in y:
                    cate_pred=f(cate_pred)
            for f in self.cate_out:
                cate_pred=f(cate_pred)
            
            
#            cate_pred=self.cate_head(cate_pred) #bz,256,S,S
#            cate_pred=self.cate_out(cate_pred) #bz,C-1,S,S
            cate_pred = self.points_nms(cate_pred).permute(0,2,3,1)     # cate_pred: (bz,S,S,C-1)
                                                                            # cate_pred: (bz,C-1,S,S)
            ##mask
            H = upsample_shape.shape[0]/2 ##200  
            W = upsample_shape.shape[1]/2 ##272
            x=torch.linspace(0,1,W)  ##100                     #bz,256+2,S,S
            y=torch.linspace(0,1,H)  ##136
            xm,ym=torch.meshgrid([x, y])
            xm=torch.unsqueeze(xm, 0).permute(0,2,1) ##xm (1,h,w)
            ym=torch.unsqueeze(ym, 0).permute(0,2,1) ##ym (1,h,w)
            xm=torch.unsqueeze(xm, 0)  ##xm (1,1,h,w)
            ym=torch.unsqueeze(ym, 0)   ##ym (1,1,h,w)
            xm = xm.repeat(bz, 1, 1, 1) ##xm (bz,1,h,w)
            ym = ym.repeat(bz, 1, 1, 1)  ##ym (bz,1,h,w)
            ins_pred=torch.cat((ins_pred,xm),dim=1)
            ins_pred=torch.cat((ins_pred,ym),dim=1)  ## (bz,256+2,h,w)
            for y in self.ins_head:
                for f in y:
                    ins_pred=f(ins_pred)   ## (bz,256,h,w)                     
            ins_pred=F.interpolate(ins_pred,size=(2*H,2*W))
            for f in fnp_idx:               ## (bz,256,200,272) 
                ins_pred=f(ins_pred)      
            
                      
        # check flag
        if eval == False:
            #category                                      # fpn_feat(bz,256,h,w)
            cate_pred=F.interpolate(fpn_feat,size=(num_grid,num_grid),mode='bilinear')  #bz,256,S,S
            for f in self.cate_head:
                for y in f:
                    cate_pred = y(cate_pred)
            for f in self.cate_out:
                cate_pred = f(cate_pred)
            #mask
            x=torch.linspace(0,1,W_feat)
            y=torch.linspace(0,1,H_feat)
            xm,ym=torch.meshgrid([x, y])
            xm=torch.unsqueeze(xm, 0).permute(0,2,1) ##xm (1,h,w)
            ym=torch.unsqueeze(ym, 0).permute(0,2,1) ##ym (1,h,w)
            xm=torch.unsqueeze(xm, 0)  ##xm (1,1,h,w)
            ym=torch.unsqueeze(ym, 0)   ##ym (1,1,h,w)
            xm = xm.repeat(bz, 1, 1, 1) ##xm (bz,1,h,w)
            ym = ym.repeat(bz, 1, 1, 1)  ##ym (bz,1,h,w)
            ins_pred=torch.cat((ins_pred,xm),dim=1)
            ins_pred=torch.cat((ins_pred,ym),dim=1)  ## (bz,256+2,h,w)  ## (bz,256+2,100,136)
            for y in self.ins_head:
                for f in y:
                    ins_pred=f(ins_pred)
            
#            ins_pred=self.ins_head(ins_pred)       ## (bz,256,h,w)  
            ins_pred=F.interpolate(ins_pred,size=(2*H_feat,2*W_feat))    ## (bz,256+2,200,272)
            for f in fnp_idx:               ## (bz,256,100,136) 
                ins_pred=f(ins_pred)      
            
#            ins_pred=fnp_idx(ins_pred)   ## (bz,s^2,h,w)  
                     
            assert cate_pred.shape[1:] == (3, num_grid, num_grid)
            assert ins_pred.shape[1:] == (num_grid**2, fpn_feat.shape[2]*2, fpn_feat.shape[3]*2)
        else:
            pass
        return cate_pred, ins_pred

    # Credit to SOLO Author's code
    # This function do a NMS on the heat map(cate_pred), grid-level
    # Input:
        # heat: (bz,C-1, S, S)
    # Output:
        # (bz,C-1, S, S)
    def points_nms(self, heat, kernel=2):
        # kernel must be 2
        hmax = nn.functional.max_pool2d(
            heat, (kernel, kernel), stride=1, padding=1)
        keep = (hmax[:, :, :-1, :-1] == heat).float()
        return heat * keep

    # This function compute loss for a batch of images
    # input:
        # cate_pred_list: list, len(fpn_level), each (bz,C-1,S,S)
        # ins_pred_list: list, len(fpn_level), each (bz, S^2, 2H_feat, 2W_feat)
        # ins_gts_list: list, len(bz), list, len(fpn), (S^2, 2H_f, 2W_f)
        # ins_ind_gts_list: list, len(bz), list, len(fpn), (S^2,)
        # cate_gts_list: list, len(bz), list, len(fpn), (S, S), {1,2,3}
    # output:
        # cate_loss, mask_loss, total_loss
    def loss(self,
             cate_pred_list,
             ins_pred_list,
             ins_gts_list,
             ins_ind_gts_list,
             cate_gts_list):
        ## TODO: compute loss, vecterize this part will help a lot. To avoid potential ill-conditioning, if necessary, add a very small number to denominator for focalloss and diceloss computation.
        pass



    # This function compute the DiceLoss
    # Input:
        # mask_pred: (2H_feat, 2W_feat)
        # mask_gt: (2H_feat, 2W_feat)
    # Output: dice_loss, scalar
    def DiceLoss(self, mask_pred, mask_gt):
        ## TODO: compute DiceLoss
        pass

    # This function compute the cate loss
    # Input:
        # cate_preds: (num_entry, C-1)
        # cate_gts: (num_entry,)
    # Output: focal_loss, scalar
    def FocalLoss(self, cate_preds, cate_gts):
        ## TODO: compute focalloss
        pass

    def MultiApply(self, func, *args, **kwargs):
        pfunc = partial(func, **kwargs) if kwargs else func
        map_results = map(pfunc, *args)

        return tuple(map(list, zip(*map_results)))

    # This function build the ground truth tensor for each batch in the training
    # Input:
        # ins_pred_list: list, len(fpn_level), each (bz, S^2, 2H_feat, 2W_feat)
        # / ins_pred_list is only used to record feature map
        # bbox_list: list, len(batch_size), each (n_object, 4) (x1y1x2y2 system)
        # label_list: list, len(batch_size), each (n_object, )
        # mask_list: list, len(batch_size), each (n_object, 800, 1088)
    # Output:
        # ins_gts_list: list, len(bz), list, len(fpn), (S^2, 2H_f, 2W_f)
        # ins_ind_gts_list: list, len(bz), list, len(fpn), (S^2,)
        # cate_gts_list: list, len(bz), list, len(fpn), (S, S), {1,2,3}
    def target(self,
              ins_pred_list,
              bbox_list,
              label_list,
              mask_list):
        # TODO: use MultiApply to compute ins_gts_list, ins_ind_gts_list, cate_gts_list. Parallel w.r.t. img mini-batch
        # remember, you want to construct target of the same resolution as prediction output in training
        feature_sizes = [(ins_pred.shape[2], ins_pred.shape[3]) for ins_pred in ins_pred_list]
        ins_gts_list, ins_ind_gts_list, cate_gts_list = self.MultiApply(self.target_single_img,
                                                                        bbox_list, label_list, mask_list,
                                                                        featmap_sizes=feature_sizes)

        # check flag
        assert ins_gts_list[0][1].shape == (self.seg_num_grids[1]**2, 200, 272)
        assert ins_ind_gts_list[0][1].shape == (self.seg_num_grids[1]**2,)
        assert cate_gts_list[0][1].shape == (self.seg_num_grids[1], self.seg_num_grids[1])

        return ins_gts_list, ins_ind_gts_list, cate_gts_list
    # -----------------------------------
    ## process single image in one batch
    # -----------------------------------
    # input:
        # gt_bboxes_raw: n_obj, 4 (x1y1x2y2 system)
        # gt_labels_raw: n_obj,
        # gt_masks_raw: n_obj, H_ori, W_ori
        # featmap_sizes: list of shapes of featmap
    # output:
        # ins_label_list: list, len: len(FPN), (S^2, 2H_feat, 2W_feat)
        # cate_label_list: list, len: len(FPN), (S, S)
        # ins_ind_label_list: list, len: len(FPN), (S^2, )
    def target_single_img(self,
                          gt_bboxes_raw,
                          gt_labels_raw,
                          gt_masks_raw,
                          featmap_sizes=None):
        ## TODO: finish single image target build
        # compute the area of every object in this single image

        # initial the output list, each entry for one featmap
        ins_label_list = []
        ins_ind_label_list = []
        cate_label_list = []

        # (for each image),compute meta data (object center region, etc)
        N_obj, W_ori, H_ori = gt_bboxes_raw.shape
        obj_scale_list = []
        obj_center_list = []
        obj_center_regions = []
        obj_indice = [[] for i in range(len(featmap_sizes))]            # each layer's positive instance number

        # compute object scale and assign to level in FPN
        for obj_idx in range(N_obj):
            # compute \sqrt(wh)
            bbox = gt_bboxes_raw[obj_idx]
            obj_w, obj_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            obj_scale = torch.sqrt(obj_w * obj_h)
            obj_scale_list.append(obj_scale)
            # assign object instance
            for level_idx, level_range in enumerate(self.scale_ranges, 0):
                if (obj_scale >= level_range[0]) and (obj_scale < level_range[1]):
                    obj_indice[level_idx].append(obj_idx)

            # calc object center region
            obj_c_x, obj_c_y = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
            obj_center_list.append(torch.tensor([obj_c_x, obj_c_y], dtype=torch.float))
            obj_center_regions.append(torch.tensor([
                obj_c_x - obj_w / 2.0,
                obj_c_y - obj_h / 2.0,
                obj_c_x + obj_w / 2.0,
                obj_c_y + obj_h / 2.0,
                ], dtype=torch.float))

        # for each level, compute the cate_label,
        # todo: note: FPN feature map [2, 3]
        # Note: the feat_size is (2 * H_feat, 2 * W_feat)
        for level_idx, feat_size in enumerate(featmap_sizes, 0):
            S = self.seg_num_grids[level_idx]
            assert feat_size.ndim == 2
            # cate label map / ins_label_list
            cate_label_map = torch.zeros((S, S), dtype=torch.long)
            ins_label_map = torch.zeros((S * S, feat_size[0], feat_size[1]), dtype=torch.float)
            ins_ind_label = torch.zeros((S * S), dtype=torch.float)
            # obj_idx w.r.t. gt_labels_raw / gt_bbox_raw
            for obj_idx in obj_indice[level_idx]:       # perfix i denotes grid cell index here
                # the 2D grid index where the center region boundary fall in
                # (4,) torch.FloatTensor
                obj_region_i = torch.floor(obj_center_regions[obj_idx] * S)
                obj_center_i = torch.floor(obj_center_list[obj_idx] * S)

                # set center point
                x_min = max(obj_center_i[0].item() - 1, obj_region_i[0].item())
                y_min = max(obj_center_i[1].item() - 1, obj_region_i[1].item())
                x_max = max(obj_center_i[0].item() + 1, obj_region_i[2].item())
                y_max = max(obj_center_i[1].item() + 1, obj_region_i[3].item())
                # set cate map
                cate_label_map[x_min : (x_max + 1), y_min : (y_max + 1)] = gt_labels_raw[obj_idx]

                # set target mask and ins_ind_label
                mask_raw = gt_masks_raw[obj_idx: (obj_idx+1)]           # 1 * H_feat * W_feat
                mask_resized = BuildDataset.torch_interpolate(mask_raw, feat_size[0], feat_size[1])
                for i in range(x_min, x_max + 1):
                    for j in range(y_min, y_max + 1):
                        ins_label_map[i * S + j] = mask_resized
                        ins_ind_label[i * S + j] = 1.0

            # prepare res
            cate_label_list.append(cate_label_map)
            ins_label_list.append(ins_label_map)
            ins_ind_label_list.append(ins_ind_label)

        # check flag
        assert ins_label_list[1].shape == (1296,200,272)
        assert ins_ind_label_list[1].shape == (1296,)
        assert cate_label_list[1].shape == (36, 36)
        return ins_label_list, ins_ind_label_list, cate_label_list

    # This function receive pred list from forward and post-process
    # Input:
        # ins_pred_list: list, len(fpn), (bz,S^2,Ori_H/4, Ori_W/4)
        # cate_pred_list: list, len(fpn), (bz,S,S,C-1)
        # ori_size: [ori_H, ori_W]
    # Output:
        # NMS_sorted_scores_list, list, len(bz), (keep_instance,)
        # NMS_sorted_cate_label_list, list, len(bz), (keep_instance,)
        # NMS_sorted_ins_list, list, len(bz), (keep_instance, ori_H, ori_W)

    def PostProcess(self,
                    ins_pred_list,
                    cate_pred_list,
                    ori_size):

        ## TODO: finish PostProcess
        pass


    # This function Postprocess on single img
    # Input:
        # ins_pred_img: (all_level_S^2, ori_H/4, ori_W/4)
        # cate_pred_img: (all_level_S^2, C-1)
    # Output:
        # NMS_sorted_scores_list, list, len(bz), (keep_instance,)
        # NMS_sorted_cate_label_list, list, len(bz), (keep_instance,)
        # NMS_sorted_ins_list, list, len(bz), (keep_instance, ori_H, ori_W)
    def PostProcessImg(self,
                       ins_pred_img,
                       cate_pred_img,
                       ori_size):

        ## TODO: PostProcess on single image.
        pass

    # This function perform matrix NMS
    # Input:
        # sorted_ins: (n_act, ori_H/4, ori_W/4)
        # sorted_scores: (n_act,)
    # Output:
        # decay_scores: (n_act,)
    def MatrixNMS(self, sorted_ins, sorted_scores, method='gauss', gauss_sigma=0.5):
        ## TODO: finish MatrixNMS
        pass

    # -----------------------------------
    ## The following code is for visualization
    # -----------------------------------
    # this function visualize the ground truth tensor
    # Input:
        # ins_gts_list: list, len(bz), list, len(fpn), (S^2, 2H_f, 2W_f)
        # ins_ind_gts_list: list, len(bz), list, len(fpn), (S^2,)
        # cate_gts_list: list, len(bz), list, len(fpn), (S, S), {1,2,3}
        # color_list: list, len(C-1)
        # img: (bz,3,Ori_H, Ori_W)
        ## self.strides: [8,8,16,32,32]
    def PlotGT(self,
               ins_gts_list,
               ins_ind_gts_list,
               cate_gts_list,
               color_list,
               img):
        ## TODO: target image recover, for each image, recover their segmentation in 5 FPN levels.
        ## This is an important visual check flag.
        pass

    # This function plot the inference segmentation in img
    # Input:
        # NMS_sorted_scores_list, list, len(bz), (keep_instance,)
        # NMS_sorted_cate_label_list, list, len(bz), (keep_instance,)
        # NMS_sorted_ins_list, list, len(bz), (keep_instance, ori_H, ori_W)
        # color_list: ["jet", "ocean", "Spectral"]
        # img: (bz, 3, ori_H, ori_W)
    def PlotInfer(self,
                  NMS_sorted_scores_list,
                  NMS_sorted_cate_label_list,
                  NMS_sorted_ins_list,
                  color_list,
                  img,
                  iter_ind):
        ## TODO: Plot predictions
        pass

from backbone import *
if __name__ == '__main__':
    solo_head = SOLOHead(num_classes=4)
    # file path and make a list
    # imgs_path = './data/hw3_mycocodata_img_comp_zlib.h5'
    # masks_path = './data/hw3_mycocodata_mask_comp_zlib.h5'
    # labels_path = "./data/hw3_mycocodata_labels_comp_zlib.npy"
    # bboxes_path = "./data/hw3_mycocodata_bboxes_comp_zlib.npy"

    imgs_path = '/workspace/data/hw3_mycocodata_img_comp_zlib.h5'
    masks_path = '/workspace/data/hw3_mycocodata_mask_comp_zlib.h5'
    labels_path = '/workspace/data/hw3_mycocodata_labels_comp_zlib.npy'
    bboxes_path = '/workspace/data/hw3_mycocodata_bboxes_comp_zlib.npy'

    paths = [imgs_path, masks_path, labels_path, bboxes_path]
    # load the data into data.Dataset
    dataset = BuildDataset(paths)

    ## Visualize debugging
    # --------------------------------------------
    # build the dataloader
    # set 20% of the dataset as the training data
    full_size = len(dataset)
    train_size = int(full_size * 0.8)
    test_size = full_size - train_size
    # random split the dataset into training and testset
    # set seed
    torch.random.manual_seed(1)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    # push the randomized training data into the dataloader

    # train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    # test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)
    batch_size = 2
    train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_loader = train_build_loader.loader()
    test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = test_build_loader.loader()


    resnet50_fpn = Resnet50Backbone()
    solo_head = SOLOHead(num_classes=4) ## class number is 4, because consider the background as one category.
    # loop the image
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for iter, data in enumerate(train_loader, 0):
        img, label_list, mask_list, bbox_list = [data[i] for i in range(len(data))]
        # fpn is a dict
        backout = resnet50_fpn(img)
        fpn_feat_list = list(backout.values())
        # make the target


        ## demo
        cate_pred_list, ins_pred_list = solo_head.forward(fpn_feat_list, eval=False)
        ins_gts_list, ins_ind_gts_list, cate_gts_list = solo_head.target(ins_pred_list,
                                                                        bbox_list,
                                                                        label_list,
                                                                        mask_list)
        mask_color_list = ["jet", "ocean", "Spectral"]
        solo_head.PlotGT(ins_gts_list,ins_ind_gts_list,cate_gts_list,mask_color_list,img)


