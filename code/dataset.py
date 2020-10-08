## Author: Lishuo Pan 2020/4/18

import torch
import torchvision.transforms.functional
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')               # No display
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
import os.path

class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        """

        :param path: the path to the dataset root: /workspace/data or XXX/data/SOLO
        """
        # all files
        imgs_path, masks_path, labels_path, bboxes_path = path

        # load dataset
        # all images and masks will be lazy read
        self.images_h5 = h5py.File(imgs_path, 'r')
        self.masks_h5 = h5py.File(masks_path, 'r')
        self.labels_all = np.load(labels_path, allow_pickle=True)
        self.bboxes_all = np.load(bboxes_path, allow_pickle=True)

        # As the mask are saved sequentially, compute the mask start index for each images
        n_objects_img = [len(self.labels_all[i]) for i in range(len(self.labels_all))]      # Number of objects per list
        self.mask_offset = np.cumsum(n_objects_img)                                         # the start index for each images
        # Add a 0 to the head. offset[0] = 0
        self.mask_offset = np.concatenate([np.array([0]), self.mask_offset])


    # output:
        # transed_img
        # label
        # transed_mask
        # transed_bbox
    def __getitem__(self, index):
        # images
        img_np = self.images_h5['data'][index] / 255.0                     # (3, 300, 400)
        img = torch.tensor(img_np, dtype=torch.float)


        # annotation
        # label: start counting from 0
        label = torch.tensor(self.labels_all[index], dtype=torch.long)
        # collect all object mask for the image
        mask_offset_s = self.mask_offset[index]
        mask_list = []
        for i in range(len(label)):
            # get the mask of the ith object in the image
            mask_np = self.masks_h5['data'][mask_offset_s + i] * 1.0
            mask_tmp = torch.tensor(mask_np, dtype=torch.float)
            mask_list.append(mask_tmp)
        # (n_obj, 300, 400)
        mask = torch.stack(mask_list)

        # normalize bounding box
        bbox_np = self.bboxes_all[index]
        bbox = torch.tensor(bbox_np, dtype=torch.float)
        transed_img, transed_mask, transed_bbox = self.pre_process_batch(img, mask, bbox)

        # check flag
        assert transed_img.shape == (3, 800, 1088)
        assert transed_bbox.shape[0] == transed_mask.shape[0]
        return transed_img, label, transed_mask, transed_bbox

    def __len__(self):
        # return len(self.imgs_data)
        return self.images_h5['data'].shape[0]

    # This function take care of the pre-process of img,mask,bbox
    # in the input mini-batch
    # input:
        # img: 3*300*400
        # mask: 3*300*400
        # bbox: n_box*4
    def pre_process_batch(self, img, mask, bbox):
        assert isinstance(img, torch.Tensor)
        assert isinstance(mask, torch.Tensor)
        assert isinstance(bbox, torch.Tensor)
        # image
        img = self.torch_interpolate(img, 800, 1066)  # (3, 800, 1066)
        img = torchvision.transforms.functional.normalize(img, mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])
        img = F.pad(img, [11, 11])

        # mask: (N_obj * 300 * 400)
        mask = self.torch_interpolate(mask, 800, 1066)  # (N_obj, 800, 1066)
        mask = F.pad(mask, [11, 11])                    # (N_obj, 800, 1088)

        # normalize bounding box
        bbox_normed = torch.zeros_like(bbox)
        for i in range(bbox.shape[0]):
            bbox_normed[i, 0] = bbox[i, 0] / 400.0
            bbox_normed[i, 1] = bbox[i, 1] / 300.0
            bbox_normed[i, 2] = bbox[i, 2] / 400.0
            bbox_normed[i, 3] = bbox[i, 3] / 300.0
        assert torch.max(bbox_normed) <= 1.0
        assert torch.min(bbox_normed) >= 0.0
        bbox = bbox_normed

        # check flag
        assert img.shape == (3, 800, 1088)
        # todo (jianxiong): following commmented was provided in code release
        # assert bbox.shape[0] == mask.squeeze(0).shape[0]
        assert bbox.shape[0] == mask.shape[0]
        return img, mask, bbox

    def torch_interpolate(self, x, H, W):
        """
        A quick wrapper fucntion for torch interpolate
        :return:
        """
        assert isinstance(x, torch.Tensor)
        C = x.shape[0]
        # require input: mini-batch x channels x [optional depth] x [optional height] x width
        tensor_in = x.view((1, 1, C, 300, 400))
        tensor_out = F.interpolate(tensor_in, (C, H, W))
        tensor_out = tensor_out.view((C, H, W))
        return tensor_out

    @staticmethod
    def unnormalize_bbox(bbox):
        """
        Unnormalize one bbox annotation. from 0-1 => 0 - 1088
        x_res = x * 1066 + 11
        y_res = x * 800
        :param bbox: the normalized bounding box (4,)
        :return: the absolute bounding box location (4,)
        """
        bbox_res = torch.tensor(bbox, dtype=torch.float)
        bbox_res[0] = bbox[0] * 1066 + 11
        bbox_res[1] = bbox[1] * 800
        bbox_res[2] = bbox[2] * 1066 + 11
        bbox_res[3] = bbox[3] * 800
        return bbox_res

    @staticmethod
    def unnormalize_img(img):
        """
        Unnormalize image to [0, 1]
        :param img:
        :return:
        """
        assert img.shape == (3, 800, 1088)
        img = torchvision.transforms.functional.normalize(img, mean=[0.0, 0.0, 0.0],
                                                          std=[1.0/0.229, 1.0/0.224, 1.0/0.225])
        img = torchvision.transforms.functional.normalize(img, mean=[-0.485, -0.456, -0.406],
                                                          std=[1.0, 1.0, 1.0])
        return img

class BuildDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

    # output:
        # img: (bz, 3, 800, 1088)
        # label_list: list, len:bz, each (n_obj,)
        # transed_mask_list: list, len:bz, each (n_obj, 800,1088)
        # transed_bbox_list: list, len:bz, each (n_obj, 4)
    def collect_fn(self, batch):
        transed_img_list = []
        label_list = []
        transed_mask_list = []
        transed_bbox_list = []
        for transed_img, label, transed_mask, transed_bbox in batch:
            transed_img_list.append(transed_img)
            label_list.append(label)
            transed_mask_list.append(transed_mask)
            transed_bbox_list.append(transed_bbox)
        return torch.stack(transed_img_list, dim=0), label_list, transed_mask_list, transed_bbox_list

    def loader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers,
                          collate_fn=self.collect_fn)

## Visualize debugging
if __name__ == '__main__':
    # file path and make a list
    # todo (jianxiong): change this back before submitting
    # imgs_path = './data/hw3_mycocodata_img_comp_zlib.h5'
    # masks_path = './data/hw3_mycocodata_mask_comp_zlib.h5'
    # labels_path = './data/hw3_mycocodata_labels_comp_zlib.npy'
    # bboxes_path = './data/hw3_mycocodata_bboxes_comp_zlib.npy'

    imgs_path = '/workspace/data/hw3_mycocodata_img_comp_zlib.h5'
    masks_path = '/workspace/data/hw3_mycocodata_mask_comp_zlib.h5'
    labels_path = '/workspace/data/hw3_mycocodata_labels_comp_zlib.npy'
    bboxes_path = '/workspace/data/hw3_mycocodata_bboxes_comp_zlib.npy'
    os.makedirs("testfig", exist_ok=True)

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

    batch_size = 2
    train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_loader = train_build_loader.loader()
    test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = test_build_loader.loader()

    mask_color_list = ["jet", "ocean", "Spectral", "spring", "cool"]
    # mask_colors = [np.array([1, 0, 0]),
    #                np.array([0, 0, 1]),
    #                np.array([0, 1, 0]),
    #                np.array([1, 1, 0]),
    #                np.array([1, 0, 1])]
    # loop the image
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for iter, data in enumerate(train_loader, 0):

        img, label, mask, bbox = [data[i] for i in range(len(data))]
        # check flag
        assert img.shape == (batch_size, 3, 800, 1088)
        assert len(mask) == batch_size

        label = [label_img.to(device) for label_img in label]
        mask = [mask_img.to(device) for mask_img in mask]
        bbox = [bbox_img.to(device) for bbox_img in bbox]


        # plot the origin img
        for i in range(batch_size):
            ## TODO: plot images with annotations
            fig,ax = plt.subplots(1)
            # the input image: to (800, 1088, 3)
            alpha = 0.2
            # img_vis = alpha * BuildDataset.unnormalize_img(img[i])
            img_vis = alpha * img[i]
            img_vis = img_vis.permute((1, 2, 0)).cpu().numpy()
            # assert np.max(img_vis) <= 0.7
            # assert np.min(img_vis) >= 0.0

            # object mask: assign color with instance id
            # for obj_i, obj_mask in enumerate(mask[i], 0):
            #     img_mask = np.zeros((800, 1088, 3))
            #     img_mask[:, :, 0] = obj_mask
            #     img_mask[:, :, 1] = obj_mask
            #     img_mask[:, :, 2] = obj_mask
            #     img_mask = img_mask * mask_colors[obj_i]
            #     img_vis = img_vis + 0.3 * img_mask

            # object mask: assign color with class label
            for obj_i, obj_mask in enumerate(mask[i], 0):
                obj_label = label[i][obj_i]
                if obj_label == 1:                  # car
                    color_channel = 2
                elif obj_label == 2:                # people
                    color_channel = 1
                elif obj_label == 3:
                    color_channel = 0
                else:                               # dataset can not handle
                    raise RuntimeError("[ERROR] obj_label = {}".format(obj_label))
                img_vis[:, :, color_channel] = img_vis[:, :, color_channel] + (1-alpha) * obj_mask.cpu().numpy()

            # overlapping objects
            img_vis = np.clip(img_vis, 0, 1)
            ax.imshow(img_vis)

            # bounding box
            for obj_i, obj_bbox_normed in enumerate(bbox[i], 0):
                obj_bbox = BuildDataset.unnormalize_bbox(obj_bbox_normed)
                obj_w = obj_bbox[2] - obj_bbox[0]
                obj_h = obj_bbox[3] - obj_bbox[1]
                rect = patches.Rectangle((obj_bbox[0], obj_bbox[1]), obj_w, obj_h, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)

            plt.savefig("./testfig/visualtrainset_{}_{}_.png".format(iter, i))
            plt.show()

        if iter == 10:
            break

