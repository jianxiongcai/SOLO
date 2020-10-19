"""
Author: Jianxiong Cai
Date: Oct 16, 2020
A portable metric tracker for map computation
"""

import numpy as np
import torch
from sklearn.metrics import auc

class MetricTracker:
    """
    Tracker for single class. For each class, there should be a tracker
    For the start of an epoch, call reset()
    At the end of an epoch, call compute_precision_recall()
    """
    def __init__(self, tracker_id):
        self.reset()
        self.tracker_id = tracker_id

    def reset(self):
        # all confidence scores across the batch (see add_match() for doc)
        self.conf_scores = np.zeros((0,))
        self.tp_indicator = np.zeros((0,), np.bool)
        self.match_indice = np.zeros((0,), np.int)
        self.num_gts = 0
        self.precision_recall = dict()

    def add_match(self, conf_scores, tp_indicator, match_indice, num_gt_batch):
        """
        Save all data from each batch, all inputs are of shape (N,)
        :param conf_scores: (numpy array, N) the confidence scores
        :param tp_indicator: (numpy binary array) binary array indicating if it is a true positive or not
        :param match_indice: (numpy array) Int array. If tp_indice[i] is True, match_indice[i] indicate the index for the match
        :param num_gt_batch: number of graound-truths in the batch
        :return:
        """
        if isinstance(conf_scores, torch.Tensor):
            conf_scores = conf_scores.detach().cpu().numpy()
        if isinstance(tp_indicator, torch.Tensor):
            tp_indicator = tp_indicator.detach().cpu().numpy()
        if isinstance(match_indice, torch.Tensor):
            match_indice = match_indice.detach().cpu().numpy()

        assert isinstance(conf_scores, np.ndarray)
        assert isinstance(tp_indicator, np.ndarray)
        assert isinstance(match_indice, np.ndarray)
        assert len(conf_scores) == len(tp_indicator)
        assert len(match_indice) == len(tp_indicator)


        if len(match_indice) == 0:      # skip everything if no prediction at all
            self.num_gts = self.num_gts + num_gt_batch
            return
        if num_gt_batch == 0:
            assert np.all(tp_indicator == False)
        else:          # sanity check when N_gt > 0
            assert np.max(match_indice) < num_gt_batch

        self.conf_scores = np.concatenate([self.conf_scores, conf_scores])
        self.tp_indicator = np.concatenate([self.tp_indicator, tp_indicator])
        # to unique identify object, the target match indice is offseted by current holding gt objects
        match_indice_new = match_indice + self.num_gts
        if self.tracker_id == 0:
            debug = True
        # add to current tracking prediction and gts
        self.match_indice = np.concatenate([self.match_indice, match_indice_new])
        self.num_gts = self.num_gts + num_gt_batch
        for i in range(match_indice_new.shape[0]):
            if (tp_indicator[i]) and (match_indice_new[i] >= self.num_gts):
                raise RuntimeError("Check Failed")

    def compute_precision_recall(self):
        """
        Compute the precision-recall curve, by increasing the confidence threshold
        :return:
            precision_recall: (python dictionary) key: recall, value: precision
        """
        precision_recall = dict()
        n_tp = 0.0
        n_tot_pred = 0.0
        obj_recall_indicator = np.zeros((self.num_gts,), dtype=np.bool)

        # sort the list (ascending order), all 1-d numpy array
        sorted_idx = np.argsort(self.conf_scores)
        sorted_scores = self.conf_scores[sorted_idx]
        sorted_tp_indicator = self.tp_indicator[sorted_idx]
        sorted_match_indice = self.match_indice[sorted_idx]

        # increase threshold and compute precision and recall
        for score, is_tp, match_idx in zip(sorted_scores, sorted_tp_indicator, sorted_match_indice):
            # update stats
            n_tot_pred += 1
            if is_tp:
                n_tp = n_tp + 1
                obj_recall_indicator[match_idx] = True
            # compute precision and recall
            precision = n_tp / n_tot_pred
            recall = np.sum(obj_recall_indicator) / self.num_gts
            precision_recall[recall] = precision

        self.precision_recall = precision_recall
        return precision_recall

    def sorted_pr_curve(self):
        assert len(self.precision_recall) != 0, "[ERROR] Call compute_precision_recall first!"
        recall_sorted = sorted(list(self.precision_recall.keys()))
        precision_sorted = [self.precision_recall[k] for k in recall_sorted]
        return recall_sorted, precision_sorted

    def compute_ap(self):
        assert len(self.precision_recall) != 0, "[ERROR] Call compute_precision_recall first!"
        recall_sorted = sorted(list(self.precision_recall.keys()))
        precision_sorted = [self.precision_recall[k] for k in recall_sorted]
        if len(recall_sorted) > 1:
            # compute auc
            return auc(recall_sorted, precision_sorted)
        else:
            return 0.0
