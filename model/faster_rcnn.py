import numpy as np
from torch import nn
from model.vgg16 import decom_vgg16
from model.rpn import RegionProposalNetwork
from model.roi_module import VGG16RoIHead
from utils.config import opt
import torch as t


class FasterRCNN(nn.Module):
    def __init__(self, ratios=[0.5, 1, 2], anchor_scales=[0.5, 1, 2], \
                    loc_normalize_mean = (0., 0., 0., 0.), \
                    loc_normalize_std = (0.1, 0.1, 0.2, 0.2)):
        super(FasterRCNN, self).__init__()
        
        # prepare
        extractor, classifier = decom_vgg16()
        rpn = RegionProposalNetwork(
            512, 512,
            ratios=ratios,
            scales=anchor_scales,
            feat_stride=16
        )
        
        head = VGG16RoIHead(
            n_class=20 + 1,
            roi_size=7,
            spatial_scale=(1. / 16),
            classifier=classifier.cuda()
        )
        self.extractor = extractor.cuda()
        self.rpn = rpn.cuda()
        self.head = head

        # mean and std
        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std


    def forward(self, x, scale=1.):

        img_size = x.shape[2:]

        h = self.extractor(x)
        rpn_locs, rpn_scores, rois, roi_indices, anchor = \
            self.rpn(h, img_size, scale)
        roi_cls_locs, roi_scores = self.head(
            h.cuda(), rois, np.array(roi_indices))
        return roi_cls_locs, roi_scores, rois, roi_indices


    def get_optimizer(self):
        """
        return optimizer, It could be overwriten if you want to specify 
        special optimizer
        """
        lr = opt.lr
        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': lr, 'weight_decay': opt.weight_decay}]
        # if opt.use_adam:
        #     self.optimizer = t.optim.Adam(params)
        # else:
        self.optimizer = t.optim.SGD(params, momentum=0.9)
        return self.optimizer
