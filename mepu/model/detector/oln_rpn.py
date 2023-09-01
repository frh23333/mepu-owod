# Copyright (c) Facebook, Inc. and its affiliates.
from asyncio.log import logger
from turtle import pos
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import collections
from ...utils.utils import get_centerness, dense_box_regression_loss

import detectron2.utils.comm as comm
from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, cat
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.utils.registry import Registry

from detectron2.modeling.anchor_generator import build_anchor_generator
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.sampling import subsample_labels
from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY
from detectron2.modeling.proposal_generator.proposal_utils import find_top_rpn_proposals


RPN_HEAD_REGISTRY = Registry("RPN_HEAD")

def build_rpn_head(cfg, input_shape):
    """
    Build an RPN head defined by `cfg.MODEL.RPN.HEAD_NAME`.
    """
    name = cfg.MODEL.RPN.HEAD_NAME
    return RPN_HEAD_REGISTRY.get(name)(cfg, input_shape)


@RPN_HEAD_REGISTRY.register()
class OLN_RPNHead(nn.Module):
    """
    Standard RPN classification and regression heads described in :paper:`Faster R-CNN`.
    Uses a 3x3 conv to produce a shared hidden state from which one 1x1 conv predicts
    objectness logits for each anchor and a second 1x1 conv predicts bounding-box deltas
    specifying how to deform each anchor into an object proposal.
    """

    @configurable
    def __init__(
        self, *, in_channels: int, num_anchors: int, box_dim: int = 4, 
        conv_dims: List[int] = (-1,), enable_oln: bool = False,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            in_channels (int): number of input feature channels. When using multiple
                input features, they must have the same number of channels.
            num_anchors (int): number of anchors to predict for *each spatial position*
                on the feature map. The total number of anchors for each
                feature map will be `num_anchors * H * W`.
            box_dim (int): dimension of a box, which is also the number of box regression
                predictions to make for each anchor. An axis aligned box has
                box_dim=4, while a rotated box has box_dim=5.
            conv_dims (list[int]): a list of integers representing the output channels
                of N conv layers. Set it to -1 to use the same number of output channels
                as input channels.
        """
        super().__init__()
        cur_channels = in_channels
        self.enable_oln = enable_oln
        # Keeping the old variable names and structure for backwards compatiblity.
        # Otherwise the old checkpoints will fail to load.
        if len(conv_dims) == 1:
            out_channels = cur_channels if conv_dims[0] == -1 else conv_dims[0]
            # 3x3 conv for the hidden representation
            self.conv = self._get_rpn_conv(cur_channels, out_channels)
            cur_channels = out_channels
        else:
            self.conv = nn.Sequential()
            for k, conv_dim in enumerate(conv_dims):
                out_channels = cur_channels if conv_dim == -1 else conv_dim
                if out_channels <= 0:
                    raise ValueError(
                        f"Conv output channels should be greater than 0. Got {out_channels}"
                    )
                conv = self._get_rpn_conv(cur_channels, out_channels)
                self.conv.add_module(f"conv{k}", conv)
                cur_channels = out_channels

        # 1x1 conv for predicting objectness logits
        self.objectness_logits = nn.Conv2d(cur_channels, num_anchors, kernel_size=1, stride=1)
        # 1x1 conv for predicting box2box transform deltas
        self.anchor_deltas = nn.Conv2d(cur_channels, num_anchors * box_dim, kernel_size=1, stride=1) 
        if self.enable_oln:
            if len(conv_dims) == 1:
                out_channels = cur_channels if conv_dims[0] == -1 else conv_dims[0]
                # 3x3 conv for the hidden representation
                self.conv_oln = self._get_rpn_conv(cur_channels, out_channels)
                cur_channels = out_channels
            else:
                self.conv_oln = nn.Sequential()
                for k, conv_dim in enumerate(conv_dims):
                    out_channels = cur_channels if conv_dim == -1 else conv_dim
                    if out_channels <= 0:
                        raise ValueError(
                            f"Conv output channels should be greater than 0. Got {out_channels}"
                        )
                    conv = self._get_rpn_conv(cur_channels, out_channels)
                    self.conv_oln.add_module(f"conv{k}", conv)
                    cur_channels = out_channels
            self.iou_score = nn.Conv2d(cur_channels, num_anchors, kernel_size=1, stride=1)
            self.centerness_score = nn.Conv2d(cur_channels, num_anchors, kernel_size=1, stride=1)
        

        # Keeping the order of weights initialization same for backwards compatiblility.
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, std=0.01)
                nn.init.constant_(layer.bias, 0)

    def _get_rpn_conv(self, in_channels, out_channels):
        return Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            activation=nn.ReLU(),
        )

    @classmethod
    def from_config(cls, cfg, input_shape):
        # Standard RPN is shared across levels:
        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"
        in_channels = in_channels[0]

        # RPNHead should take the same input as anchor generator
        # NOTE: it assumes that creating an anchor generator does not have unwanted side effect.
        anchor_generator = build_anchor_generator(cfg, input_shape)
        num_anchors = anchor_generator.num_anchors
        box_dim = anchor_generator.box_dim
        assert (
            len(set(num_anchors)) == 1
        ), "Each level must have the same number of anchors per spatial position"
        return {
            "in_channels": in_channels,
            "num_anchors": num_anchors[0],
            "box_dim": box_dim,
            "conv_dims": cfg.MODEL.RPN.CONV_DIMS,
            "enable_oln": cfg.OPENSET.ENABLE_OLN,
        }

    def forward(self, features: List[torch.Tensor]):
        """
        Args:
            features (list[Tensor]): list of feature maps

        Returns:
            list[Tensor]: A list of L elements.
                Element i is a tensor of shape (N, A, Hi, Wi) representing
                the predicted objectness logits for all anchors. A is the number of cell anchors.
            list[Tensor]: A list of L elements. Element i is a tensor of shape
                (N, A*box_dim, Hi, Wi) representing the predicted "deltas" used to transform anchors
                to proposals.
        """

        pred_objectness_logits = []
        pred_anchor_deltas = []
        pred_iou_score = []
        pred_centerness_score = []
        # recons_error_by_level = []
        for x in features:
            t = self.conv(x)
            pred_objectness_logits.append(self.objectness_logits(t))
            pred_anchor_deltas.append(self.anchor_deltas(t))
            if self.enable_oln:
                t_oln = self.conv_oln(x)
                pred_iou_score.append(self.iou_score(t_oln).sigmoid())
                pred_centerness_score.append(self.centerness_score(t_oln).sigmoid())
                
        return pred_objectness_logits, pred_anchor_deltas, pred_iou_score, pred_centerness_score


@PROPOSAL_GENERATOR_REGISTRY.register()
class OLN_RPN(nn.Module):
    """
    Region Proposal Network, introduced by :paper:`Faster R-CNN`.
    """
    @configurable
    def __init__(
        self,
        *,
        in_features: List[str],
        head: nn.Module,
        anchor_generator: nn.Module,
        anchor_matcher: Matcher,
        anchor_matcher_oln: Matcher, 
        box2box_transform: Box2BoxTransform,
        batch_size_per_image: int,
        positive_fraction: float,
        positive_fraction_oln: float,
        pre_nms_topk: Tuple[float, float],
        post_nms_topk: Tuple[float, float],
        pre_nms_topk_oln: Tuple[float, float],
        post_nms_topk_oln: Tuple[float, float],
        nms_thresh: float = 0.7,
        min_box_size: float = 0.0,
        anchor_boundary_thresh: float = -1.0,
        loss_weight: Union[float, Dict[str, float]] = 1.0,
        box_reg_loss_type: str = "smooth_l1",
        smooth_l1_beta: float = 0.0,
        enable_oln: bool = False,
        batch_size_per_image_oln: int = 256, 
        oln_inference: bool = False, 
    ):
        """
        NOTE: this interface is experimental.

        Args:
            in_features (list[str]): list of names of input features to use
            head (nn.Module): a module that predicts logits and regression deltas
                for each level from a list of per-level features
            anchor_generator (nn.Module): a module that creates anchors from a
                list of features. Usually an instance of :class:`AnchorGenerator`
            anchor_matcher (Matcher): label the anchors by matching them with ground truth.
            box2box_transform (Box2BoxTransform): defines the transform from anchors boxes to
                instance boxes
            batch_size_per_image (int): number of anchors per image to sample for training
            positive_fraction (float): fraction of foreground anchors to sample for training
            pre_nms_topk (tuple[float]): (train, test) that represents the
                number of top k proposals to select before NMS, in
                training and testing.
            post_nms_topk (tuple[float]): (train, test) that represents the
                number of top k proposals to select after NMS, in
                training and testing.
            nms_thresh (float): NMS threshold used to de-duplicate the predicted proposals
            min_box_size (float): remove proposal boxes with any side smaller than this threshold,
                in the unit of input image pixels
            anchor_boundary_thresh (float): legacy option
            loss_weight (float|dict): weights to use for losses. Can be single float for weighting
                all rpn losses together, or a dict of individual weightings. Valid dict keys are:
                    "loss_rpn_cls" - applied to classification loss
                    "loss_rpn_loc" - applied to box regression loss
            box_reg_loss_type (str): Loss type to use. Supported losses: "smooth_l1", "giou".
            smooth_l1_beta (float): beta parameter for the smooth L1 regression loss. Default to
                use L1 loss. Only used when `box_reg_loss_type` is "smooth_l1"
        """
        super().__init__()
        self.in_features = in_features
        self.rpn_head = head
        self.anchor_generator = anchor_generator
        self.anchor_matcher = anchor_matcher
        self.box2box_transform = box2box_transform
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        # Map from self.training state to train/test settings
        self.pre_nms_topk = {True: pre_nms_topk[0], False: pre_nms_topk[1]}
        self.post_nms_topk = {True: post_nms_topk[0], False: post_nms_topk[1]}
        self.nms_thresh = nms_thresh
        self.min_box_size = float(min_box_size)
        self.anchor_boundary_thresh = anchor_boundary_thresh
        if isinstance(loss_weight, float):
            loss_weight = {"loss_rpn_cls": loss_weight, "loss_rpn_loc": loss_weight}
        self.loss_weight = loss_weight
        self.box_reg_loss_type = box_reg_loss_type
        self.smooth_l1_beta = smooth_l1_beta
        # self._logger = logging.getLogger(__name__)
        
        self.anchor_matcher_oln = anchor_matcher_oln
        self.enable_oln = enable_oln
        self.batch_size_per_image_oln = batch_size_per_image_oln
        self.oln_inference = oln_inference
        self.pre_nms_topk_oln = {True: pre_nms_topk_oln[0], False: pre_nms_topk_oln[1]}
        self.post_nms_topk_oln = {True: post_nms_topk_oln[0], False: post_nms_topk_oln[1]}
        self.positive_fraction_oln = positive_fraction_oln
        
    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        in_features = cfg.MODEL.RPN.IN_FEATURES
        ret = {
            "in_features": in_features,
            "min_box_size": cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE,
            "nms_thresh": cfg.MODEL.RPN.NMS_THRESH,
            "batch_size_per_image": cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE,
            "positive_fraction": cfg.MODEL.RPN.POSITIVE_FRACTION,
            "loss_weight": {
                "loss_rpn_cls": cfg.MODEL.RPN.LOSS_WEIGHT,
                "loss_rpn_loc": cfg.MODEL.RPN.BBOX_REG_LOSS_WEIGHT * cfg.MODEL.RPN.LOSS_WEIGHT,
            },
            "anchor_boundary_thresh": cfg.MODEL.RPN.BOUNDARY_THRESH,
            "box2box_transform": Box2BoxTransform(weights=cfg.MODEL.RPN.BBOX_REG_WEIGHTS),
            "box_reg_loss_type": cfg.MODEL.RPN.BBOX_REG_LOSS_TYPE,
            "smooth_l1_beta": cfg.MODEL.RPN.SMOOTH_L1_BETA,

            "enable_oln": cfg.OPENSET.ENABLE_OLN,
            "batch_size_per_image_oln": cfg.OPENSET.OLN.BATCH_SIZE_PER_IMAGE, 
            "oln_inference": cfg.OPENSET.OLN_INFERENCE,
            "positive_fraction_oln": cfg.OPENSET.OLN.POSITIVE_FRACTION,
        }

        ret["pre_nms_topk"] = (cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN, cfg.MODEL.RPN.PRE_NMS_TOPK_TEST)
        ret["post_nms_topk"] = (cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN, cfg.MODEL.RPN.POST_NMS_TOPK_TEST)

        ret["anchor_generator"] = build_anchor_generator(cfg, [input_shape[f] for f in in_features])

        ret["anchor_matcher"] = Matcher(
            cfg.MODEL.RPN.IOU_THRESHOLDS, cfg.MODEL.RPN.IOU_LABELS, allow_low_quality_matches=True
        )
        ret["anchor_matcher_oln"] = Matcher(
            cfg.OPENSET.OLN.IOU_THRESHOLDS, cfg.OPENSET.OLN.IOU_LABELS, allow_low_quality_matches=True
        )

        ret["head"] = build_rpn_head(cfg, [input_shape[f] for f in in_features])

        ret["pre_nms_topk_oln"] = (cfg.OPENSET.OLN.PRE_NMS_TOPK_TRAIN, cfg.OPENSET.OLN.PRE_NMS_TOPK_TEST)
        ret["post_nms_topk_oln"] = (cfg.OPENSET.OLN.POST_NMS_TOPK_TRAIN, cfg.OPENSET.OLN.POST_NMS_TOPK_TEST)


        return ret

    def _subsample_labels(self, label):
        """
        Randomly sample a subset of positive and negative examples, and overwrite
        the label vector to the ignore value (-1) for all elements that are not
        included in the sample.

        Args:
            labels (Tensor): a vector of -1, 0, 1. Will be modified in-place and returned.
        """
        pos_idx, neg_idx = subsample_labels(
            label, self.batch_size_per_image, self.positive_fraction, 0
        )
        # Fill with the ignore label (-1), then set positive and negative labels
        label.fill_(-1)
        label.scatter_(0, pos_idx, 1)
        label.scatter_(0, neg_idx, 0)
        return label

    def _subsample_labels_oln(self, label):
        """
        Randomly sample a subset of positive and negative examples, and overwrite
        the label vector to the ignore value (-1) for all elements that are not
        included in the sample.

        Args:
            labels (Tensor): a vector of -1, 0, 1. Will be modified in-place and returned.
        """
        pos_idx, neg_idx = subsample_labels(
            label, self.batch_size_per_image_oln, 0.5, 0
        )
        # Fill with the ignore label (-1), then set positive and negative labels
        label.fill_(-1)
        label.scatter_(0, pos_idx, 1)
        label.scatter_(0, neg_idx, 0)
        return label

    @torch.jit.unused
    @torch.no_grad()
    def label_and_sample_anchors(
        self, anchors: List[Boxes], gt_instances: List[Instances], 
    ):
        """
        Args:
            anchors (list[Boxes]): anchors for each feature map.
            gt_instances: the ground-truth instances for each image.

        Returns:
            list[Tensor]:
                List of #img tensors. i-th element is a vector of labels whose length is
                the total number of anchors across all feature maps R = sum(Hi * Wi * A).
                Label values are in {-1, 0, 1}, with meanings: -1 = ignore; 0 = negative
                class; 1 = positive class.
            list[Tensor]:
                i-th element is a Rx4 tensor. The values are the matched gt boxes for each
                anchor. Values are undefined for those anchors not labeled as 1.
        """
        anchors = Boxes.cat(anchors)
        
        gt_boxes = [x.gt_boxes for x in gt_instances]
        image_sizes = [x.image_size for x in gt_instances]
        gt_classes = [x.gt_classes for x in gt_instances]
        soft_labels = [x.gt_classes for x in gt_instances]
        if gt_instances[0].has("soft_labels"):
            soft_labels = [x.soft_labels for x in gt_instances]

        unk_idxs = []
        for gt_classes_i in gt_classes:
            unk_idxs_i = []
            for idx, gt_class in enumerate(gt_classes_i):
                if gt_class == 80:
                    unk_idxs_i.append(idx)
            unk_idxs.append(unk_idxs_i)

        gt_labels = []
        matched_gt_boxes = []
        matched_gt_classes = []
        matched_soft_labels = []
        matched_idx_list = []
        
        for image_size_i, gt_boxes_i, gt_classes_i, soft_labels_i in zip(image_sizes, gt_boxes, \
            gt_classes, soft_labels):
            """
            image_size_i: (h, w) for the i-th image
            gt_boxes_i: ground-truth boxes for i-th image
            """
            match_quality_matrix = retry_if_cuda_oom(pairwise_iou)(gt_boxes_i, anchors)
            matched_idxs, gt_labels_i = retry_if_cuda_oom(self.anchor_matcher)(match_quality_matrix)
            # Matching is memory-expensive and may result in CPU tensors. But the result is small
            gt_labels_i = gt_labels_i.to(device=gt_boxes_i.device)
            del match_quality_matrix

            if self.anchor_boundary_thresh >= 0:
                # Discard anchors that go out of the boundaries of the image
                # NOTE: This is legacy functionality that is turned off by default in Detectron2
                anchors_inside_image = anchors.inside_box(image_size_i, self.anchor_boundary_thresh)
                gt_labels_i[~anchors_inside_image] = -1
            
            gt_labels_i = self._subsample_labels(gt_labels_i)

            matched_gt_classes_i = gt_classes_i[matched_idxs]
            matched_gt_boxes_i = gt_boxes_i[matched_idxs].tensor

            if gt_instances[0].has("soft_labels"):
                matched_soft_labels_i = soft_labels_i[matched_idxs]

            gt_labels.append(gt_labels_i)  # N,AHW
            matched_gt_boxes.append(matched_gt_boxes_i)
            matched_gt_classes.append(matched_gt_classes_i)
            matched_idx_list.append(matched_idxs)
            if gt_instances[0].has("soft_labels"):
                matched_soft_labels.append(matched_soft_labels_i)
        return gt_labels, matched_gt_boxes, matched_gt_classes, matched_idx_list, unk_idxs, \
            matched_soft_labels
    
    @torch.jit.unused
    @torch.no_grad()
    def label_and_sample_anchors_oln(
        self, anchors: List[Boxes], gt_instances: List[Instances], 
    ):
        """
        Args:
            anchors (list[Boxes]): anchors for each feature map.
            gt_instances: the ground-truth instances for each image.

        Returns:
            list[Tensor]:
                List of #img tensors. i-th element is a vector of labels whose length is
                the total number of anchors across all feature maps R = sum(Hi * Wi * A).
                Label values are in {-1, 0, 1}, with meanings: -1 = ignore; 0 = negative
                class; 1 = positive class.
            list[Tensor]:
                i-th element is a Rx4 tensor. The values are the matched gt boxes for each
                anchor. Values are undefined for those anchors not labeled as 1.
        """
        anchors = Boxes.cat(anchors)
        
        gt_boxes = [x.gt_boxes for x in gt_instances]
        soft_labels = [x.gt_classes for x in gt_instances]
        if gt_instances[0].has("soft_labels"):
            soft_labels = [x.soft_labels for x in gt_instances]
        
        gt_labels_oln = []
        matched_gt_boxes_oln = []
        matched_soft_labels_oln = []
        iou_imgs = []
        for gt_boxes_i, soft_labels_i in zip(gt_boxes, soft_labels):
            """
            image_size_i: (h, w) for the i-th image
            gt_boxes_i: ground-truth boxes for i-th image
            """
            match_quality_matrix = retry_if_cuda_oom(pairwise_iou)(gt_boxes_i, anchors)
            matched_idxs_oln, gt_labels_i_oln = retry_if_cuda_oom(self.anchor_matcher_oln)\
                (match_quality_matrix)
            iou_i = match_quality_matrix.max(dim = 0)[0]
            gt_labels_i_oln = gt_labels_i_oln.to(device=gt_boxes_i.device)
            del match_quality_matrix
            
            gt_labels_i_oln = self._subsample_labels_oln(gt_labels_i_oln)
            matched_gt_boxes_i_oln = gt_boxes_i[matched_idxs_oln].tensor
            if gt_instances[0].has("soft_labels"):
                matched_soft_labels_i_oln = soft_labels_i[matched_idxs_oln]

            iou_imgs.append(iou_i)
            gt_labels_oln.append(gt_labels_i_oln)  # N,AHW
            matched_gt_boxes_oln.append(matched_gt_boxes_i_oln)
            if gt_instances[0].has("soft_labels"):
                matched_soft_labels_oln.append(matched_soft_labels_i_oln)

        return iou_imgs, gt_labels_oln, matched_gt_boxes_oln, matched_soft_labels_oln

    @torch.jit.unused
    def losses(
        self,
        anchors: List[Boxes],
        pred_objectness_logits: List[torch.Tensor],
        pred_anchor_deltas: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
        gt_boxes: List[torch.Tensor],
        soft_labels: List[torch.Tensor], 
        gt_classes: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Return the losses from a set of RPN predictions and their associated ground-truth.

        Args:
            anchors (list[Boxes or RotatedBoxes]): anchors for each feature map, each
                has shape (Hi*Wi*A, B), where B is box dimension (4 or 5).
            pred_objectness_logits (list[Tensor]): A list of L elements.
                Element i is a tensor of shape (N, Hi*Wi*A) representing
                the predicted objectness logits for all anchors.
            gt_labels (list[Tensor]): Output of :meth:`label_and_sample_anchors`.
            pred_anchor_deltas (list[Tensor]): A list of L elements. Element i is a tensor of shape
                (N, Hi*Wi*A, 4 or 5) representing the predicted "deltas" used to transform anchors
                to proposals.
            gt_boxes (list[Tensor]): Output of :meth:`label_and_sample_anchors`.

        Returns:
            dict[loss name -> loss value]: A dict mapping from loss name to loss value.
                Loss names are: `loss_rpn_cls` for objectness classification and
                `loss_rpn_loc` for proposal localization.
        """
        num_images = len(gt_labels)
        storage = get_event_storage()
            
        gt_labels = torch.stack(gt_labels)  # (N, sum(Hi*Wi*Ai))
        if (len(soft_labels) == 0):
            pos_mask = gt_labels == 1
        else:
            soft_labels = torch.stack(soft_labels)
            pos_mask = (gt_labels == 1) & (soft_labels == 1.0)

        localization_loss = dense_box_regression_loss(
            anchors,
            self.box2box_transform,
            pred_anchor_deltas,
            gt_boxes,
            pos_mask,
            box_reg_loss_type=self.box_reg_loss_type,
            smooth_l1_beta=self.smooth_l1_beta,
            reduction = "sum"
        )
        
        if (len(soft_labels) == 0):
            valid_mask = gt_labels >= 0
            objectness_loss = F.binary_cross_entropy_with_logits(
                cat(pred_objectness_logits, dim=1)[valid_mask].float(),
                gt_labels[valid_mask].float(),
                reduction="sum",
            )
        else:
            # soft_labels = torch.stack(soft_labels)
            valid_mask = gt_labels >= 0
            soft_labels[gt_labels == 0] = 1
            objectness_loss = F.binary_cross_entropy_with_logits(
                cat(pred_objectness_logits, dim=1)[valid_mask].float(),
                gt_labels[valid_mask].float(),
                reduction="none",
            )
            objectness_loss = (soft_labels[valid_mask] * objectness_loss).sum()

        normalizer = self.batch_size_per_image * num_images
        losses = {}

        losses["loss_rpn_loc"] = localization_loss / normalizer
        losses["loss_rpn_cls"] = objectness_loss / normalizer

        losses = {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}
        return losses

    @torch.jit.unused
    def losses_oln(
        self,
        anchors: List[Boxes],
        pred_iou_score: List[torch.Tensor],
        pred_centerness_score: List[torch.Tensor],
        iou_imgs: List[torch.Tensor],
        gt_labels_oln: List[torch.Tensor], 
        gt_boxes_oln: List[torch.Tensor], 
        soft_labels_oln: List[torch.Tensor], 
    ) -> Dict[str, torch.Tensor]:
        """
        Return the losses from a set of RPN predictions and their associated ground-truth.

        Args:
            anchors (list[Boxes or RotatedBoxes]): anchors for each feature map, each
                has shape (Hi*Wi*A, B), where B is box dimension (4 or 5).
            pred_objectness_logits (list[Tensor]): A list of L elements.
                Element i is a tensor of shape (N, Hi*Wi*A) representing
                the predicted objectness logits for all anchors.
            gt_labels (list[Tensor]): Output of :meth:`label_and_sample_anchors`.
            pred_anchor_deltas (list[Tensor]): A list of L elements. Element i is a tensor of shape
                (N, Hi*Wi*A, 4 or 5) representing the predicted "deltas" used to transform anchors
                to proposals.
            gt_boxes (list[Tensor]): Output of :meth:`label_and_sample_anchors`.

        Returns:
            dict[loss name -> loss value]: A dict mapping from loss name to loss value.
                Loss names are: `loss_rpn_cls` for objectness classification and
                `loss_rpn_loc` for proposal localization.
        """
        num_images = len(gt_labels_oln)

        if self.enable_oln:
            gt_labels_oln = torch.stack(gt_labels_oln)  # (N, sum(Hi*Wi*Ai))
            valid_mask_oln = gt_labels_oln >= 0

            anchors_tensor = Boxes.cat(anchors).tensor
            num_anchors = anchors_tensor.shape[0]
            anchors_x = ((anchors_tensor[:, 0] + anchors_tensor[:, 2])/2).reshape([num_anchors, 1])
            anchors_y = ((anchors_tensor[:, 1] + anchors_tensor[:, 3])/2).reshape([num_anchors, 1])
            anchors_center = torch.cat([anchors_x, anchors_y], dim = 1)
            anchors_center = [anchors_center for i in range(num_images)]
            anchors_center = torch.stack(anchors_center)
            gt_boxes_oln = torch.stack(gt_boxes_oln)
            
            centerness_target = get_centerness(anchors_center[valid_mask_oln], \
                gt_boxes_oln[valid_mask_oln])
            iou_target = torch.stack(iou_imgs)[valid_mask_oln]
            pred_iou_score = cat(pred_iou_score, dim=1)
            pred_centerness_score = cat(pred_centerness_score, dim=1)
            if (len(soft_labels_oln) == 0):
                loss_iou = F.smooth_l1_loss(
                    pred_iou_score[valid_mask_oln],
                    iou_target, 
                    beta=0.0, 
                    reduction="mean")
                loss_centerness = F.smooth_l1_loss(
                    pred_centerness_score[valid_mask_oln],
                    centerness_target, 
                    beta=0.0, 
                    reduction="mean")
            else:
                soft_labels_oln = torch.stack(soft_labels_oln)
                loss_iou = F.smooth_l1_loss(
                    pred_iou_score[valid_mask_oln],
                    iou_target, 
                    beta=0.0, 
                    reduction="none")
                loss_centerness = F.smooth_l1_loss(
                    pred_centerness_score[valid_mask_oln],
                    centerness_target, 
                    beta=0.0, 
                    reduction="none")

                loss_iou = (loss_iou * soft_labels_oln[valid_mask_oln]).mean()
                loss_centerness = (loss_centerness * soft_labels_oln[valid_mask_oln]).mean()

        losses = {}
        losses["loss_rpn_iou"] = loss_iou
        losses["loss_rpn_centerness"] = loss_centerness

        return losses
            

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        gt_instances: Optional[List[Instances]] = None,
    ):
        """
        Args:
            images (ImageList): input images of length `N`
            features (dict[str, Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            gt_instances (list[Instances], optional): a length `N` list of `Instances`s.
                Each `Instances` stores ground-truth instances for the corresponding image.

        Returns:
            proposals: list[Instances]: contains fields "proposal_boxes", "objectness_logits"
            loss: dict[Tensor] or None
        """
        features = [features[f] for f in self.in_features]
        anchors = self.anchor_generator(features)

        pred_objectness_logits, pred_anchor_deltas, pred_iou_score, \
            pred_centerness_score = self.rpn_head(features)

        # Transpose the Hi*Wi*A dimension to the middle:
        pred_objectness_logits = [
            # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
            score.permute(0, 2, 3, 1).flatten(1)
            for score in pred_objectness_logits
        ]
        pred_iou_score = [
            # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
            score.permute(0, 2, 3, 1).flatten(1)
            for score in pred_iou_score
        ]
        pred_centerness_score = [
            # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
            score.permute(0, 2, 3, 1).flatten(1)
            for score in pred_centerness_score
        ]

        pred_anchor_deltas = [
            x.view(x.shape[0], -1, self.anchor_generator.box_dim, x.shape[-2], x.shape[-1])
            .permute(0, 3, 4, 1, 2)
            .flatten(1, -2)
            for x in pred_anchor_deltas
        ]

        if self.training:
            assert gt_instances is not None, "RPN requires gt_instances in training!"
    

            gt_labels, gt_boxes, gt_classes, idx_list, unk_idxs, soft_labels = \
                self.label_and_sample_anchors(anchors, gt_instances)
            
            losses = self.losses(
                anchors, pred_objectness_logits, pred_anchor_deltas, 
                gt_labels, gt_boxes, soft_labels, gt_classes
            )
            if self.enable_oln:
                iou_imgs, gt_labels_oln, gt_boxes_oln, soft_labels_oln = \
                    self.label_and_sample_anchors_oln(anchors, gt_instances)
                losses_oln = self.losses_oln(
                    anchors, pred_iou_score, pred_centerness_score, iou_imgs, gt_labels_oln, \
                        gt_boxes_oln, soft_labels_oln
                )
                losses.update(losses_oln)
        else:
            losses = {}

        if self.training:
            proposals = self.predict_proposals(
                anchors, pred_objectness_logits, pred_anchor_deltas, images.image_sizes
            )
            if self.enable_oln:
                proposals_oln = self.predict_proposals_oln(
                    anchors, pred_centerness_score, pred_anchor_deltas, images.image_sizes
                )
                proposals_oln = self.merge_proposals(
                    proposals_oln, proposals, images.image_sizes
                )
            else:
                proposals_oln = []

            return proposals, losses, proposals_oln

        elif self.oln_inference:
            proposals = self.predict_proposals_oln(
                anchors, pred_centerness_score, pred_anchor_deltas, images.image_sizes
            )
        else:
            proposals = self.predict_proposals(
                anchors, pred_objectness_logits, pred_anchor_deltas, images.image_sizes
            )

        return proposals, losses, []

    def predict_proposals(
        self,
        anchors: List[Boxes],
        pred_objectness_logits: List[torch.Tensor],
        pred_anchor_deltas: List[torch.Tensor],
        image_sizes: List[Tuple[int, int]],
    ):
        """
        Decode all the predicted box regression deltas to proposals. Find the top proposals
        by applying NMS and removing boxes that are too small.

        Returns:
            proposals (list[Instances]): list of N Instances. The i-th Instances
                stores post_nms_topk object proposals for image i, sorted by their
                objectness score in descending order.
        """
        # The proposals are treated as fixed for joint training with roi heads.
        # This approach ignores the derivative w.r.t. the proposal boxes’ coordinates that
        # are also network responses.
        with torch.no_grad():
            pred_proposals = self._decode_proposals(anchors, pred_anchor_deltas)
            return find_top_rpn_proposals(
                pred_proposals,
                pred_objectness_logits,
                image_sizes,
                self.nms_thresh,
                self.pre_nms_topk[self.training],
                self.post_nms_topk[self.training],
                self.min_box_size,
                self.training,
            )

    def predict_proposals_oln(
        self,
        anchors: List[Boxes],
        pred_objectness_logits: List[torch.Tensor],
        pred_anchor_deltas: List[torch.Tensor],
        image_sizes: List[Tuple[int, int]],
    ):
        """
        Decode all the predicted box regression deltas to proposals. Find the top proposals
        by applying NMS and removing boxes that are too small.

        Returns:
            proposals (list[Instances]): list of N Instances. The i-th Instances
                stores post_nms_topk object proposals for image i, sorted by their
                objectness score in descending order.
        """
        # The proposals are treated as fixed for joint training with roi heads.
        # This approach ignores the derivative w.r.t. the proposal boxes’ coordinates that
        # are also network responses.
        with torch.no_grad():
            pred_proposals = self._decode_proposals(anchors, pred_anchor_deltas)
            return find_top_rpn_proposals(
                pred_proposals,
                pred_objectness_logits,
                image_sizes,
                self.nms_thresh,
                self.pre_nms_topk_oln[self.training], 
                self.post_nms_topk_oln[self.training], 
                self.min_box_size,
                self.training,
            )
    
    def merge_proposals(self, proposals_a, proposals_b, image_sizes):
        with torch.no_grad():
            proposals_new = []
            for p_a, p_b, img_size in zip(proposals_a, proposals_b, image_sizes):
                p_new = Instances(img_size)
                proposal_boxes_a = p_a.proposal_boxes.tensor
                proposal_boxes_b = p_b.proposal_boxes.tensor
                proposal_boxes_new = Boxes(torch.cat([proposal_boxes_a, proposal_boxes_b]))
                p_new.proposal_boxes = proposal_boxes_new
                objectness_logits_a = p_a.objectness_logits
                objectness_logits_b = p_b.objectness_logits
                objectness_logits_new = torch.cat([objectness_logits_a, objectness_logits_b])
                p_new.objectness_logits = objectness_logits_new
                proposals_new.append(p_new)
            return proposals_new

    def _decode_proposals(self, anchors: List[Boxes], pred_anchor_deltas: List[torch.Tensor]):
        """
        Transform anchors into proposals by applying the predicted anchor deltas.

        Returns:
            proposals (list[Tensor]): A list of L tensors. Tensor i has shape
                (N, Hi*Wi*A, B)
        """
        N = pred_anchor_deltas[0].shape[0]
        proposals = []
        # For each feature map
        for anchors_i, pred_anchor_deltas_i in zip(anchors, pred_anchor_deltas):
            B = anchors_i.tensor.size(1)
            pred_anchor_deltas_i = pred_anchor_deltas_i.reshape(-1, B)
            # Expand anchors to shape (N*Hi*Wi*A, B)
            anchors_i = anchors_i.tensor.unsqueeze(0).expand(N, -1, -1).reshape(-1, B)
            proposals_i = self.box2box_transform.apply_deltas(pred_anchor_deltas_i, anchors_i)
            # Append feature map proposals with shape (N, Hi*Wi*A, B)
            proposals.append(proposals_i.view(N, -1, B))
        return proposals
