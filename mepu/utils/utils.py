import torch
from typing import List, Tuple, Union
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.structures import Boxes
from detectron2.layers import cat, ciou_loss, diou_loss
from fvcore.nn import giou_loss, smooth_l1_loss
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

def get_centerness(anchor_centers:torch.Tensor, gt_boxes:torch.Tensor, ):
    x1g, y1g, x2g, y2g = gt_boxes.unbind(dim=-1)
    x_c, y_c = anchor_centers.unbind(dim=-1)
    l = x_c - x1g
    r = x2g - x_c
    t = y_c - y1g
    b = y2g - y_c
    valid_mask = (l >= 0)&(r >= 0)&(t >= 0)&(b >= 0)
    l[valid_mask == False] = 0
    r[valid_mask == False] = 0
    t[valid_mask == False] = 0
    b[valid_mask == False] = 0
    min_lr = torch.min(l,r)
    max_lr = torch.max(l,r)
    min_tb = torch.min(t,b)
    max_tb = torch.max(t,b)
    objectness_score = torch.sqrt(min_lr * min_tb / (max_lr * max_tb + 1e-3))
    assert not (objectness_score > 1).any()
    return objectness_score

def dense_box_regression_loss(
    anchors: List[Union[Boxes, torch.Tensor]],
    box2box_transform: Box2BoxTransform,
    pred_anchor_deltas: List[torch.Tensor],
    gt_boxes: List[torch.Tensor],
    fg_mask: torch.Tensor,
    box_reg_loss_type="smooth_l1",
    smooth_l1_beta=0.0,
    reduction="sum",
):
    """
    Compute loss for dense multi-level box regression.
    Loss is accumulated over ``fg_mask``.

    Args:
        anchors: #lvl anchor boxes, each is (HixWixA, 4)
        pred_anchor_deltas: #lvl predictions, each is (N, HixWixA, 4)
        gt_boxes: N ground truth boxes, each has shape (R, 4) (R = sum(Hi * Wi * A))
        fg_mask: the foreground boolean mask of shape (N, R) to compute loss on
        box_reg_loss_type (str): Loss type to use. Supported losses: "smooth_l1", "giou",
            "diou", "ciou".
        smooth_l1_beta (float): beta parameter for the smooth L1 regression loss. Default to
            use L1 loss. Only used when `box_reg_loss_type` is "smooth_l1"
    """
    if isinstance(anchors[0], Boxes):
        anchors = type(anchors[0]).cat(anchors).tensor  # (R, 4)
    else:
        anchors = cat(anchors)
    if box_reg_loss_type == "smooth_l1":
        gt_anchor_deltas = [box2box_transform.get_deltas(anchors, k) for k in gt_boxes]
        gt_anchor_deltas = torch.stack(gt_anchor_deltas)  # (N, R, 4)
        loss_box_reg = smooth_l1_loss(
            cat(pred_anchor_deltas, dim=1)[fg_mask],
            gt_anchor_deltas[fg_mask],
            beta=smooth_l1_beta,
            reduction=reduction,
        )
    elif box_reg_loss_type == "giou":
        pred_boxes = [
            box2box_transform.apply_deltas(k, anchors) for k in cat(pred_anchor_deltas, dim=1)
        ]
        loss_box_reg = giou_loss(
            torch.stack(pred_boxes)[fg_mask], torch.stack(gt_boxes)[fg_mask], reduction=reduction
        )
    elif box_reg_loss_type == "diou":
        pred_boxes = [
            box2box_transform.apply_deltas(k, anchors) for k in cat(pred_anchor_deltas, dim=1)
        ]
        loss_box_reg = diou_loss(
            torch.stack(pred_boxes)[fg_mask], torch.stack(gt_boxes)[fg_mask], reduction=reduction
        )
    elif box_reg_loss_type == "ciou":
        pred_boxes = [
            box2box_transform.apply_deltas(k, anchors) for k in cat(pred_anchor_deltas, dim=1)
        ]
        loss_box_reg = ciou_loss(
            torch.stack(pred_boxes)[fg_mask], torch.stack(gt_boxes)[fg_mask], reduction=reduction
        )
    else:
        raise ValueError(f"Invalid dense box regression loss type '{box_reg_loss_type}'")
    return loss_box_reg

def vis_res(img, bboxes_anno, bboxes_pl, save_path):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(img)
    for box in bboxes_anno:
        box = box.reshape([-1])
        rect = mpatches.Rectangle(
            (box[0], box[1]), box[2] - box[0], box[3] - box[1], \
                fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)
    for box in bboxes_pl:
        box = box.reshape([-1])
        rect = mpatches.Rectangle(
            (box[0], box[1]), box[2] - box[0], box[3] - box[1], \
                fill=False, edgecolor='blue', linewidth=1)
        ax.add_patch(rect)
    plt.show()
    plt.savefig(save_path)
    plt.close()