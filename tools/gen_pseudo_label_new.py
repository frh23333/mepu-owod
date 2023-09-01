import json
from webbrowser import get
import numpy as np
import torch
from tqdm import tqdm
import torchvision
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import argparse
import sys
import shutil
import random
sys.path.append(".")
from mepu.utils.utils import get_centerness
from detectron2.data.datasets.pascal_voc import load_voc_instances

ALL_CLS_NAMES_OWDETR = ["airplane","bicycle","bird","boat","bus","car",
    "cat","cow","dog","horse","motorcycle","sheep",
    "train","elephant","bear","zebra","giraffe","truck","person",
    # t2
    "traffic light","fire hydrant","stop sign",
    "parking meter","bench","chair","dining table",
    "potted plant","backpack","umbrella","handbag","tie",
    "suitcase","microwave","oven","toaster","sink","refrigerator","bed","toilet","couch",
    # t3
    "frisbee","skis","snowboard","sports ball","kite",
    "baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","banana","apple","sandwich","orange","broccoli",
    "carrot","hot dog","pizza","donut","cake",
    # t4
    "laptop","mouse","remote","keyboard","cell phone",
    "book","clock","vase","scissors","teddy bear","hair drier",
    "toothbrush","wine glass","cup","fork","knife","spoon","bowl","tv","bottle",]

ALL_CLS_NAMES_OWOD = [
    # voc
    "airplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "dining table", "dog", "horse", "motorcycle", "person",
    "potted plant", "sheep", "couch", "train", "tv",
    # t2
    "truck", "traffic light", "fire hydrant", "stop sign", "parking meter",
    "bench", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase",
    "microwave", "oven", "toaster", "sink", "refrigerator", 
    # t3 
    "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", 
    # t4
    "bed", "toilet", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", 
    # Unknown
    "unknown",
] 

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

def box_iou(box_as, box_bs):
    box_asExtend=box_as.unsqueeze(1).expand(box_as.shape[0],box_bs.shape[0],4)
    box_bsExtend = box_bs.unsqueeze(0).expand(box_as.shape[0], box_bs.shape[0], 4)
    box1 = box_asExtend
    box2 = box_bsExtend
    leftTop = torch.max(box1[..., 0:2], box2[..., 0:2])
    bottomRight = torch.min(box1[..., 2:4], box2[..., 2:4])
    b1AndB2 = torch.clamp(bottomRight - leftTop, min=0)
    b1AndB2Area = b1AndB2[..., 0:1] * b1AndB2[..., 1:2]
    b1Area = (box1[...,2:3]-box1[...,0:1])*(box1[...,3:4]-box1[...,1:2])
    b2Area = (box2[...,2:3]-box2[...,0:1])*(box2[...,3:4]-box2[...,1:2])
    return b1AndB2Area / (b1Area + b2Area - b1AndB2Area)

def gen_pl(proposal_path, save_path, dataset_path, split, iou_thr, centerness_thr, num_keep, \
    score_keep, percent_keep, keep_type, known_cls_num, num_vis, vis_path, setting):
    
    if os.path.exists(vis_path):
        shutil.rmtree(vis_path)
        os.mkdir(vis_path)
    else:
        os.mkdir(vis_path)
    
    assert ((keep_type != "num") or (keep_type != "score") or (keep_type != "percent"))
    proposals = json.load(open(proposal_path))
    bboxes_by_id = {}
    scores_by_id = {}
    for image_id, p in tqdm(proposals.items(), ncols=80):
        bboxes = p["bboxes"]
        scores = p['scores']
        bboxes_by_id[image_id] = bboxes
        scores_by_id[image_id] = scores
    print("proposal file loaded")
    
    if setting == "owdetr":
        data_dict = load_voc_instances(dataset_path, split, ALL_CLS_NAMES_OWDETR)
    elif setting == "owod":
        data_dict = load_voc_instances(dataset_path, split, ALL_CLS_NAMES_OWOD)
        
    data_by_id = {}
    anno_cnt = 0 
    for d in tqdm(data_dict, ncols=80):
        annos = d.get('annotations', None)
        annos_ = []
        for anno in annos:
            category_id = anno['category_id']
            if category_id < known_cls_num:
                annos_.append(anno)
                anno_cnt = anno_cnt + 1
        if len(annos_) > 0:
            d['annotations'] = annos_
            image_id = d['image_id']
            data_by_id[image_id] = d
        
    print("dataset file loaded, total known annos:", anno_cnt) 
    
    for image_id, img_data in tqdm(data_by_id.items(), ncols=80):
        bboxes = bboxes_by_id.get(image_id, None)
        if bboxes == None:
            continue
        bboxes = torch.tensor(bboxes).reshape([-1, 4])
        scores = scores_by_id[image_id]
        scores = torch.tensor(scores)
        w = bboxes[:,2] - bboxes[:,0]
        h = bboxes[:,3] - bboxes[:,1]
        size = w * h
        # img_data = data_by_id[image_id]
        size_img = img_data['height'] * img_data['width']
        mask = size > size_img
        assert True not in mask
        mask1 = size <= 0.98 * size_img
        mask2 = w / h <= 4
        mask3 = h / w <= 4
        mask4 = size >= 2000
        mask = mask1 & mask2 & mask3 & mask4
        bboxes = bboxes[mask]
        scores = scores[mask]

        output = torchvision.ops.nms(boxes=bboxes.float(), scores=scores, iou_threshold=0.3)
        bboxes = bboxes[output]
        scores = scores[output]
        bboxes_by_id[image_id] = bboxes
        scores_by_id[image_id] = scores

    res_save = {}
    pl_cnt = 0
    score_record = torch.tensor([])
    score_list = []
    bboxes_proposal_list = []
    bboxes_anno_list = []
    image_id_list = []
    vis_cnt = 0
    image_ids = list(data_by_id.keys())
    random.shuffle(image_ids)
    for image_id in tqdm(image_ids, ncols=80):
        img_data = data_by_id.get(image_id, None)
        if img_data != None:
            annos = img_data.get('annotations', None)
        bboxes_proposal = bboxes_by_id.get(image_id, None)
        if bboxes_proposal == None:
            continue
        scores_res = scores_by_id[image_id]
        bboxes_anno = []
        if annos != None:
            for a in annos:
                category_id = a['category_id']
                if category_id < known_cls_num:
                    bbox = torch.tensor(a["bbox"]).reshape([-1,4])
                    bboxes_anno.append(bbox)
            if len(bboxes_anno) == 0:
                continue
            bboxes_anno = torch.cat(bboxes_anno, dim=0)
            
            iou = box_iou(bboxes_anno, bboxes_proposal).squeeze(dim = -1)
            iou_max, target_idx = iou.max(dim = 0)
            
            bboxes_target = bboxes_anno[target_idx]
            keep = iou_max <= iou_thr
            bboxes_proposal = bboxes_proposal[keep]
            scores_res = scores_res[keep]
            bboxes_target = bboxes_target[keep]

            bboxes_proposal_x = ((bboxes_proposal[:, 0] + bboxes_proposal[:, 2])/2).reshape([-1,1])
            bboxes_proposal_y = ((bboxes_proposal[:, 1] + bboxes_proposal[:, 3])/2).reshape([-1,1])
            bboxes_proposal_center = torch.cat([bboxes_proposal_x, bboxes_proposal_y], dim = 1)
            centerness = get_centerness(bboxes_proposal_center, bboxes_target)
            keep = centerness <= centerness_thr
            bboxes_proposal = bboxes_proposal[keep]
            scores_res = scores_res[keep]

        if keep_type == "num":
            if bboxes_proposal.shape[0] > num_keep:
                bboxes_proposal = bboxes_proposal[:num_keep]
                scores_res = scores_res[:num_keep]
        elif keep_type == "score":
            keep = scores_res > score_keep
            bboxes_proposal = bboxes_proposal[keep]
            scores_res = scores_res[keep]
        elif keep_type == "percent":
            scores_res = scores_res[:10]
            bboxes_proposal = bboxes_proposal[:10]
            score_record = torch.cat([score_record, scores_res])
            score_list.append(scores_res)
            bboxes_proposal_list.append(bboxes_proposal)
            bboxes_anno_list.append(bboxes_anno)
            image_id_list.append(image_id)

        if keep_type == "num" or keep_type == "score":
            pl_cnt = pl_cnt + bboxes_proposal.shape[0]
            res_save[image_id] = {"bboxes": bboxes_proposal.cpu().numpy().tolist()}
            if vis_cnt <= num_vis:
                image_id = image_id + ".jpg"
                img = cv2.imread(os.path.join(dataset_path, "JPEGImages", image_id))
                img = img[:, :, [2,1,0]]
                img_save_path = os.path.join(vis_path, image_id)
                vis_res(img, bboxes_anno, bboxes_proposal, img_save_path)
                vis_cnt = vis_cnt + 1

    if keep_type == "percent":
        score_record = score_record.cpu().numpy()
        num_total = score_record.shape[0]
        num_keep = int(num_total * percent_keep)
        score_record = np.sort(score_record)
        score_thr = score_record[-num_keep]
        vis_cnt = 0
        for image_id, bboxes_proposal, score_res, bboxes_anno in zip(image_id_list, bboxes_proposal_list, score_list, bboxes_anno_list):
            keep = score_res > score_thr
            bboxes_proposal = bboxes_proposal[keep]
            pl_cnt = pl_cnt + bboxes_proposal.shape[0]
            res_save[image_id] = {"bboxes": bboxes_proposal.cpu().numpy().tolist()}

            if vis_cnt <= num_vis:
                image_id = image_id + ".jpg"
                img = cv2.imread(os.path.join(dataset_path, "JPEGImages", image_id))
                img = img[:, :, [2,1,0]]
                img_save_path = os.path.join(vis_path, image_id)
                vis_res(img, bboxes_anno, bboxes_proposal, img_save_path)
                vis_cnt = vis_cnt + 1
    
    print("visualizations of pseudo labels are shown in", vis_path)
    json.dump(res_save, open(save_path, 'w'))
    print("total pseudo labels:", pl_cnt)
    print("save pseudo labels to", save_path)
    

def parse_args(in_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposal_path", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--iou_thr", type=float, default=0.3)
    parser.add_argument("--centerness_thr", type=float, default=1.0)
    parser.add_argument("--num_keep", type=int, default=20)
    parser.add_argument("--score_keep", type=float, default=0.3)
    parser.add_argument("--percent_keep", type=float, default=0.3)
    parser.add_argument("--keep_type", type=str)
    parser.add_argument("--known_cls_num", type=int)
    parser.add_argument("--num_vis", type=int, default=50)
    parser.add_argument("--vis_path", type=str, default="./img_vis")
    parser.add_argument("--data_split", type=str, default="all_task_train")
    parser.add_argument("--setting", type=str, default="owdetr")
    
    return parser.parse_args(in_args)

if __name__ == "__main__":
    args = parse_args()
    proposal_path = args.proposal_path
    data_path = args.data_path
    save_path = args.save_path
    iou_thr = args.iou_thr
    centerness_thr = args.centerness_thr
    num_keep = args.num_keep
    score_keep = args.score_keep
    percent_keep = args.percent_keep
    keep_type = args.keep_type
    known_cls_num = args.known_cls_num
    num_vis = args.num_vis
    vis_path = args.vis_path
    data_split = args.data_split
    setting = args.setting
    print(args)
    gen_pl(proposal_path, save_path, data_path, data_split, iou_thr, centerness_thr, num_keep, score_keep, \
        percent_keep, keep_type, known_cls_num, num_vis, vis_path, setting)





