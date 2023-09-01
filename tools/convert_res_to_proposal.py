import json
import torch
from tqdm import tqdm
import os
import argparse
import sys

def gen_pl(proposal_path, save_path):
        
    proposal_list = json.load(open(proposal_path))
    bbox_by_name = {}
    score_by_name = {}
    proposals_by_name = {}
    for image_id, p in tqdm(proposal_list.items(), ncols=80):
        # image_id = str(p["image_id"]) 
        # bbox = p["bbox"]
        # score = p['score']
        bbox_by_name[image_id] = p['bboxes']
        score_by_name[image_id] = p['scores']
        # if bbox_by_name.get(image_id, None) != None:
        #     bbox_by_name[image_id].append(bbox)
        #     score_by_name[image_id].append(score)
        # else:
        #     bbox_by_name[image_id] = [bbox]
        #     score_by_name[image_id] = [score]
    print("proposal file loaded")
    pro_cnt = 0
    for img_id, bboxes in tqdm(bbox_by_name.items(), ncols=80):
        bboxes = torch.tensor(bboxes).reshape([-1, 4])
        w = bboxes[:,2] - bboxes[:,0]
        h = bboxes[:,3] - bboxes[:,1]
        
        size = w * h
        mask1 = w / h <= 4
        mask2 = h / w <= 4
        mask3 = size >= 2000
        mask = mask1 & mask2 & mask3
        bboxes = bboxes[mask]
        

        # bboxes[:,2] = bboxes[:,0] + bboxes[:,2]
        # bboxes[:,3] = bboxes[:,1] + bboxes[:,3]
        bboxes = bboxes.cpu().tolist()
        scores = torch.tensor(score_by_name[img_id])[mask].cpu().tolist()
        # scores = ((scores - scores.mean()) / (scores.std()))
        # scores = (scores * 0.5 + (torch.randn(scores.shape))).sigmoid().cpu().tolist()
        pro_cnt = pro_cnt + len(bboxes)
        
        proposals = {"bboxes": bboxes, "scores": scores}
        
        proposals_by_name[img_id] = proposals
    print(pro_cnt)
    json.dump(proposals_by_name, open(save_path, "w"))
    print("proposal file saved to ", save_path)
        
def parse_args(in_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposal_path", type=str)
    parser.add_argument("--save_path", type=str)
    return parser.parse_args(in_args)

if __name__ == "__main__":
    args = parse_args()
    proposal_path = args.proposal_path
    save_path = args.save_path
    print(args)
    gen_pl(proposal_path, save_path)
    