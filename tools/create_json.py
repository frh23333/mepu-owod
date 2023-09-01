import tqdm
import json

ALL_CLS_NAMES = ["airplane","bicycle","bird","boat","bus","car",
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

if __name__ == "__main__":
    json_path = "datasets/coco/annotations/instances_train2017.json"
    save_path = ["datasets/coco/annotations/coco_st_t1.json", 
                 "datasets/coco/annotations/coco_st_t2.json", 
                 "datasets/coco/annotations/coco_st_t3.json"]
    anno_dict = json.load(open(json_path))
    anno_list = anno_dict['annotations']
    img_list = anno_dict['images']
    anno_dict['categories'].append({'supercategory': 'unknown', 'id': 91, 'name': 'unknown'})
    num_known_class_per_task = [19, 40, 60]
    num_prev_known_class_per_task = [0, 19, 40]
    anno_cnt = 0

    # print(anno_dict['categories'])
    print("total annotations", len(anno_list))
    for i in range(3):
        num_known_class = num_known_class_per_task[i]
        num_prev_known_class = num_prev_known_class_per_task[i]
        img_id = {}
        cid_to_cname = {}
        for c in anno_dict["categories"]:
            cid_to_cname[c["id"]] = c['name']
        anno_list_ = []
        for a in tqdm.tqdm(anno_list, ncols=80):
            cname = cid_to_cname[a["category_id"]]
            if cname not in ALL_CLS_NAMES[num_prev_known_class:num_known_class]:
                continue
            else:
                anno_list_.append(a)
                img_id[a["image_id"]] = True
        print("num annotations on Task", i+1, len(anno_list_))
        img_list_ = []
        for img in img_list:
            if img_id.get(img['id']) == True:
                img_list_.append(img)
        print("num images on Task", i+1, len(img_list_))
        anno_dict['annotations'] = anno_list_
        anno_dict['images'] = img_list_
        print("saving json file for Task", i+1)
        json.dump(anno_dict, open(save_path[i], 'w'))