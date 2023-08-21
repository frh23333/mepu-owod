import os
from .voc_coco import register_sowod, register_sowod_with_extrabboxes, register_mowod, register_mowod_with_pl
from .coco import register_coco_instances
from detectron2.data import MetadataCatalog

def register_all_sowod(root):
    SPLITS = [
        ("sowod_val", "sowod", "instances_val2017"),
        ("sowod_train", "sowod", "instances_train2017"),
        ("sowod_train_t1", "sowod", "t1_train"),
        ("sowod_train_t2", "sowod", "t2_train"),
        ("sowod_train_t3", "sowod", "t3_train"),
        ("sowod_train_t4", "sowod", "t4_train"),
        ("sowod_oracle_t1", "sowod", "t1_train"),
        ("sowod_oracle_t2", "sowod", "t2_train"),
        ("sowod_oracle_t3", "sowod", "t3_train"),
        ("sowod_oracle_t4", "sowod", "t4_train"),
    ]
    for name, dirname, split in SPLITS:
        year = 2007 if "2007" in name else 2012
        register_sowod(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"

def register_all_sowod_with_pseudo_label(root):
    SPLITS = [
        ("sowod_train_t1_fs", "sowod", "t1_train", "fs"),
        ("sowod_train_t1_ss", "sowod", "t1_train", "ss"),
        ("sowod_train_t1_eb", "sowod", "t1_train", "eb"),
        ("sowod_train_t1_gop", "sowod", "t1_train", "gop"),
        ("sowod_train_t1_detreg", "sowod", "t1_train", "detreg"),
        
        ("sowod_train_t1_st", "sowod", "t1_train", "st"),
        ("sowod_train_t2_st", "sowod", "t2_train", "st"),
        ("sowod_train_t3_st", "sowod", "t3_train", "st"),
        
        ("sowod_t2_ft", "sowod", "t2_ft", "none"),
        ("sowod_t2_train_and_ft", "sowod", "t2_train_and_ft", "none"),
        ("sowod_t3_ft", "sowod", "t3_ft", "none"),
        ("sowod_t3_train_and_ft", "sowod", "t3_train_and_ft", "none"),
        ("sowod_t4_ft", "sowod", "t4_ft", "none"),

        ("sowod_t2_ft_st", "sowod", "t2_ft", "st"),
        ("sowod_t3_ft_st", "sowod", "t3_ft", "st"),
    ]
    for name, dirname, split, box_type in SPLITS:
        year = 2007 if "2007" in name else 2012
        register_sowod_with_extrabboxes(
            name, os.path.join(root, dirname), split, year, box_type)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"
        
def register_all_mowod(root):
    SPLITS = [
        ("mowod_train", "mowod", "all_task_train"),
        ("mowod_val", "mowod", "all_task_test"),
        ("mowod_train_t1", "mowod", "t1_train"),
        ("mowod_train_t2", "mowod", "t2_train"),
        ("mowod_train_t3", "mowod", "t3_train"),
        ("mowod_train_t4", "mowod", "t4_train"),
        ("mowod_oracle_t1", "mowod", "t1_train"),
        ("mowod_oracle_t2", "mowod", "t2_train"),
        ("mowod_oracle_t3", "mowod", "t3_train"),
        ("mowod_oracle_t4", "mowod", "t4_train"),
    ]
    for name, dirname, split in SPLITS:
        year = 2007 if "2007" in name else 2012
        register_mowod(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"

def register_all_mowod_with_pseudo_label(root):
    SPLITS = [
        ("mowod_train_t1_fs", "mowod", "t1_train", "fs"),
        ("mowod_train_t1_ss", "mowod", "t1_train", "ss"),
        
        ("mowod_train_t1_st", "mowod", "t1_train", "st"),
        ("mowod_train_t2_st", "mowod", "t2_train", "st"),
        ("mowod_train_t3_st", "mowod", "t3_train", "st"),
        
        ("mowod_t2_ft", "mowod", "t2_ft", "none"),
        ("mowod_t2_train_and_ft", "mowod", "t2_train_and_ft", "none"),
        ("mowod_t3_ft", "mowod", "t3_ft", "none"),
        ("mowod_t3_train_and_ft", "mowod", "t3_train_and_ft", "none"),
        ("mowod_t4_ft", "mowod", "t4_ft", "none"),

        ("mowod_t2_ft_st", "mowod", "t2_ft", "st"),
        ("mowod_t3_ft_st", "mowod", "t3_ft", "st"),
    ]
    for name, dirname, split, box_type in SPLITS:
        year = 2007 if "2007" in name else 2012
        register_mowod_with_pl(
            name, os.path.join(root, dirname), split, year, box_type)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"

if __name__.endswith(".builtin"):
    # Register them all under "./datasets"
    
    _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    
    register_all_sowod(_root)
    register_all_mowod(_root)
    register_all_sowod_with_pseudo_label(_root)
    register_all_mowod_with_pseudo_label(_root)
    register_coco_instances("coco_st_t1", 
        {}, 
        os.path.join(_root, "coco/annotations/coco_st_t1.json"), 
        os.path.join(_root, "coco/train2017/"))
    register_coco_instances("coco_st_t2", 
        {}, 
        os.path.join(_root, "coco/annotations/coco_st_t2.json"), 
        os.path.join(_root, "coco/train2017/"))
    register_coco_instances("coco_st_t3", 
        {}, 
        os.path.join(_root, "coco/annotations/coco_st_t3.json"), 
        os.path.join(_root, "coco/train2017/"))
    register_coco_instances("objects365_train", 
        {}, 
        os.path.join(_root, "objects365/annotations/train.json"), 
        os.path.join(_root, "objects365/train"))