DATA_DIR=datasets/s-owod
COCO_DIR=datasets/coco

# # make neccesary dirs
rm $DATA_DIR -rf
echo "make dirs"
mkdir -p $DATA_DIR
mkdir -p $DATA_DIR/Annotations
# mkdir -p $DATA_DIR/JPEGImages
mkdir -p $DATA_DIR/ImageSets
mkdir -p $DATA_DIR/ImageSets/Main

# cp data
# make use you have $COCO_DIR
echo "copy coco images"
cp -r $COCO_DIR/train2017 $DATA_DIR/JPEGImages 
cp $COCO_DIR/val2017/* $DATA_DIR/JPEGImages/

echo "convert coco annotation to voc"
python tools/convert_coco_to_voc.py --dir $DATA_DIR --ann_path $COCO_DIR/annotations/instances_train2017.json
python tools/convert_coco_to_voc.py --dir $DATA_DIR --ann_path $COCO_DIR/annotations/instances_val2017.json

echo "copy owod spilit files"
cp ./dataset_splits/s-owod/ImageSets/Main/* $DATA_DIR/ImageSets/Main/





