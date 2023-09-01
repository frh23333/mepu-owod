# t1 known mAP, U-Recall, WI and A-OSE
python train_net.py --eval-only --resume --dist-url auto --num-gpus 4 --config config/MEPU-SOWOD/t1/train.yaml \
	OUTPUT_DIR training_dir/mepu-sowod/fs-t1-self-train \

# t1 unknown R@100
python train_net.py --eval-only --resume --dist-url auto --num-gpus 4 --config config/MEPU-SOWOD/t1/train.yaml \
	OUTPUT_DIR training_dir/mepu-sowod/fs-t1-self-train OPENSET.EVAL_UNKNOWN True TEST.DETECTIONS_PER_IMAGE 100 \
	
# t2 known mAP, U-Recall, WI and A-OSE
python train_net.py --eval-only --resume --dist-url auto --num-gpus 4 --config config/MEPU-SOWOD/t2/train.yaml \
	OUTPUT_DIR training_dir/mepu-sowod/fs-t2-ft \

# t2 unknown R@100
python train_net.py --eval-only --resume --dist-url auto --num-gpus 4 --config config/MEPU-SOWOD/t2/train.yaml \
	OUTPUT_DIR training_dir/mepu-sowod/fs-t2-ft OPENSET.EVAL_UNKNOWN True TEST.DETECTIONS_PER_IMAGE 100 \

# t3 known mAP, U-Recall, WI and A-OSE
python train_net.py --eval-only --resume --dist-url auto --num-gpus 4 --config config/MEPU-SOWOD/t3/train.yaml \
	OUTPUT_DIR training_dir/mepu-sowod/fs-t2-ft \

# t3 unknown R@100
python train_net.py --eval-only --resume --dist-url auto --num-gpus 4 --config config/MEPU-SOWOD/t3/train.yaml \
	OUTPUT_DIR training_dir/mepu-sowod/fs-t2-ft OPENSET.EVAL_UNKNOWN True TEST.DETECTIONS_PER_IMAGE 100 \
	

