# t1 known mAP
# python train_net.py --eval-only --resume --dist-url auto --num-gpus 4 --config config/MEPU-SOWOD/t1/self-train.yaml \
# 	OUTPUT_DIR training_dir/mepu-fs-t1-self-train \

# t1 unknown R@100
python train_net.py --eval-only --resume --dist-url auto --num-gpus 4 --config config/MEPU-SOWOD/t1/train.yaml \
	OUTPUT_DIR training_dir/mepu-sowod/ss-t1-self-train OPENSET.EVAL_UNKNOWN True TEST.DETECTIONS_PER_IMAGE 100 \

python train_net.py --eval-only --resume --dist-url auto --num-gpus 4 --config config/MEPU-SOWOD/t1/train.yaml \
	OUTPUT_DIR training_dir/mepu-sowod/ss-t1-self-train OPENSET.EVAL_UNKNOWN True TEST.DETECTIONS_PER_IMAGE 10 \

python train_net.py --eval-only --resume --dist-url auto --num-gpus 4 --config config/MEPU-SOWOD/t1/train.yaml \
	OUTPUT_DIR training_dir/mepu-sowod/ss-t1-self-train OPENSET.EVAL_UNKNOWN True TEST.DETECTIONS_PER_IMAGE 30 \
	
# python train_net.py --eval-only --resume --dist-url auto --num-gpus 4 --config config/MEPU-SOWOD/t1/self-train.yaml \
# 	OUTPUT_DIR training_dir/mepu-sowod/ss-t1-self-train OPENSET.EVAL_UNKNOWN True TEST.DETECTIONS_PER_IMAGE 100 \

# python train_net.py --eval-only --resume --dist-url auto --num-gpus 4 --config config/MEPU-SOWOD/t1/self-train.yaml \
# 	OUTPUT_DIR training_dir/mepu-sowod/ss-t1-self-train OPENSET.EVAL_UNKNOWN True TEST.DETECTIONS_PER_IMAGE 30 \

# python train_net.py --eval-only --resume --dist-url auto --num-gpus 4 --config config/MEPU-SOWOD/t1/self-train.yaml \
# 	OUTPUT_DIR training_dir/mepu-sowod/ss-t1-self-train OPENSET.EVAL_UNKNOWN True TEST.DETECTIONS_PER_IMAGE 10 \

# # t2 known mAP
# python train_net.py --eval-only --resume --dist-url auto --num-gpus 4 --config config/MEPU-SOWOD/t2/ft.yaml \
# 	OUTPUT_DIR training_dir/mepu-fs-t2-ft \

# # t2 unknown R@100
# python train_net.py --eval-only --resume --dist-url auto --num-gpus 4 --config config/MEPU-SOWOD/t2/ft.yaml \
# 	OUTPUT_DIR training_dir/mepu-fs-t2-ft OPENSET.EVAL_UNKNOWN True TEST.DETECTIONS_PER_IMAGE 100 \

# # t3 known mAP
# python train_net.py --eval-only --resume --dist-url auto --num-gpus 4 --config config/MEPU-SOWOD/t3/ft.yaml \
# 	OUTPUT_DIR training_dir/mepu-fs-t3-ft \

# # t3 unknown R@100
# python train_net.py --eval-only --resume --dist-url auto --num-gpus 4 --config config/MEPU-SOWOD/t3/ft.yaml \
# 	OUTPUT_DIR training_dir/mepu-fs-t3-ft OPENSET.EVAL_UNKNOWN True TEST.DETECTIONS_PER_IMAGE 100 \

# # t4 known mAP
# python train_net.py --eval-only --resume --dist-url auto --num-gpus 4 --config config/MEPU-SOWOD/t4/ft.yaml \
# 	OUTPUT_DIR training_dir/mepu-fs-t4-ft \

