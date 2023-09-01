# unsupervised pretraining of REW using COCO train2017
# pretrained SoCo backbone can be downloaded in https://github.com/hologerry/SoCo
python train_net.py --resume --dist-url auto --num-gpus 4 --config config/REW/rew_pretrain.yaml \
	OUTPUT_DIR training_dir/rew \
    
    