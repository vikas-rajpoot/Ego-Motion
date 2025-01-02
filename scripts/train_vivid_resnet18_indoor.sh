# DATA_ROOT=/home/vk/03/ThermalSfMLearner/ProcessedData/
TRAIN_SET=/home/vk/03/ThermalSfMLearner/ProcessedData/
GPU_NUM=1

CUDA_VISIBLE_DEVICES=${GPU_NUM} \
python train.py $TRAIN_SET \
--resnet-layers 18 \
--num-scales 1 \
--scene_type indoor \
-b 4 -s 0 -t 0.25 -r 1.0 \
--rgb-ssim 0.85 --thr-ssim 0.15 \
--epoch-size 400 --sequence-length 3 \
--with-ssim 1 \
--with-thr-mask 1 \
--with-rgb-mask 1 \
--with-auto-mask 1 \
--with-pretrain 1 \
--log-output --with-gt \
--name vivid_resnet18_indoor
