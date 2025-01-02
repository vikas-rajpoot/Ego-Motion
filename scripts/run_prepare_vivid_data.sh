# for VIVID rgb-thermal dataset
DATASET=/home/vk/02/ThermalDepth/data
TRAIN_SET=/home/vk/03/ThermalSfMLearner/ProcessedData/
mkdir -p $TRAIN_SET
python common/data_prepare/prepare_train_data_VIVID.py $DATASET --dump-root $TRAIN_SET --width 320  --height 256 --num-threads 16 --with-depth  --with-pose
