#!/bin/bash
# run script : bash test_vivid_indoor.sh

NAME=vivid_resnet18_indoor
GPU_NUM=1
DATA_ROOT=/home/vk/03/ThermalSfMLearner/ProcessedData
RESULTS_DIR=results

RESNET=18
IMG_H=256
IMG_W=320
POSE_NET=checkpoints/${NAME}/exp_pose_pose_model_best.pth.tar
DISP_NET=checkpoints/${NAME}/dispnet_disp_model_best.pth.tar
DATASET=VIVID
INPUT_TYPE=T 
DEPTH_GT_DIR=Depth_T 
POSE_GT=poses_T.txt 

# indoor testset
SEQS=("indoor_robust_dark"	"indoor_robust_varying" "indoor_aggresive_dark"	"indoor_aggresive_local" "indoor_unstable_dark" "indoor_robust_varying_well_lit")

for SEQ in ${SEQS[@]}; do
	echo "Seq_name : ${SEQ}"
	SCENE=indoor

	#mkdir -p ${RESULTS_DIR}
	DATA_DIR=${DATA_ROOT}/${SEQ}/
	OUTPUT_DEPTH_DIR=${RESULTS_DIR}/${NAME}/Depth/${SEQ}
	mkdir -p ${OUTPUT_DEPTH_DIR}

	# Detph Evaulation 
	CUDA_VISIBLE_DEVICES=${GPU_NUM} python test_disp.py --resnet-layers $RESNET --pretrained-dispnet $DISP_NET \
	--img-height $IMG_H --img-width $IMG_W --input ${INPUT_TYPE} --scene_type ${SCENE} \
	--dataset-dir ${DATA_DIR} --output-dir $OUTPUT_DEPTH_DIR >> ${OUTPUT_DEPTH_DIR}/disp.txt

	CUDA_VISIBLE_DEVICES=${GPU_NUM} python eval_vivid/eval_depth.py --dataset $DATASET --input $INPUT_TYPE --scene ${SCENE} \
	--pred_depth ${OUTPUT_DEPTH_DIR}/predictions.npy --gt_depth ${DATA_DIR}/${DEPTH_GT_DIR}/ \
	--img_dir ${DATA_DIR} --vis_dir ${OUTPUT_DEPTH_DIR} --img_dir ${DATA_DIR} >> ${OUTPUT_DEPTH_DIR}/eval_depth.txt

done 

SEQS=("indoor_robust_dark" "indoor_robust_varying" "indoor_aggresive_dark"	"indoor_aggresive_local" "indoor_unstable_dark")

for SEQ in ${SEQS[@]}; do
	echo "Seq_name : ${SEQ}"
	SCENE=indoor

	DATA_DIR=${DATA_ROOT}/${SEQ}/
	OUTPUT_POSE_DIR=${RESULTS_DIR}/${NAME}/POSE/${SEQ}/
	mkdir -p ${OUTPUT_POSE_DIR}

	# Pose Evaulation 
	CUDA_VISIBLE_DEVICES=${GPU_NUM} python test_pose.py --resnet-layers $RESNET --pretrained-posenet $POSE_NET \
	--img-height $IMG_H --img-width $IMG_W --scene_type ${SCENE} \
	--dataset-dir ${DATA_ROOT} --output-dir ${OUTPUT_POSE_DIR} \
	--input ${INPUT_TYPE} --sequences ${SEQ} >> ${OUTPUT_POSE_DIR}/eval_pose.txt
done




