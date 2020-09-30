#!/bin/bash

flaginfo="double_state_attention_just_test_th30_googlenet_bn_sample30_msdn_obj_top10"
flag="-flag "$flaginfo
data_dir=features/
datainput=" -input_h5 ${data_dir}/msvd_all_sample30_frame_googlenet_bn_pool5.h5 "
labelinput=" -label_file data/labels_th30.npy -input_json data/info_th30.json "
#topic=" -input_h5_local ${data_dir}/msvd_all_sample30_frame_googlenet_bn_in5b.h5 "
#topic=" -input_h5_local ${data_dir}/msvd_all_sample30_frame_msdn_relation_scoreweight_top50.h5 "
topic=" -input_h5_local ${data_dir}/msvd_all_sample30_frame_msdn_obj_top10.h5 "
#topic=" -input_h5_local ${data_dir}/msvd_all_sample30_frame_msdn_region_top5.h5 "

params=" -gpuid 2 -learningRate 2e-4 -dropout 0.5 -grad_clip 10 -learning_rate_decay_every 20 -add_supervision 0 "
size=" -batchsize 64 -hiddensize 512 -eval_every 100 "
cmd=" th train_dual_memory_model.lua -checkpoint_dir checkpoints_th30_googlenet_bn_sample30_msdn_obj_top10 -max_epochs 80 "

a=$(date +%y%m%d-%H%M)

log="logs/"$a$flaginfo".log"

$cmd $datainput $topic $labelinput $params $size $flag 2>&1 | tee $log
echo "$cmd $datainput $topic $labelinput $params $size $flag" >> $log
wait



