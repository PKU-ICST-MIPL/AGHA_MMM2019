#!/bin/bash

flaginfo="double_state_attention_just_test"
flag="-flag "$flaginfo
data_dir=features/
datainput=" -label_file data/labels_th30.npy -input_h5 ${data_dir}/msvd_all_sample30_frame_googlenet_bn_pool5.h5 -input_json data/info_th30.json "
topic=" -input_h5_local ${data_dir}/msvd_all_sample30_frame_msdn_obj_top10.h5 "

params=" -gpuid 0 -learningRate 2e-4 -dropout 0.5 -grad_clip 10 -learning_rate_decay_every 20 -add_supervision 0"
size=" -batchsize 64 -hiddensize 512 -eval_every 100 "
model_dir=checkpoints_th30_googlenet_bn_sample30_msdn_obj_top10
cmd=" th test_model_single.lua -init_from ${model_dir}/180403-2322_double_state_attention_just_test_th30_googlenet_bn_sample30_msdn_obj_top10_26.t7"

a=$(date +%y%m%d-%H%M)

log="logs/"$a$flaginfo".log"

$cmd $datainput $topic $params $size $flag

wait



