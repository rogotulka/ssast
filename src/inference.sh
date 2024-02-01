#!/bin/bash


checkpoint='./exp/test01-amd_no_other_with_rnd_end-f128-t1-b128-lr2.5e-4-ft_avgtok-base--SSAST-Base-Frame-400-1x-noiseTrue/models/best_audio_model.pth'
label_dim=4
input_tdim=256
input_fdim=128

model_size=tiny
fshape=128
tshape=2
fstride=128
tstride=1

pretrain_stage='False'

# base config for advance mlp head
# advance_head=true
# head_type='MGCrP'
# fusion_type='sum_fc'
# num_heads=6
# wr_dim=14
# normalization_type='svPN'
# normalization_alpha=0.5
# normalization_iterNum=1
# normalization_svNum=1
# normalization_regular=0
# normalization_input_dim=14
# normalization_qkv_dim=14

CUDA_VISIBLE_DEVICES=1 python src/inference_ssast.py --checkpoint $checkpoint \
--label_dim $label_dim --input_tdim $input_tdim --fshape $fshape \
--tshape $tshape --fstride $fstride --tstride $tstride --input_fdim $input_fdim \
 --model_size $model_size --pretrain_stage $pretrain_stage \
#   --advance_head $advance_head \
#   --head_type $head_type \
#   --fusion_type $fusion_type \
#   --num_heads $num_heads \
#   --wr_dim $wr_dim \
#   --normalization_type $normalization_type \
#   --normalization_alpha $normalization_alpha \
#   --normalization_iterNum $normalization_iterNum \
#   --normalization_regular $normalization_regular \
#   --normalization_input_dim $normalization_input_dim \
#   --normalization_qkv_dim $normalization_qkv_dim