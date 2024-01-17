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

CUDA_VISIBLE_DEVICES=1 python src/inference_ssast.py --checkpoint $checkpoint \
--label_dim $label_dim --input_tdim $input_tdim --fshape $fshape \
--tshape $tshape --fstride $fstride --tstride $tstride --input_fdim $input_fdim \
 --model_size $model_size --pretrain_stage $pretrain_stage