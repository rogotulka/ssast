#!/bin/bash


checkpoint='/home/korovski_y_nfs/ss_ast/ssast/exp/test01-speechcommands-f128-t2-b128-lr2.5e-4-ft_avgtok-base-1x-noiseTrue/models/best_audio_model.pth'
label_dim=35
input_tdim=128
input_fdim=128

model_size=base
fshape=128
tshape=2
fstride=128
tstride=2

pretrain_stage='False'
task='ft_avgtok'

python inference_ssast.py --checkpoint $checkpoint \
--label_dim $label_dim --input_tdim $input_tdim --fshape $fshape \
--tshape $tshape --fstride $fstride --tstride $tstride --input_fdim $input_fdim \
 --model_size $model_size --pretrain_stage $pretrain_stage --task $task