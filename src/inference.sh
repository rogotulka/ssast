#!/bin/bash


checkpoint='/home/korovski_y_nfs/ss_ast/ssast/exp/mask01-tiny-f128-t2-b120-lr5e-4-m10-pretrain_joint-speechcommands/models/best_audio_model.pth'
label_dim=35
input_tdim=128
input_fdim=128

model_size=tiny
fshape=128
tshape=2
fstride=128
tstride=2

pretrain_stage='False'

python inference_ssast.py --checkpoint $checkpoint \
--label_dim $label_dim --input_tdim $input_tdim --fshape $fshape \
--tshape $tshape --fstride $fstride --tstride $tstride --input_fdim $input_fdim \
 --model_size $model_size --pretrain_stage $pretrain_stage