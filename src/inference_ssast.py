import os, csv, argparse, wget
import torch, torchaudio, timm
import numpy as np
from torch.cuda.amp import autocast
from models.ast_models import ASTModel


def inference(args):
    pretrain_stage = True if args.pretrain_stage == 'True' else False
    ast_mdl = ASTModel(label_dim=args.label_dim,
                       input_tdim=args.input_tdim,
                       fshape=args.fshape,
                       tshape=args.tshape,
                       fstride=args.fstride,
                       tstride=args.tstride,
                       input_fdim=args.input_fdim,
                       model_size=args.model_size,
                       pretrain_stage=pretrain_stage,
                       load_pretrained_mdl_path=args.checkpoint)
    print(f'[*INFO] loading checkpoint: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    audio_model = torch.nn.DataParallel(ast_mdl, device_ids=[0])
    audio_model.load_state_dict(checkpoint, strict=False)
    audio_model = audio_model.to(torch.device("cuda:0"))
    audio_model.eval()


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', help='Path to checkpoint')
parser.add_argument('--label_dim', default=35, type=int, help='Label dimension')
parser.add_argument('--input_tdim', type=int, help='')
parser.add_argument('--fshape', type=int, help='')
parser.add_argument('--tshape', type=int, help='')
parser.add_argument('--fstride', type=int, help='')
parser.add_argument('--tstride', type=int, help='')
parser.add_argument('--input_fdim', type=int, help='')
parser.add_argument('--model_size', type=str, help='')
parser.add_argument('--pretrain_stage', type=bool, help='True or False')


args = parser.parse_args()
inference(args)


