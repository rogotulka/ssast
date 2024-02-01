import os, csv, argparse, wget
import torch, torchaudio, timm
import numpy as np
from torch.cuda.amp import autocast
from models.ast_models import ASTModel
import dataloader

def inference(args):
    pretrain_stage = True if args.pretrain_stage == 'True' else False
    checkpoint_path = args.checkpoint
    
    if args.advance_head:
        advance_head_conf = dict(
            type=args.head_type,
            fusion_type=args.fusion_type,
            args=dict(
                num_heads=args.num_heads,
                wr_dim=args.wr_dim,
                normalization=dict(
                    type=args.normalization_type,
                    alpha=args.normalization_alpha,
                    iterNum=args.normalization_iterNum,
                    svNum=args.normalization_svNum,
                    regular=None if args.normalization_regular == 0 else torch.nn.Dropout(args.normalization_regular),
                    input_dim=args.normalization_input_dim,
                    qkv_dim=args.normalization_qkv_dim
                ),
            ),
        )
    else:
        advance_head_conf = None
    
    ast_mdl = ASTModel(label_dim=args.label_dim,
                       input_tdim=args.input_tdim,
                       fshape=args.fshape,
                       tshape=args.tshape,
                       fstride=args.fstride,
                       tstride=args.tstride,
                       input_fdim=args.input_fdim,
                       model_size=args.model_size,
                       pretrain_stage=pretrain_stage,
                       load_pretrained_mdl_path=args.checkpoint,
                       representationConfig=advance_head_conf)
    print(f'[*INFO] loading checkpoint: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    audio_model = torch.nn.DataParallel(ast_mdl, device_ids=[0])
    audio_model.load_state_dict(checkpoint, strict=False)
    audio_model = audio_model.to(torch.device("cuda:0"))
    audio_model.eval()

    dataset_mean=-6.845978
    dataset_std=5.5654526
    val_audio_conf = {'num_mel_bins': 128, 'target_length': 256, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': 'speechcommands',
                    'mode': 'evaluation', 'mean': dataset_mean, 'std': dataset_std, 'noise': False}

    eval_loader = torch.utils.data.DataLoader(
        dataloader.AudioDataset('./data/amd/amd_data_test.json', label_csv='./data/amd_class_labels_indices.csv',
                                audio_conf=val_audio_conf),
        batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    
    test_input = torch.zeros([10, args.input_tdim, 128])
    # print(next(iter(eval_loader)))
    # print(next(iter(eval_loader)))
    for X, y, file_path in eval_loader:
        # X, y = next(iter(eval_loader))
        print(file_path)
        # print(y.shape)
        print('true', np.argmax(y, axis=1))
        
        prediction = audio_model(X, task='ft_avgtok')
        print(torch.argmax(torch.nn.functional.softmax(prediction, dim=-1), dim=-1))
        print(torch.nn.functional.softmax(prediction, dim=-1))


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

parser.add_argument("--advance_head", type=bool, default=False, help="if use fuse mlp head")
parser.add_argument("--head_type", type=str, default='MGCrP', choices=["MGCrP", "GAP", "GCP", "fc"], help="type of mlp head architecture")
parser.add_argument("--fusion_type", type=str, default='sum_fc', choices=["sum_fc", "concat", "aggre_all"], help="type of visual and cls tokens fusion")
parser.add_argument("--num_heads", type=int, default=6, help="number of sub heads for GCP and MGCrP")
parser.add_argument("--wr_dim", type=int, default=14, help="inner dim for MGCrP")
parser.add_argument("--normalization_type", type=str, default='svPN', help="type of normalization for MGCrP")
parser.add_argument("--normalization_alpha", type=float, default=0.5, help="alpha for MGCrP with svPN")
parser.add_argument("--normalization_iterNum", type=int, default=1, help="iterNum for MGCrP with svPN")
parser.add_argument("--normalization_svNum", type=int, default=1, help="svNum for MGCrP with svPN")
parser.add_argument("--normalization_regular", type=str, default='svPN', help="dropout for MGCrP with svPN")
parser.add_argument("--normalization_input_dim", type=int, default=14, help="some inner dim for MGCrP with svPN")
parser.add_argument("--normalization_qkv_dim", type=int, default=14, help="some inner dim for GCP")

args = parser.parse_args()
inference(args)