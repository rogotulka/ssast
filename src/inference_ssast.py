import os, csv, argparse, wget
import torch, torchaudio, timm
import numpy as np
from torch.cuda.amp import autocast
from models.ast_models import ASTModel
import dataloader

def inference(args):
    pretrain_stage = True if args.pretrain_stage == 'True' else False
    checkpoint_path = args.checkpoint
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


args = parser.parse_args()
inference(args)