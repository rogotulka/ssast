import os, csv, argparse, wget
import torch, torchaudio, timm
import numpy as np
from torch.cuda.amp import autocast
from models.ast_models import ASTModel



def make_features(wav_name, mel_bins, target_length=1024):
    waveform, sr = torchaudio.load(wav_name)
    assert sr == 16000, 'input audio sampling rate must be 16kHz'

    fbank = torchaudio.compliance.kaldi.fbank(
        waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
        window_type='hanning', num_mel_bins=mel_bins, dither=0.0, frame_shift=10)

    n_frames = fbank.shape[0]

    p = target_length - n_frames
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:target_length, :]

    fbank = (fbank - (-4.2677393)) / (4.5689974 * 2)
    return fbank


def load_label(label_csv):
    with open(label_csv, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        lines = list(reader)
    labels = []
    ids = []  # Each label has a unique id such as "/m/068hy"
    for i1 in range(1, len(lines)):
        id = lines[i1][1]
        label = lines[i1][2]
        ids.append(id)
        labels.append(label)
    return labels

# Create an AST model and download the AudioSet pretrained weights


def inference(args):
    # Assume each input spectrogram has 1024 time frames
    pretrain_stage = True if args.pretrain_stage == 'True' else False
    # now load the model
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
    print(f'[*INFO] load checkpoint: {args.checkpoint}')
    ast_mdl.eval()

    # Load the AudioSet label set
    label_csv = '/home/korovski_y_nfs/ast_project/ast/egs/speechcommands/data/speechcommands_class_labels_indices.csv'      # label and indices for audioset data
    labels = load_label(label_csv)
    audio_file = r'/home/korovski_y_nfs/ast_project/ast/egs/speechcommands/data/speech_commands_v0.02/right/97f4c236_nohash_2.wav'
    feats = make_features(audio_file, mel_bins=128, target_length=128)
    feats_data = feats.unsqueeze(0)
    with torch.no_grad():
        with autocast():
            output = ast_mdl.forward(feats_data, task=args.task)
            output = torch.nn.functional.softmax(output, dim=1)
            print('Top prediction: ', torch.argmax(output, dim=1).item())
            #output = torch.sigmoid(output)
    result_output = output.data.cpu().numpy()[0]
    sorted_indexes = np.argsort(result_output)[::-1]
    # Print audio tagging top probabilities
    print('Prediction probabilities:')
    for k in range(35):
        print('- {}: {:.4f}'.format(np.array(labels)[sorted_indexes[k]], result_output[sorted_indexes[k]]))




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
parser.add_argument('--task', type=str, help='Task for inference')


args = parser.parse_args()
inference(args)


