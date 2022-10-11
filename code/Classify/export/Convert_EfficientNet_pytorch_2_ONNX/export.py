'''
Export efficient-net model to onnx engine

author: phatnt
date: May-01-2022
'''
import os
import sys
import torch
import argparse
sys.path.append('./efficientnet')
from model import EfficientNet

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", type=str, default='../trained_models/source_model_efficientnet-b4_imgz_224.pt')
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--name", type=str, default='source_model_efficientnetb4_imgz_224_2206_batch_1.onnx')
    parser.add_argument("--opset", type=int, default=9)
    parser.add_argument("--verbose", action='store_true', default=False)
    return parser.parse_args()

def export(args):
    '''
        Export pytorch weight to onnx engine.
    '''
    print(f"[INFO] Exporting {args.weight} to Onnx engine.")
    assert args.batch > 0
    assert os.path.isfile(args.weight), f'[ERROR] {args.weight} not found!'

    checkpoint = torch.load(args.weight, map_location='cuda')
    imgsz = 224
    in_channels = 3
    num_classes = 2
    model = EfficientNet.from_name(f'efficientnet-b4', num_classes=num_classes,
                                    image_size=imgsz, in_channels=in_channels)
    
    print('[INFO] Model info:')
    # print(f'\t + Architecture:  EfficientNet-b{checkpoint["arch"]}')
    print(f'\t + Max batch size: {args.batch}')
    print(f'\t + Input shape: {args.batch} x {in_channels} x {imgsz} x {imgsz}')
    # print(f'\t + Ouput shape: {checkpoint["num_classes"]}')
    

    model.load_state_dict(checkpoint)
    model.set_swish(memory_efficient=False)
    model.eval()
    dummy_input = torch.randn(args.batch, in_channels, imgsz, imgsz, requires_grad=True)
    saved_name = args.name if args.name is not None else args.weight.replace('.pth', '.onnx')
    torch.onnx.export(model, dummy_input, saved_name,
                    opset_version = args.opset,
                    verbose=args.verbose, export_params=True, do_constant_folding=True)
    
    print(f'[INFO]: {saved_name} created!, Exporting Done!')

if __name__ == '__main__':
	args = parser_args()
	export(args)