import torch
import argparse
from GoogleNet.model import GoogLeNet
import os

# parser
arg_parser = argparse.ArgumentParser(description='Test a model')
arg_parser.add_argument("--weight", type=str, default='../trained_models/revgrad.pt')
arg_parser.add_argument('--imgz', type=int, default=256)
arg_parser.add_argument('--num_classes', type=int, default=6)
arg_parser.add_argument('--in_channels', type=int, default=3)
arg_parser.add_argument('--aux_logits', action='store_true', default=False)
arg_parser.add_argument("--batch", type=int, default=1)
arg_parser.add_argument("--name", type=str, default=None)
arg_parser.add_argument("--opset", type=int, default=9)
arg_parser.add_argument("--verbose", action='store_true', default=False)
args = arg_parser.parse_args()

# init model & load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GoogLeNet(num_classes = args.num_classes, aux_logits=args.aux_logits, init_weights=True)
model.load_state_dict(torch.load(args.weight, map_location='cuda'))
model.eval()

# convert
if args.name == None:
    name = args.weight.split("/")[-1].replace("pt",'onnx')
else:
    name = args.name


input_names = [ "actual_input" ]
output_names = [ "output" ]
dummy_input = torch.randn(args.batch, args.in_channels, args.imgz, args.imgz)
torch.onnx.export(model, 
                  dummy_input,
                  name,
                  input_names=input_names,
                  output_names=output_names,
                  verbose=args.verbose,
                  export_params=True,
                  opset_version = args.opset,
                  do_constant_folding=True
                  )

print("DONE")


