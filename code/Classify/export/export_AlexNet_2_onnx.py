'''
Export efficient-net model to onnx engine

author: phatnt
date: May-01-2022
'''
import os
import torch
from models import AlexNet

def export():
    weight = 'trained_models/source_model_AlexNet_imgz_64.pt'
    batch = 10
    '''
        Export pytorch weight to onnx engine.
    '''
    print(f"[INFO] Exporting {weight} to Onnx engine.")
    assert batch > 0
    assert os.path.isfile(weight), f'[ERROR] {weight} not found!'

    checkpoint = torch.load(weight, map_location='cuda')
    imgsz = 64
    in_channels = 3
    num_classes = 2
    model = AlexNet(num_classes = num_classes)

    model.load_state_dict(checkpoint)
    # model.set_swish(memory_efficient=False)
    model.eval()
    dummy_input = torch.randn(batch, in_channels, imgsz, imgsz, requires_grad=True)
    saved_name =   weight.replace('.pt', '')+f'_batchz{batch}.onnx'
    torch.onnx.export(model, dummy_input, saved_name,
                    opset_version = 9,
                    verbose=False, 
                    export_params=True, 
                    do_constant_folding=True)
    
    print(f'[INFO]: {saved_name} created!, Exporting Done!')

if __name__ == '__main__':
	export()