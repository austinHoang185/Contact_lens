import argparse
import numpy as np
import torch
import cv2
import torchvision
import os
import shutil
from PIL import *
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
import torch.nn as nn
from models import AlexNet, ResNet
import glob
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from efficientnet.model import EfficientNet
from GoogleNet.model import GoogLeNet


def Test_time_augmentation(model,PIL_img, train_transforms):
    best_probability=0

    sm = nn.Softmax(dim=1)
    for i in range(3):
        img = PIL_img.rotate(90*i, Image.NEAREST, expand = 1)
        image_tensor = train_transforms(img).unsqueeze_(0).to(device)
        results_model = sm(model(image_tensor))
        probability = results_model.max().item()
        if probability > best_probability:
            best_probability = probability
            best_class =  torch.max(results_model, 1)[1].item()
    return best_probability, best_class
 

def inference_image(args, img_path, model, train_transforms):
    PIL_img = Image.open(img_path)
    if args.test_time_augment:
        # Test time aug for inference
        probability, output_class = Test_time_augmentation(model,PIL_img, train_transforms)
    else:
        # Basic Inference
        sm = nn.Softmax(dim=1)
        image_tensor = train_transforms(PIL_img).unsqueeze_(0).to(device)
        results_model = sm(model(image_tensor))
        probability = results_model.max().item()
        output_class = torch.max(results_model, 1)[1].item()
    
    
    print("images_name: ",img_path)
    print("probability: ",probability)
    print("Class: ",output_class)
    print("-"*25)
    img_name = str(probability)[:10]+".png"
    if os.path.exists("images_results")==False:
        os.mkdir('images_results')
    try:
        os.mkdir('images_results/'+str(output_class))
        shutil.copy(img_path,'images_results/'+str(output_class)+"/"+ img_name)
    except:
        shutil.copy(img_path,'images_results/'+str(output_class)+"/"+ img_name)
    


def main(args):
    
    backbone = args.backbone
    train_transforms = transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize((args.imgz, args.imgz)),
                transforms.ToTensor(),
                transforms.Normalize([0.456], [0.224])])

    dataset = torchvision.datasets.ImageFolder(args.folder_test, transform = train_transforms)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False,
                            drop_last=False, num_workers=2, pin_memory=True)

    if backbone == 'AlexNet':  
        model = AlexNet(num_classes=args.num_classes).to(device)
    elif backbone == 'ResNet':
        model = ResNet(num_classes=args.num_classes).to(device)
    elif backbone == 'GoogLeNet':
        model = GoogLeNet(num_classes = args.num_classes, aux_logits=args.aux_logits, init_weights=True).to(device)
    else:
        model =  EfficientNet.from_name(f'{args.backbone}', num_classes=args.num_classes,image_size=args.imgz, 
                                        in_channels=args.in_channels).to(device)

    model.load_state_dict(torch.load(args.backbone))
    model.eval()

    total_accuracy = []
    with torch.no_grad():
        sm = nn.Softmax(dim=1)
        print(dataloader)
        for x, y_true in tqdm(dataloader, leave=False):
            x, y_true = x.to(device), y_true.to(device)
            y_pred = model(x)
            # probability = str(sm(y_pred).max().item())[0:5]
            output_class = torch.max(sm(y_pred), 1)[1].item()
            total_accuracy.append((output_class == y_true).float().mean().item())
    
    mean_accuracy = np.mean(total_accuracy)
    tqdm.write(f'Accuracy on target data: {mean_accuracy:.4f} ')


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Test a model')
    arg_parser.add_argument('--batch-size', type=int, default=1)
    arg_parser.add_argument("--folder_test", type=str, default="datasets/train")
    arg_parser.add_argument("--backbone", type=str, default="GoogLeNet", help='AlexNet, ResNet, GoogLeNet, efficientnet-b0,1,2,3,4')
    arg_parser.add_argument('--imgz', type=int, default=128)
    arg_parser.add_argument('--num_classes', type=int, default=2)
    arg_parser.add_argument('--in_channels', type=int, default=1)
    arg_parser.add_argument('--aux_logits', action='store_true', default=False)
    arg_parser.add_argument('--test_time_augment', action='store_true', default=False)
    arg_parser.add_argument('--model-path', type=str)
    #['BCE', 'Gap', 'Hump', 'Tear', 'Torn']
    args = arg_parser.parse_args()
    #  Print information 
    print("\n")
    print(f"[MODEL]: \t\t{args.model_path}")
    print(f"[IMAGE-SIZE]: \t\t{args.imgz}")
    print(f"[BACKBONE]: \t\t{args.backbone}")
    print(f"[NUM-CLASSES]: \t\t{args.num_classes}")
    print(f"[TEST-TIME-AUGMENT]: \t{args.test_time_augment}")
    # main(args)

    # predict images dir   
    # Load model
    
    if args.backbone == 'AlexNet':  
        model = AlexNet(num_classes=args.num_classes).to(device)
    elif args.backbone == 'ResNet':
        model = ResNet(num_classes=args.num_classes).to(device)
    elif args.backbone == 'GoogLeNet':
        model = GoogLeNet(num_classes = args.num_classes, aux_logits=args.aux_logits, init_weights=True, in_channels=args.in_channels).to(device)
    else:
        model =  EfficientNet.from_name(f'{args.backbone}', num_classes=args.num_classes,image_size=args.imgz, 
                                        in_channels=args.in_channels).to(device)
        
        
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    
    train_transforms = transforms.Compose([
                transforms.Resize((args.imgz, args.imgz)),
                transforms.ToTensor(),
                transforms.Normalize([0.456], [0.224])])
    
    for img_path in glob.glob(args.folder_test+"/*"):
        inference_image(args, img_path, model, train_transforms)

