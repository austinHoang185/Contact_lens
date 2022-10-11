from genericpath import exists
import torch
import torch.nn as nn
import torchvision.models as model
from torchvision import transforms
import argparse
from PIL import *
import glob
import os
import shutil
from models import AlexNet
from GoogleNet.model import GoogLeNet

def parser_args():
    arg_parser = argparse.ArgumentParser(description='Test a model')
    arg_parser.add_argument("--folder_test", type=str, default="datasets/train")
    arg_parser.add_argument("--backbone", type=str, default="AlexNet")
    arg_parser.add_argument('--imgz', type=int, default=64)
    arg_parser.add_argument('--num_classes', type=int, default=2)
    arg_parser.add_argument('--in_channels', type=int, default=3)
    arg_parser.add_argument('--model_path', type=str, default=f"trained_models/source_model_{arg_parser.parse_args().backbone}_imgz_{arg_parser.parse_args().imgz}.pt", help='A model in trained_models')
    args = arg_parser.parse_args()
    return args

# Inference 
def show_information(args):
    print(f"[MODEL]: \t\t{args.model_path}")
    print(f"[IMAGE-SIZE]: \t\t{args.imgz}")
    print(f"[BACKBONE]: \t\t{args.backbone}")
    print(f"[NUM-CLASSES]: \t\t{args.num_classes}")
    print(f"[IN-CHANNELS]: \t\t{args.in_channels}")

def load_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlexNet(num_classes=args.num_classes, 
                    in_channels = args.in_channels).to(device)
    
    
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    return model

def classify(img_path, model, transforms):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PIL_img = Image.open(img_path)
    sm = nn.Softmax(dim=1)
    image_tensor = transforms(PIL_img).unsqueeze_(0).to(device)
    results_model = sm(model(image_tensor))
    probability = results_model.max().item()
    output_class = torch.max(results_model, 1)[1].item()
    return probability, output_class
    
def check_ovk_udk(class_pre, class_gt, img_path, img_name):
    if (class_pre != class_gt) and (class_gt == 'BB'):
            if os.path.exists("ovk")==False:
                os.mkdir("ovk")
            try:
                os.mkdir('ovk/'+str(class_pre))
                shutil.copy(img_path,'ovk/'+str(class_pre)+"/"+ img_name)
            except:
                shutil.copy(img_path,'ovk/'+str(class_pre)+"/"+ img_name)
                
    elif (class_pre != class_gt) and (class_gt == 'UD'):
        if os.path.exists("udk")==False:
            os.mkdir("udk")
        try:
            os.mkdir('udk/'+str(class_pre))
            shutil.copy(img_path,'udk/'+str(class_pre)+"/"+ img_name)
        except:
            shutil.copy(img_path,'udk/'+str(class_pre)+"/"+ img_name)

def infer(args, transforms):
    class_list = ['BB', 'UD']
    show_information(args)
    model = load_model(args)
    for img_path in glob.glob(args.folder_test+"/*/*"):
        class_gt = img_path.split("/")[-2]
        probability, output_class = classify( 
                                            img_path, 
                                            model, 
                                            transforms
                                            )
        class_pre = class_list[output_class]
        img_name = str(probability)[:10]+".png"
        check_ovk_udk(class_pre, class_gt, img_path, img_name)
        
                
        print("images_name: ",img_path)
        print("probability: ",probability)
        print("Class: ",class_pre)
        print("-"*25)
    

if __name__ == '__main__':
    args = parser_args()
    transforms = transforms.Compose([
                transforms.Resize((args.imgz, args.imgz)),
                transforms.ToTensor(),
                transforms.Normalize([0.456], [0.224])])
    infer(args, transforms)
    
    