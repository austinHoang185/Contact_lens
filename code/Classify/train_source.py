from genericpath import exists
import argparse
from numpy.core.defchararray import asarray
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from sklearn.metrics import confusion_matrix
import time
import os
import copy
import glob
import cv2
import sys
import random
from PIL import Image
from torch.autograd import Variable
import torchvision.models as models
from models import AlexNet, ResNet
from focal_loss import FocalLoss
from efficientnet.model import EfficientNet
from GoogleNet.model import GoogLeNet


def dataTransforms(imgz):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation((0,270)),
            transforms.Resize((imgz, imgz)),
            transforms.ToTensor(),
            transforms.Normalize([0.456], [0.224])
        ]),
        'val': transforms.Compose([
            transforms.RandomRotation((0,270)),
            transforms.Resize((imgz, imgz)),
            transforms.ToTensor(),
            transforms.Normalize([0.456], [0.224])
        ]),
        'test': transforms.Compose([
            transforms.Resize((imgz, imgz)),
            transforms.ToTensor(),
            transforms.Normalize([0.456], [0.224])
        ]),
    }
    return data_transforms


def dataLoader(data_dir, data_transforms):
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val', 'test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
                                                  shuffle=True, num_workers=4)
                   for x in ['train', 'val', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    
    class_names = image_datasets['train'].classes
    print(class_names)
    # print("len class: ", len(class_names))
    return dataloaders, dataset_sizes, len(class_names)


def train_model(model, dataloaders, dataset_sizes, device, num_epochs, save_model, backbone, lr):
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to training mode

            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    
                    if backbone != 'GoogLeNet':
                        outputs = model(inputs)
                    else:
                        if phase == 'train' and args.aux_logits:
                            outputs = model(inputs)[0]
                        else:
                            outputs = model(inputs)

                        # print("output_full: ",outputs)
                        # print("output: ",outputs[0])
                    _, preds = torch.max(outputs, 1)
                    # labels = F.one_hot(labels,4).type_as(outputs)
                    loss = criterion(input=outputs, target = labels.type(torch.cuda.LongTensor))
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if epoch_acc > best_acc and phase == 'val':
                print("saved!")
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), os.path.join(save_model, f'source_model_{args.backbone}_imgz_{args.imgz}.pt'))
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def loadModel(model_path, model_name, device, args):
    if args.backbone == 'AlexNet':  
        model = AlexNet(num_classes=args.num_classes)
    elif args.backbone == 'ResNet':
        model = ResNet(num_classes=args.num_classes)
    elif args.backbone == 'GoogLeNet':
        model = GoogLeNet(num_classes = args.num_classes, aux_logits=args.aux_logits, init_weights=False, in_channels=args.in_channels)
    else:
        model =  EfficientNet.from_name(f'{args.backbone}', num_classes=args.num_classes,image_size=args.imgz, in_channels=args.in_channels)

    model.load_state_dict(torch.load(os.path.join(model_path, model_name)))
    model.to(device)
    model.eval()
    return model


def validate(model, dataloaders, device):
    model.eval()
    predicts = np.array([])
    labels = np.array([])
    with torch.no_grad():
        for i, (inputs, label) in enumerate(dataloaders):
            inputs = inputs.to(device)
            label = label.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            preds = preds.cpu().detach().numpy()
            label = label.cpu().detach().numpy()
            predicts = np.append(predicts, preds)
            labels = np.append(labels, label)

        print("confusion matrix: ")
        print(confusion_matrix(labels, predicts))


def training(args):
    # parser parameters
    num_classes = args.num_classes
    data_dir = args.data_dir
    save_model = args.save_model
    epochs = args.epochs
    backbone = args.backbone
    aux_logits_status = args.aux_logits
    imgz = args.imgz
    lr = args.lr
    
    ####### data directory
    data_transform = dataTransforms(imgz)
    data_loader, datasets_size, num_classes = dataLoader(os.path.dirname(data_dir), data_transform)
    
    ####### model initiation
    
    if backbone == 'AlexNet':  
        model_ft = AlexNet(num_classes=num_classes)
    elif backbone == 'ResNet':
        model_ft = ResNet(num_classes=num_classes)
    elif backbone == 'GoogLeNet':
        model_ft = GoogLeNet(num_classes=num_classes, aux_logits=aux_logits_status, init_weights=False, in_channels=args.in_channels)
    else:
        model_ft =  EfficientNet.from_name(f'{backbone}', num_classes=num_classes,image_size=imgz, 
                                           in_channels=args.in_channels)
    
    ####### training
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = model_ft.to(device)
    
    print("training...")
    model_ft = train_model(model_ft, data_loader, datasets_size, device, epochs, save_model, backbone, lr)
    ######## testing
    print("testing...")
    resnet_best_weights = loadModel(save_model, f'source_model_{backbone}_imgz_{imgz}.pt', device, args)
    print("traning set: ")
    validate(resnet_best_weights, data_loader["train"], device)
    print("validation set: ")
    validate(resnet_best_weights, data_loader["val"], device)
    print("test set: ")
    validate(resnet_best_weights, data_loader["test"], device)
    return model_ft


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Test a model')
    arg_parser.add_argument("--backbone", type=str, default="GoogLeNet", help='AlexNet, ResNet, GoogLeNet, efficientnet-b0,1,2,3,4')
    arg_parser.add_argument('--data_dir', type=str, default="datasets_classify_after06092022/data_06092022/datasets_source/")
    arg_parser.add_argument('--save_model', type=str, default="trained_models/")
    arg_parser.add_argument('--imgz', type=int, default=256)
    arg_parser.add_argument('--num_classes', type=int, default=4)
    arg_parser.add_argument('--in_channels', type=int, default=3)
    arg_parser.add_argument('--epochs', type=int, default=50)
    arg_parser.add_argument('--lr', type=float, default=0.0001)
    # parser for GoogleNet
    arg_parser.add_argument('--aux_logits', action='store_true', default=False)
    args = arg_parser.parse_args()

    # train
    if os.path.exists(args.save_model)==False:
        os.mkdir(args.save_model)
    #  Print information 
    print("\n")
    print(f"[MODE-SAVE]: \t\t{args.save_model}")
    print(f"[IMAGE-SIZE]: \t\t{args.imgz}")
    print(f"[BACKBONE]: \t\t{args.backbone}")
    print(f"[NUM-CLASSES]: \t\t{args.num_classes}")
    print(f"[LEARNING-RATE]: \t{args.lr}")
    model = training(args)
