"""
Implements RevGrad:
Unsupervised Domain Adaptation by Backpropagation, Ganin & Lemptsky (2014)
Domain-adversarial training of neural networks, Ganin et al. (2016)
"""
import argparse
import torchvision
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import config
import numpy as np
from data import MNISTM
from models import Efficientnet, AlexNet, ResNet, DANet
from utils import GrayscaleToRgb, GradientReversal
from GoogleNet.model import GoogLeNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    train_transforms = transforms.Compose([
                transforms.Resize((args.imgz, args.imgz)),
                transforms.ToTensor(),
                transforms.Normalize([0.456], [0.224])])
                
    if args.backbone == 'AlexNet':  
        model = AlexNet().to(device)
        discriminator = DANet(input=9216).to(device)
        
    elif args.backbone == 'ResNet':
        model = ResNet().to(device)
        discriminator = DANet(input=1000).to(device)
        
    elif args.backbone == 'Efficientnet':
        model = Efficientnet().to(device)
        discriminator = DANet(input=1000).to(device)
    
    elif args.backbone == 'GoogLeNet':
        model = GoogLeNet(num_classes = args.num_classes, aux_logits=False, init_weights=False, in_channels=args.in_channels).to(device)
        discriminator = DANet(input=1024).to(device)

    model.load_state_dict(torch.load(args.MODEL_FILE))
    
    feature_extractor = model.feature_extractor
    clf = model.classifier
    half_batch = args.batch_size // 2    
    ##################### 
    source_dataset = torchvision.datasets.ImageFolder(args.data_source_dir, transform = train_transforms)
    target_dataset = torchvision.datasets.ImageFolder(args.data_target_dir, transform = train_transforms)
    
    source_loader = DataLoader(source_dataset, batch_size = half_batch , shuffle = True)
    target_loader = DataLoader(target_dataset, batch_size = half_batch , shuffle = True)
    ##################### 

    optim = torch.optim.Adam(list(discriminator.parameters()) + list(model.parameters()),lr=0.0001)

    # Loss functional
    CELoss = nn.CrossEntropyLoss() 
    
    for epoch in range(1, args.epochs+1):
        batches = zip(source_loader, target_loader)
        n_batches = min(len(source_loader), len(target_loader))

        total_loss = []
        total_accuracy = []
        source_loss = []
        count = 0 
        for (source_x, source_labels), (target_x, _) in tqdm(batches, leave=False, total=n_batches):
                x = torch.cat([source_x, target_x])
                x = x.to(device)
                domain_y = torch.cat([torch.ones(source_x.shape[0]),
                                      torch.zeros(target_x.shape[0])])
                domain_y = domain_y.to(device)
                label_y = source_labels.to(device)
                features = feature_extractor(x)
                features = torch.flatten(features, 1)
                domain_preds = discriminator(features).squeeze()
                label_preds = clf(features[:source_x.shape[0]])
                
                domain_loss = F.binary_cross_entropy_with_logits(domain_preds, domain_y)
                label_loss = CELoss(label_preds, label_y)
                loss = domain_loss + label_loss
                
                optim.zero_grad()
                loss.backward()
                optim.step()
                count +=1
                total_loss.append((loss.item()/2))
                source_loss.append((label_loss.item()))
                total_accuracy.append((label_preds.max(1)[1] == label_y).float().mean().item())

        mean_loss = np.mean(total_loss)
        mean_accuracy = np.mean(total_accuracy)
        mean_source_loss = np.mean(source_loss)
        tqdm.write(f'EPOCH {epoch:03d}: total_loss={mean_loss:.4f}, domain_loss = {mean_loss-mean_source_loss:.4f}, source_loss={mean_source_loss:.4f}, '
                   f'source_accuracy={mean_accuracy:.4f}')

        torch.save(model.state_dict(), 'trained_models/revgrad.pt')


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Domain adaptation using RevGrad')
    arg_parser.add_argument('MODEL_FILE', help='A model in trained_models')
    arg_parser.add_argument('--batch-size', type=int, default=16)
    arg_parser.add_argument('--imgz', type=int, default=256)
    arg_parser.add_argument('--num_classes', type=int, default=4)
    arg_parser.add_argument('--in_channels', type=int, default=3)
    arg_parser.add_argument('--data_source_dir', type=str, default="datasets_classify_after06092022/data_06092022/datasets_source/")
    arg_parser.add_argument('--data_target_dir', type=str, default="datasets_classify_after06092022/data_06092022/datasets_target/")
    arg_parser.add_argument("--backbone", type=str, default="GoogLeNet", help='AlexNet, ResNet, Efficientnet')
    arg_parser.add_argument('--epochs', type=int, default=19)
    args = arg_parser.parse_args()
    main(args)