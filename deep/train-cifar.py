import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader

from models.resnet import resnet18
from utils import train, test, accuracy, AverageMeter


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    model_path = 'resnet18.pth.tar'
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    trainset = datasets.CIFAR10(root='./data', train=True, download=True, 
                        transform=transform_train)
    testset = datasets.CIFAR10(root='./data', train=False, transform=transform_test)

    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)

    model = resnet18()
    model = nn.DataParallel(model).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    best_acc = 0
    for epoch in range(args.epochs):
        train_loss, train_acc = train(trainloader, model, criterion, optimizer, device)
        test_loss, test_acc = test(testloader, model, criterion, device)
        print(f'Epoch {epoch + 1} Train Loss {train_loss:.4f} ' + 
            f'Test Loss {test_loss:.4f} Train Acc {train_acc:.4f} Test Acc {test_acc:.4f}')
        
        # Save the best model
        is_best = test_acc > best_acc
        if is_best:
            torch.save(model.state_dict(), model_path)
        best_acc = max(test_acc, best_acc)

    print(f'Best acc: {best_acc}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    main(args)