# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 11:38:38 2021

@author: Administrator
"""
import torch
import torch.nn as nn
import visdom
import torch.nn.functional as F
import Res
import datasets
import torch.utils.data as data
import torchvision.transforms as transforms
from torch import optim

def evaluate(model, loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    correct = 0
    total = len(loader.dataset)
    for x, y in loader:
        x, y = x.to(device), y.squeeze().long().to(device)
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
        correct += torch.eq(pred, y).sum().float().item()
    return correct / total

def train_model(epochs, train_loader, val_loader, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Res.ResNet18().to(device)
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)
    criteon = nn.CrossEntropyLoss()
    best_acc ,best_epoch = 0,0
    global_step = 0
    viz = visdom.Visdom()
    viz.line([0], [-1], win='loss', opts=dict(title='loss'))
    viz.line([0], [-1], win='val_acc', opts=dict(title='val_acc'))
    viz.line([0], [-1], win='train_acc', opts=dict(title='train_acc'))
    for epoch in range(epochs):
        for step,(x,y) in enumerate(train_loader):
            x,y = x.to(device), y.squeeze().long().to(device)
            model.train()
            logits = model(x)
            loss = criteon(logits,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([loss.item()], [global_step], win='loss', update='append')
            global_step += 1
        if epoch % 1 == 0:
            train_acc = evaluate(model, train_loader)
            viz.line([train_acc], [global_step], win='train_acc', update='append')
            val_acc = evaluate(model, val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                viz.line([val_acc], [global_step], win='val_acc', update='append')
                torch.save(model.state_dict(), 'params.pkl')
    print('train_acc:', train_acc)
    print('best acc:', best_acc, 'best epoch:', best_epoch)
    model.load_state_dict(torch.load('params.pkl'))
    print('loaded from ckpt!')
    test_acc = evaluate(model, test_loader)
    print('test_acc:', test_acc)
    
def main():
    batch_size = 128
    
    train_transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[.5], std=[.5])])

    val_transform = transforms.Compose( [transforms.ToTensor(),
                                             transforms.Normalize(mean=[.5], std=[.5])])

    test_transform = transforms.Compose( [transforms.ToTensor(),
                                             transforms.Normalize(mean=[.5], std=[.5])])
    train_dataset = MedMNIST('train',transform=train_transform)
    val_dataset = MedMNIST('val',transform=val_transform)
    test_dataset = MedMNIST('test',transform=test_transform)
    train_loader = data.DataLoader(dataset=train_dataset,
                                   batch_size=batch_size,
                                   shuffle=True)
    val_loader = data.DataLoader(dataset=val_dataset,
                                 batch_size=batch_size,
                                 shuffle=True)
    test_loader = data.DataLoader(dataset=test_dataset,
                                  batch_size=batch_size,
                                  shuffle=True)
    print('==> Building and training model...')
    train_model(20, train_loader, val_loader, test_loader)
    
main()
    
