import numpy as np
import torch
import torch.nn as nn


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def train(trainloader, model, criterion, optimizer, device):
    # switch to train mode
    model.train()

    losses = AverageMeter()
    top1 = AverageMeter()

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)


        # compute output
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        prec1 = accuracy(outputs.float().data, targets)[0]
        losses.update(loss.float().item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))

    return (losses.avg, top1.avg)


@torch.no_grad()
def test(testloader, model, criterion, device):
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1 = accuracy(outputs.float().data, targets)[0]
        losses.update(loss.float().item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
    return (losses.avg, top1.avg)
        