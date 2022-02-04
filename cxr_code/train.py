"""
Train script for CheXNet
"""
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import pprint
import warnings
import json
from torch.optim import lr_scheduler
from torch.autograd import Variable
from densenet import densenet121

import util


class DenseNet(nn.Module):
    def __init__(self, config, nclasses):
        super(DenseNet, self).__init__()
        self.model_ft = densenet121(
            pretrained=not config.scratch, drop_rate=config.drop_rate)
        num_ftrs = self.model_ft.classifier.in_features
        self.model_ft.classifier = nn.Linear(num_ftrs, nclasses)
        self.config = config

    def forward(self, x):
        return self.model_ft(x)


def transform_data(data, use_gpu, train=False):
    inputs, labels = data
    labels = labels.type(torch.FloatTensor)
    if use_gpu is True:
        inputs = inputs.cuda()
        labels = labels.cuda()
    inputs = Variable(inputs, requires_grad=False, volatile=not train)
    labels = Variable(labels, requires_grad=False, volatile=not train)
    return inputs, labels


def train_epoch(epoch, args, model, loader, criterion, optimizer):
    model.train()
    batch_losses = []
    for batch_idx, data in enumerate(loader):
        inputs, labels = transform_data(data, True, train=True)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels, epoch=epoch)
        loss.backward()
        optimizer.step()
        print("Epoch: {:d} Batch: {:d} ({:d}) Train Loss: {:.6f}".format(
            epoch, batch_idx, args.batch_size, loss.data[0]))
        sys.stdout.flush()
        batch_losses.append(loss.data[0])
    train_loss = np.mean(batch_losses)
    print("Training Loss: {:.6f}".format(train_loss))
    return train_loss


def test_epoch(model, loader, criterion, epoch=1):
    """
    Returns: (AUC, ROC AUC, F1, validation loss)
    """
    model.eval()
    test_losses = []
    outs = []
    gts = []
    for data in loader:
        for gt in data[1].numpy().tolist():
            gts.append(gt)
        inputs, labels = transform_data(data, True, train=False)
        outputs = model(inputs)
        loss = criterion(outputs, labels, epoch=epoch)
        test_losses.append(loss.data[0])
        out = torch.sigmoid(outputs).data.cpu().numpy()
        outs.extend(out)
    avg_loss = np.mean(test_losses)
    print("Validation Loss: {:.6f}".format(avg_loss))
    outs = np.array(outs)
    gts = np.array(gts)
    return util.evaluate(gts, outs, loader.dataset.pathologies) + (avg_loss,)


def get_loss(dataset, weighted):

    criterion = nn.MultiLabelSoftMarginLoss()

    def loss(preds, target, epoch):

        if weighted:

            return dataset.weighted_loss(preds, target, epoch=epoch)

        else:

            return criterion(preds, target)

    return loss


def run(args):

    use_gpu = torch.cuda.is_available()
    model = None

    train, val = util.load_data(args)
    nclasses = train.dataset.n_classes
    print("Number of classes:", nclasses)

    if args.model == "densenet":
        model = DenseNet(args, nclasses)
    else:
        print("{} is not a valid model.".format(args.model))

    if use_gpu:
        model = model.cuda()

    train_criterion = get_loss(
        train.dataset, args.train_weighted)  # train_weighted = True

    val_criterion = get_loss(val.dataset, args.valid_weighted)

    if args.optimizer == "adam":
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.model_ft.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay)
    elif args.optimizer == "rmsprop":
        optimizer = optim.RMSprop(
            filter(lambda p: p.requires_grad, model.model_ft.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay)
    else:
        print("{} is not a valid optimizer.".format(args.optimizer))

    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=1, threshold=0.001, factor=0.1)
    best_model_wts, best_loss = model.state_dict(), float("inf")

    counter = 0
    for epoch in range(1, args.epochs + 1):
        print("Epoch {}/{}".format(epoch, args.epochs))
        print("-" * 10)
        train_loss = train_epoch(
            epoch, args, model, train, train_criterion, optimizer)
        _, epoch_auc, _, valid_loss = test_epoch(
            model, val, val_criterion, epoch)
        scheduler.step(valid_loss)

        if (valid_loss < best_loss):
            best_loss = valid_loss
            best_model_wts = model.state_dict()
            counter = 0
        else:
            counter += 1

        if counter > 3:
            break

        torch.save(best_model_wts, os.path.join(args.save_path,
                   "val%f_train%f_epoch%d" % (valid_loss, train_loss, epoch)))

    print("Best Validation Loss:", best_loss)


if __name__ == "__main__":
    """
    Usage
        Download the images data at https://nihcc.app.box.com/v/ChestXray-NIHCC
        To train on the original labels:
            python train.py --save_path run_dir --model densenet --batch_size 8 --horizontal_flip --epochs 10 --lr 0.0001 --train_weighted --valid_weighted --scale 512
        To train on the relabels:
            python train.py --save_path run_dir --model densenet --batch_size 8 --horizontal_flip --epochs 10 --lr 0.0001 --train_weighted --valid_weighted --scale 512 --tag relabeled
    """
    parser = util.get_parser()
    args = parser.parse_args()
    pp = pprint.PrettyPrinter()
    pp.pprint(vars(args))

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)

    with open(os.path.join(args.save_path, "params.txt"), 'w') as out:
        json.dump(vars(args), out, indent=4)
    run(args)
