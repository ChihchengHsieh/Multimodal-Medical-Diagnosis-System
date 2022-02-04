from __future__ import division

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import sklearn.metrics
import argparse
import os

from torch.autograd import Variable
from PIL import Image


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="densenet", type=str)
    parser.add_argument("--optimizer", default="adam", type=str)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--drop_rate", default=0.0, type=float)
    parser.add_argument("--epochs", default=15, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--seed", default=123456, type=int)
    parser.add_argument("--tag", default="", type=str)
    parser.add_argument("--toy", action="store_true")
    parser.add_argument("--save_path", default=None, type=str)
    parser.add_argument("--scale", default=224, type=int)
    parser.add_argument("--horizontal_flip", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--scratch", action="store_true")
    parser.add_argument("--train_weighted", action="store_true")
    parser.add_argument("--valid_weighted", action="store_true")
    parser.add_argument("--size", default=None, type=str)
    return parser



class Dataset(data.Dataset):
    def __init__(self, args, data_split):
        super(Dataset, self).__init__()

        if args.tag:
            tag = "_"+args.tag
        df = pd.read_csv(os.path.join("data/%s%s.csv" % (data_split, tag)))

        if args.toy:
            df = df.sample(frac=0.01)

        self.df = df
        self.img_paths = df["Path"].tolist()
        self.pathologies = [col for col in df.columns.values if col != "Path"]

        self.labels = df[self.pathologies].as_matrix().astype(int)

        self.n_classes = self.labels.shape[1]

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        if data_split == "train":
            transforms_lst = [
                transforms.Resize((args.scale, args.scale)),
                transforms.RandomHorizontalFlip() if args.horizontal_flip else None,
                transforms.ToTensor(),
                normalize,
            ]
            self.transform = transforms.Compose([t for t in transforms_lst if t])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((args.scale, args.scale)),
                transforms.ToTensor(),
                normalize,
            ])
        self.df = df


        if (data_split == "train" and args.train_weighted) or (data_split == "valid" and args.valid_weighted):
            self.get_weights(args, data_split)


    def get_weights(self, args, data_split):

        self.use_gpu = torch.cuda.is_available()
        p_count = (self.labels == 1).sum(axis = 0)
        self.p_count = p_count
        n_count = (self.labels == 0).sum(axis = 0)
        total = p_count + n_count

        # invert *opposite* weights to obtain weighted loss
        # (positives weighted higher, all weights same across batches, and p_weight + n_weight == 1)
        p_weight = n_count / total
        n_weight = p_count / total

        self.p_weight_loss = Variable(torch.FloatTensor(p_weight), requires_grad=False)
        self.n_weight_loss = Variable(torch.FloatTensor(n_weight), requires_grad=False)

        print ("Positive %s Loss weight:" % data_split, self.p_weight_loss.data.numpy())
        print ("Negative %s Loss weight:" % data_split, self.n_weight_loss.data.numpy())
        random_loss = sum((p_weight[i] * p_count[i] + n_weight[i] * n_count[i]) *\
                                               -np.log(0.5) / total[i] for i in range(self.n_classes)) / self.n_classes
        print ("Random %s Loss:" % data_split, random_loss)


    def __getitem__(self, index):
        img = Image.open(self.img_paths[index]).convert("RGB")
        label = self.labels[index]

        return self.transform(img), torch.LongTensor(label)

    def __len__(self):
        return len(self.img_paths)

    def weighted_loss(self, preds, target, epoch=1):

        weights = target.type(torch.FloatTensor) * (self.p_weight_loss.expand_as(target)) + \
                  (target == 0).type(torch.FloatTensor) * (self.n_weight_loss.expand_as(target))
        if self.use_gpu:
            weights = weights.cuda()
        loss = 0.0
        for i in range(self.n_classes):
            loss += nn.functional.binary_cross_entropy_with_logits(preds[:,i], target[:,i], weight=weights[:,i])
        return loss / self.n_classes


def evaluate(gts, probabilities, pathologies, use_only_index = None):
    assert(np.all(probabilities >= 0) == True)
    assert(np.all(probabilities <= 1) == True)

    def compute_metrics_for_class(i):
         p, r, t = sklearn.metrics.precision_recall_curve(gts[:, i], probabilities[:, i])
         PR_AUC = sklearn.metrics.auc(r, p)
         ROC_AUC = sklearn.metrics.roc_auc_score(gts[:, i], probabilities[:, i])
         F1 = sklearn.metrics.f1_score(gts[:, i], preds[:, i])
         acc = sklearn.metrics.accuracy_score(gts[:, i], preds[:, i])
         count = np.sum(gts[:, i])
         return PR_AUC, ROC_AUC, F1, acc, count

    PR_AUCs = []
    ROC_AUCs = []
    F1s = []
    accs = []
    counts = []
    preds = probabilities >= 0.5

    classes = [use_only_index] if use_only_index is not None else range(len(gts[0]))

    for i in classes:
        try:
            PR_AUC, ROC_AUC, F1, acc, count = compute_metrics_for_class(i)
        except ValueError:
            continue
        PR_AUCs.append(PR_AUC)
        ROC_AUCs.append(ROC_AUC)
        F1s.append(F1)
        accs.append(acc)
        counts.append(count)
        print('Class: {!s} Count: {:d} PR AUC: {:.4f} ROC AUC: {:.4f} F1: {:.3f} Acc: {:.3f}'.format(pathologies[i], count, PR_AUC, ROC_AUC, F1, acc))

    avg_PR_AUC = np.average(PR_AUCs)
    avg_ROC_AUC = np.average(ROC_AUCs, weights=counts)
    avg_F1 = np.average(F1s, weights=counts)

    print('Avg PR AUC: {:.3f}'.format(avg_PR_AUC))
    print('Avg ROC AUC: {:.3f}'.format(avg_ROC_AUC))
    print('Avg F1: {:.3f}'.format(avg_F1))
    return avg_PR_AUC, avg_ROC_AUC, avg_F1


def loader_to_gts(data_loader):
    gts = []
    for (inputs, labels) in data_loader:
        for label in labels.cpu().numpy().tolist():
            gts.append(label)
    gts = np.array(gts)
    return gts


def load_data(args):

    train_dataset = Dataset(args, "train")
    valid_dataset = Dataset(args, "valid")

    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, sampler=None)

    valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True, sampler=None)

    return train_loader, valid_loader
