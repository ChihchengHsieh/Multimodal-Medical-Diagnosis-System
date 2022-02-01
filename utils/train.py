from logging import exception
import os
import sys
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import seaborn as sn

import matplotlib.pyplot as plt

from datetime import datetime
from torch.autograd import Variable
from torch.utils.data import DataLoader
from IPython.display import clear_output
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay


class RecordUnit:
    pass


def split_dataset(dataset,  batch_size, traing_portion=.8, test_portion=.1, seed=123):
    train_dataset_len = int(len(dataset) * traing_portion)
    test_dataset_len = int(len(dataset) * test_portion)
    val_dataset_len = len(dataset) - (train_dataset_len + test_dataset_len)

    # how does it seperate tthem?
    (
        train_dataset,
        val_dataset,
        test_dataset
    ) = torch.utils.data.random_split(
        dataset=dataset,
        lengths=[train_dataset_len, val_dataset_len, test_dataset_len],
        generator=torch.Generator().manual_seed(seed)
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dataset.train_collate_fn
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dataset.test_collate_fn
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dataset.test_collate_fn
    )

    return train_dataloader, val_dataloader, test_dataloader


def transform_data(data, device):
    image, clinical_data, label = data
    image = image.to(device)
    label = label.to(device)
    clinical_numerical_data, clinical_categorical_data = clinical_data
    clinical_numerical_data = clinical_numerical_data.to(device)

    for col in clinical_categorical_data.keys():
        clinical_categorical_data[col] = clinical_categorical_data[col].to(
            device)

    clinical_data = (clinical_numerical_data, clinical_categorical_data)

    image = Variable(image, requires_grad=False)
    label = Variable(label, requires_grad=False)

    clinical_numerical_data, clinical_categorical_data = clinical_data
    clinical_numerical_data = Variable(
        clinical_numerical_data, requires_grad=False)

    for col in clinical_categorical_data.keys():
        clinical_categorical_data[col] = Variable(
            clinical_categorical_data[col], requires_grad=False)

    clinical_data = (clinical_numerical_data, clinical_categorical_data)

    return image, clinical_data, label


# implement test 1 epoch here
def train_epoch(epoch, model, device, dataloader, loss_fn, optimizer):
    model.train()
    model.to(device)

    batch_losses = []
    batch_accuracy = []
    batch_auc = []
    batch_pred = []
    batch_target = []

    for batch_idx, data in enumerate(dataloader):
        image, clinical_data, label = transform_data(data, device)
        optimizer.zero_grad()
        outputs = model(image, clinical_data)
        loss = loss_fn(outputs, label)
        loss.backward()
        optimizer.step()
        print("Epoch: {:d} Batch:  ({:d}) Train Loss: {:.4f}".format(
            epoch, batch_idx, loss.item()))
        sys.stdout.flush()

        batch_losses.append(loss.item())

        # want accuracy here.
        batch_accuracy.append(
            accuracy_score(
                label.detach().cpu().numpy().flatten(), (outputs.detach().cpu().numpy() > 0.5).astype('int64').flatten())
        )

        batch_pred.extend(outputs.detach().cpu().numpy())
        batch_target.extend(label.detach().cpu().numpy())

        RecordUnit.batch_pred = batch_pred
        RecordUnit.batch_target = batch_target

        try:
            auc = roc_auc_score(label.detach().cpu().numpy(
            ).flatten(), outputs.detach().cpu().numpy().flatten())
        except ValueError:
            auc = 0

        batch_auc.append(auc)

    train_loss = np.mean(batch_losses)
    train_acc = np.mean(batch_accuracy)
    train_auc = np.mean(batch_auc)

    print(
        f"Epoch {epoch} | Loss: {train_loss:.2f} | ACC: {train_acc*100:.2f}% | AUC: {train_auc:.2f}")

    return train_loss, train_acc, train_auc, batch_pred, batch_target


# implement train 1 epoch here
def test_epoch(epoch, model, device, dataloader, loss_fn):
    model.eval()
    model.to(device)

    batch_losses = []
    batch_accuracy = []
    batch_auc = []
    batch_pred = []
    batch_target = []

    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            image, clinical_data, label = transform_data(
                data, device)
            outputs = model(image, clinical_data)
            loss = loss_fn(outputs, label)
            batch_losses.append(loss.item())

            # want accuracy here.
            batch_accuracy.append(
                accuracy_score(
                    label.detach().cpu().numpy().flatten(), (outputs.detach().cpu().numpy() > 0.5).astype('int64').flatten())
            )

            batch_pred.extend(outputs.detach().cpu().numpy())
            batch_target.extend(label.detach().cpu().numpy())

            try:
                auc = roc_auc_score(label.detach().cpu().numpy(
                ).flatten(), outputs.detach().cpu().numpy().flatten())
            except ValueError:
                auc = 0

            batch_auc.append(auc)

            print("Epoch: {:d} Batch:  ({:d}) Test Loss: {:.4f}".format(
                epoch, batch_idx, loss.item()))
            sys.stdout.flush()

    test_loss = np.mean(batch_losses)
    test_acc = np.mean(batch_accuracy)
    test_auc = np.mean(batch_auc)

    print(
        f"Epoch {epoch} | Loss: {test_loss:.2f} | ACC: {test_acc*100:.2f}% | AUC: {test_auc:.2f}")

    return test_loss, test_acc, test_auc, batch_pred, batch_target


def get_loss(dataset, weighted, device):

    def loss(preds, target):
        if weighted:
            return dataset.weighted_loss(preds, target, device=device)

        else:
            criterion = nn.MultiLabelSoftMarginLoss()
            return criterion(preds, target)

    return loss


def plot_training(train_data, val_data):
    fig = plt.figure(figsize=(30, 10), dpi=80)

    if not plt.fignum_exists(fig.number):
        plt.show()

    plt.subplot().cla()

    plt.subplot(311)
    plt.plot([t['loss'] for t in train_data],
             marker='o', label='Training loss')
    plt.plot([v['loss'] for v in val_data],
             marker='o', label='Validation loss')
    plt.ylabel('Loss', fontsize=8)
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.draw()
    plt.pause(0.001)

    plt.subplot(312)
    plt.plot([t['acc'] for t in train_data],
             marker='o', label='Training Accuracy')
    plt.plot([v['acc'] for v in val_data],
             marker='o', label='Validation Accuracy')
    plt.ylabel('Accuracy', fontsize=8)
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.draw()
    plt.pause(0.001)

    plt.subplot(313)
    plt.plot([t['auc'] for t in train_data], marker='o', label='Training AUC')
    plt.plot([v['auc'] for v in val_data],
             marker='o', label='Validation AUC')
    plt.ylabel('AUC', fontsize=8)
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.draw()
    plt.pause(0.001)


def plot_training_v2(epoch, fig, subplots, train_data, val_data):
    (loss_sub, acc_sub, auc_sub) = subplots

    # loss_sub = fig.add_subplot(3,1,1)
    # acc_sub = fig.add_subplot(3,1,2)
    # auc_sub = fig.add_subplot(3,1,3)

    if not plt.fignum_exists(fig.number):
        plt.show()

    loss_sub.cla()
    acc_sub.cla()
    auc_sub.cla()

    fig.suptitle(f'Epoch {epoch}')

    loss_sub.set_title("LOSS")
    loss_sub.plot([t['loss'] for t in train_data],
                  marker='o', label='Training loss', color='steelblue')
    loss_sub.plot([v['loss'] for v in val_data],
                  marker='o', label='Validation loss', color='darkorange')

    loss_sub.legend(loc="upper left")

    acc_sub.set_title("Accuracy")
    acc_sub.plot([t['acc'] for t in train_data],
                 marker='o', label='Training Accuracy', color='steelblue')
    acc_sub.plot([v['acc'] for v in val_data],
                 marker='o', label='Validation Accuracy', color='steelblue')
    acc_sub.legend(loc="upper left")

    auc_sub.set_title("AUC")
    auc_sub.plot([t['auc'] for t in train_data],
                 marker='o', label='Training AUC')
    auc_sub.plot([v['auc'] for v in val_data],
                 marker='o', label='Validation AUC')
    auc_sub.set_xlabel('Epoch')
    auc_sub.legend(loc='upper left')

    plt.draw()
    plt.pause(0.001)


def plot_training_v3(epoch, train_data, val_data):

    clear_output(wait=True)

    fig, (loss_sub, acc_sub, auc_sub) = plt.subplots(
        3, figsize=(10, 10), dpi=80, sharex=True)

    fig.suptitle(f'Epoch {epoch}')

    loss_sub.set_title("LOSS")
    loss_sub.plot([t['loss'] for t in train_data],
                  marker='o', label='Training loss', color='steelblue')
    loss_sub.plot([v['loss'] for v in val_data],
                  marker='o', label='Validation loss', color='darkorange')
    loss_sub.legend(loc="upper left")

    acc_sub.set_title("Accuracy")
    acc_sub.plot([t['acc'] for t in train_data],
                 marker='o', label='Training Accuracy', color='steelblue')
    acc_sub.plot([v['acc'] for v in val_data],
                 marker='o', label='Validation Accuracy', color='darkorange')
    acc_sub.legend(loc="upper left")

    auc_sub.set_title("AUC")
    auc_sub.plot([t['auc'] for t in train_data],
                 marker='o', label='Training AUC', color='steelblue')
    auc_sub.plot([v['auc'] for v in val_data],
                 marker='o', label='Validation AUC', color='darkorange')
    auc_sub.set_xlabel('Epoch')
    auc_sub.legend(loc='upper left')

    plt.plot()
    plt.pause(0.01)


def print_confusion_matrix(pred, target, label_cols):

    cms = {}

    for idx, col in enumerate(label_cols):
        cms[col] = confusion_matrix(np.array(target)[:, idx].astype(
            bool), (np.array(pred) > 0.5)[:, idx], labels=list(range(2)))

    # print the confusion matrix here.
    for idx, col in enumerate(label_cols):
        print("="*20)
        print(col)
        print("="*20)

        # columns = []
        # indexes = []

        # # we do false first
        # if (~(np.array(pred)[:, idx] > 0.5).astype(bool)).any():
        #     columns.append("Pred_False")

        # if (np.array(pred)[:, idx] > 0.5).astype(bool).any():
        #     columns.append("Pred_True")

        # if (~(np.array(target)[:, idx]).astype(bool)).any():
        #     indexes.append("Target_False")

        # if (np.array(target)[:, idx]).astype(bool).any():
        #     indexes.append("Target_True")

        # if len(columns) >= 2 or len(indexes) >= 2:
        columns = ['Pred_False', 'Pred_True']
        indexes = ['Target_False', 'Target_True']

        RecordUnit.pred= pred
        RecordUnit.target= target
        RecordUnit.cms = cms
        RecordUnit.label_cols = label_cols

        df_cm= pd.DataFrame(cms[col], columns=columns, index=indexes)
        print(df_cm)
        print("="*20)

def train(
        num_epochs,
        model,
        dataset,
        dataloaders,
        optimizer,
        scheduler,
        device,
        loss_weighted=True,
        early_stop_count=3
):
    best_model_wts, best_loss= model.state_dict(), float("inf")
    counter= 0

    plt.ion()

    train_data= []
    val_data= []

    loss_fn= get_loss(dataset, weighted=loss_weighted, device=device)

    for epoch in range(1, num_epochs + 1):
        print("Epoch {}/{}".format(epoch, num_epochs))
        print("-" * 10)

        train_dataloader, val_dataloader, test_dataloader= dataloaders

        train_loss, train_acc, train_auc, train_pred, train_target= train_epoch(epoch, model, dataloader=train_dataloader,
                                                                                 loss_fn=loss_fn, optimizer=optimizer, device=device)

        train_data.append(
            {
                "loss": train_loss,
                "acc": train_acc,
                "auc": train_auc
            }
        )

        val_loss, val_acc, val_auc, val_pred, val_target= test_epoch(epoch, model,
                                                                      dataloader=val_dataloader, loss_fn=loss_fn, device=device)

        val_data.append(
            {
                "loss": val_loss,
                "acc": val_acc,
                "auc": val_auc
            }
        )

        scheduler.step(val_loss)

        if (val_loss < best_loss):
            best_loss= val_loss
            best_model_wts= model.state_dict()
            counter= 0

        else:
            counter += 1

        if not (early_stop_count is None) and counter > early_stop_count:
            break

        torch.save(best_model_wts, os.path.join("saved_models",
                   f"{val_auc:.4f}_{str(datetime.now())}".replace(":", "_")))

        # plot the training process
        # plot_training(train_data, val_data)
        plot_training_v3(epoch, train_data, val_data)

        print("Training CM")
        print_confusion_matrix(train_pred, train_target, dataset.labels_cols)
        print("Validation CM")
        print_confusion_matrix(val_pred, val_target, dataset.labels_cols)

    print(f"Best Validation Loss: {best_loss:.4f}")

    test_loss, test_acc, test_auc, test_pred, test_target= test_epoch(
        epoch,
        model,
        dataloader=test_dataloader,
        loss_fn=loss_fn,
        device=device,
    )

    print("Test CM")
    print_confusion_matrix(test_pred, test_target, dataset.labels_cols)

    print(
        f"Training Done | TEST LOSS {test_loss:.4f} | TEST ACC {test_acc:.4f} | TEST AUC {test_auc:.4f}")

    return train_data, val_data
