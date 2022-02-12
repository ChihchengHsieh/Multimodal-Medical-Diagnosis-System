import os
import sys
import torch
import numpy as np
import pandas as pd
import torch.nn as nn

import matplotlib.pyplot as plt

from datetime import datetime
from torch.utils.data import DataLoader
from IPython.display import clear_output
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

from libauc.losses import AUCM_MultiLabel
from libauc.optimizers import PESG

from utils.print import print_block
from utils.transform import transform_data


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


def get_loss(dataset, weighted, device):

    def loss(preds, target):
        if weighted:
            return dataset.weighted_loss(preds, target, device=device)

        else:
            criterion = nn.MultiLabelSoftMarginLoss()
            return criterion(preds, target)

    return loss


def plot_training(epoch, train_data, val_data):

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
        print_block(col)

        columns = ['Pred_False', 'Pred_True']
        indexes = ['Target_False', 'Target_True']

        df_cm = pd.DataFrame(cms[col], columns=columns, index=indexes)
        print(df_cm)
        print("="*40)


def train_with_chexnext(
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
    best_model_wts, best_loss = model.state_dict(), float("inf")
    counter = 0

    train_dataloader, val_dataloader, test_dataloader = dataloaders

    plt.ion()

    train_data = []
    val_data = []

    loss_fn = get_loss(dataset, weighted=loss_weighted, device=device)

    for epoch in range(1, num_epochs + 1):
        print_block(f"Epoch: {epoch}/{num_epochs}")

        train_loss, train_acc, train_auc, train_pred, train_target = train_epoch_chexnext(epoch, model, dataloader=train_dataloader,
                                                                                 loss_fn=loss_fn, optimizer=optimizer, device=device)

        train_data.append(
            {
                "loss": train_loss,
                "acc": train_acc,
                "auc": train_auc
            }
        )

        val_loss, val_acc, val_auc, val_pred, val_target = test_epoch(epoch, model,
                                                                      dataloader=val_dataloader, loss_fn=loss_fn, device=device)

        val_data.append(
            {
                "loss": val_loss,
                "acc": val_acc,
                "auc": val_auc
            }
        )

        if not scheduler is None:
            scheduler.step(val_loss)

        if (val_loss < best_loss):
            best_loss = val_loss
            best_model_wts = model.state_dict()
            counter = 0

        else:
            counter += 1

        if not (early_stop_count is None) and counter > early_stop_count:
            break

        # plot the training process
        # plot_training(train_data, val_data)
        plot_training(epoch, train_data, val_data)

        print(f"Current learning rate is {optimizer.param_groups[0]['lr']}")

        print("================Training CM================")
        print_confusion_matrix(train_pred, train_target, dataset.labels_cols)
        print("================Validation CM================")
        print_confusion_matrix(val_pred, val_target, dataset.labels_cols)

    print(f"Best Validation Loss: {best_loss:.4f}")

    test_loss, test_acc, test_auc, test_pred, test_target = test_epoch(
        epoch,
        model,
        dataloader=test_dataloader,
        loss_fn=loss_fn,
        device=device,
    )

    print("================Test CM================")
    print_confusion_matrix(test_pred, test_target, dataset.labels_cols)

    torch.save(best_model_wts, os.path.join("saved_models",
                                            f"{test_auc:.4f}_{str(datetime.now())}".replace(":", "_")))

    print(
        f"Training Done | TEST LOSS {test_loss:.4f} | TEST ACC {test_acc:.4f} | TEST AUC {test_auc:.4f}")

    return train_data, val_data


# 1. add the pretrain function.
# 2. rebuild the train function for auc.

def train_with_auc_margin_loss(
    num_epochs,
    model,
    dataloaders,
    dataset,
    device,
    scheduler_freq=750,
    scheduler_factor=5,
    gamma=500,
    weight_decay=1e-5,
    margin=1.0,
    lr=0.1,
    model_note="",
):

    train_dataloader, val_dataloader, test_dataloader = dataloaders

    plt.ion()

    imratio_list = [dataset.df[col].sum() / len(dataset)
                    for col in dataset.labels_cols]  # indicate the portion of positive cases.

    loss_fn = AUCM_MultiLabel(imratio=imratio_list,
                              num_classes=len(dataset.labels_cols))

    # define the optimiser

    optimizer = PESG(model,
                     a=loss_fn.a,
                     b=loss_fn.b,
                     alpha=loss_fn.alpha,
                     lr=lr,
                     gamma=gamma,
                     margin=margin,
                     weight_decay=weight_decay, device='cuda')

    best_val_auc = 0

    train_data = []
    val_data = []

    batch_count = 0
    best_model_name = None

    clinial_cond = "With" if model.use_clinical else "Without"

    for epoch in range(1, num_epochs+1):

        # log current epoch.
        print_block(f"Epoch: {epoch}/{num_epochs}")

        # feed the data into the model to get the result.
        (
            batch_count,
            train_loss,
            train_acc,
            train_auc,
            train_pred,
            train_target
        ) = train_epoch_auc(
            epoch,
            batch_count,
            model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            scheduler_freq=scheduler_freq,
            scheduler_factor=scheduler_factor,
        )

        # record the result.
        train_data.append(
            {
                "loss": train_loss,
                "acc": train_acc,
                "auc": train_auc
            }
        )

        # once we have been trained 1 epoch, we perfrom test on validation dataset.
        (
            val_loss,
            val_acc,
            val_auc,
            val_pred,
            val_target
        ) = test_epoch(
            epoch,
            model,
            dataloader=val_dataloader,
            loss_fn=loss_fn,
            device=device
        )

        # record the validation information.

        val_data.append(
            {
                "loss": val_loss,
                "acc": val_acc,
                "auc": val_auc
            }
        )

        # update the best AUC and ave the best model.
        if val_acc > best_val_auc:
            # update best AUC.
            best_val_auc = val_auc


            # Save the model.
            best_model_name = f"val_{val_auc:.4f}_{model_note}_epoch{epoch}_{clinial_cond}Clincal_dim{model.model_dim}_{str(datetime.now())}".replace(
                ":", "_").replace(".", "_")

            torch.save(
                model.state_dict(),
                os.path.join('saved_models',best_model_name) ,
            )

        # plot loss, accuracy and AUC curves.
        plot_training(epoch, train_data, val_data)

        # Log current information.
        print_block(
            f"{optimizer.param_groups[0]['lr']}",
            title="Current Learning Rate",
        )

        print_block(f"LOSS {train_loss:.4f} | ACC {train_acc:.4f} | AUC {train_auc: .4f}", title="Training Result")
        print_block("Training Confusion Matrix")
        print_confusion_matrix(
            train_pred,
            train_target,
            dataset.labels_cols
        )

        print_block(f"LOSS {val_loss:.4f} | ACC {val_acc:.4f} | AUC {val_auc: .4f}", title="Validation Result")
        print_block("Validation Confusion Matrix")
        print_confusion_matrix(
            val_pred,
            val_target,
            dataset.labels_cols
        )

    # tell the best validation AUC we get.
    print_block(f"{best_val_auc}", title="Best Validation AUC")

    # perform testing.
    test_loss, test_acc, test_auc, test_pred, test_target = test_epoch(
        epoch,
        model,
        dataloader=test_dataloader,
        loss_fn=loss_fn,
        device=device,
    )

    print_block("Testing Confusion Matrix")

    print_confusion_matrix(test_pred, test_target, dataset.labels_cols)

    print_block(f"LOSS {test_loss:.4f} | ACC {test_acc:.4f} | AUC {test_auc: .4f}", title="Training Done - Test Result")

    final_model_path =  f"test_{test_auc:.4f}_{model_note}_epoch{epoch}_{clinial_cond}Clincal_dim{model.model_dim}_{str(datetime.now())}".replace(":", "_").replace(".", "_")

    torch.save(
                model.state_dict(),
                os.path.join('saved_models',final_model_path) ,
            )

    print_block(best_model_name, title="Best Model")

    print_block(final_model_path, "Test Model")

    return (train_pred, train_target), (val_pred, val_target), (test_pred, test_target), (best_model_name, final_model_path)

def train_epoch_auc(
        epoch,
        batch_count,
        model,
        device,
        dataloader,
        loss_fn,
        optimizer,
        scheduler_freq=200,
        scheduler_factor=.5,
):
    model.train()
    model.to(device)

    batch_losses = []
    batch_pred = []
    batch_target = []

    for _, data in enumerate(dataloader):

        batch_count += 1

        image, clinical_data, label = transform_data(data, device)
        y_pred = model(image, clinical_data)
        loss = loss_fn(y_pred, label)
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        if (not scheduler_freq is None) and batch_count % scheduler_freq == 0:
            optimizer.update_regularizer(decay_factor=scheduler_factor)

        # print the batch_training informatino.
        print_block(
            f"LOSS {loss.item():.4f}",
            title=f"| Training Epoch: {epoch} | Batch: {batch_count} |"
        )

        sys.stdout.flush()

        batch_losses.append(loss.item())
        batch_pred.extend(y_pred.detach().cpu().numpy())
        batch_target.extend(label.detach().cpu().numpy())

    train_loss = np.mean(batch_losses)
    train_acc = accuracy_score(
        np.array(batch_target).flatten(), np.array(batch_pred).flatten() > 0.5)
    train_auc = roc_auc_score(np.array(batch_target), np.array(batch_pred))

    print_block(
        f"LOSS {train_loss:.2f} | ACC: {train_acc*100:.2f}% | AUC: {train_auc:.2f}",
        title=f"| Epoch {epoch} Training Done! |"
    )

    return batch_count, train_loss, train_acc, train_auc, batch_pred, batch_target


def train_epoch_chexnext(epoch, model, device, dataloader, loss_fn, optimizer, ):
    model.train()
    model.to(device)

    batch_losses = []
    batch_pred = []
    batch_target = []

    for batch_idx, data in enumerate(dataloader):
        image, clinical_data, label = transform_data(data, device)
        y_pred = model(image, clinical_data)
        # y_pred = torch.sigmoid(y_pred)
        loss = loss_fn(y_pred, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("Epoch: {:d} Batch:  ({:d}) Train Loss: {:.4f}".format(
            epoch, batch_idx, loss.item()))
        sys.stdout.flush()

        batch_losses.append(loss.item())
        batch_pred.extend(y_pred.detach().cpu().numpy())
        batch_target.extend(label.detach().cpu().numpy())

    train_loss = np.mean(batch_losses)
    train_acc = accuracy_score(
        np.array(batch_target).flatten(), np.array(batch_pred).flatten() > 0.5)
    train_auc = roc_auc_score(np.array(batch_target), np.array(batch_pred))

    print(
        f"Epoch {epoch} | Loss: {train_loss:.2f} | ACC: {train_acc*100:.2f}% | AUC: {train_auc:.2f}")

    return train_loss, train_acc, train_auc, batch_pred, batch_target


# implement train 1 epoch here
def test_epoch(epoch, model, device, dataloader, loss_fn):
    model.eval()
    model.to(device)

    batch_losses = []
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
            batch_pred.extend(outputs.detach().cpu().numpy())
            batch_target.extend(label.detach().cpu().numpy())

            print_block(
                f"LOSS {loss.item():.4f}",
                title=f"| Training Epoch: {epoch} | Batch: {batch_idx} |"
            )

            sys.stdout.flush()

    test_loss = np.mean(batch_losses)
    test_acc = accuracy_score(
        np.array(batch_target).flatten(), np.array(batch_pred).flatten() > 0.5)
    test_auc = roc_auc_score(np.array(batch_target), np.array(batch_pred))

    print_block(
        f"LOSS {test_loss:.2f} | ACC: {test_acc*100:.2f}% | AUC: {test_auc:.2f}",
        title=f"| Epoch {epoch} Testing Done! |"
    )

    return test_loss, test_acc, test_auc, batch_pred, batch_target


def get_aus_loss(dataset):
    imratio_list = [dataset.df[col].sum() / len(dataset)
                for col in dataset.labels_cols]  # indicate the portion of positive cases.

    loss_fn = AUCM_MultiLabel(imratio=imratio_list,
                                num_classes=len(dataset.labels_cols))

    return loss_fn