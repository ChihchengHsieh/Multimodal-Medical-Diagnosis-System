"""
Make valid/test predictions on a saved model.
"""
from __future__ import division
from __future__ import print_function
import numpy as np
import torch
import time
import os
import json
import argparse
import itertools
from sklearn import metrics
import util
from train import transform_data, DenseNet
from joblib import Memory


memory = Memory(cachedir='./cache', verbose=0)


# convert dictionary to object
class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def load_model(args, load_path, nclasses):
    model = DenseNet(args, nclasses)
    model.load_state_dict(torch.load(load_path))
    model = model.cuda()
    model.eval()
    return model


@memory.cache
def get_model_predictions(args_dict, model_path, loader):

    args = Struct(**args_dict)

    model = load_model(args, model_path, loader.dataset.n_classes)
    outs = []
    # gather predictions for all images in the validation set
    for i, (inputs, labels) in enumerate(loader):
        inputs, _ = transform_data((inputs, labels), use_gpu=True)
        outputs = model(inputs)
        out = torch.sigmoid(outputs).data.cpu().numpy()
        outs.append(out)
    outs = np.concatenate(outs, axis=0)
    return outs


@memory.cache
def get_loader_on_split(args_dict, split):

    if "tag" not in args_dict:
        args_dict["tag"] = args_dict["size"]

    args = Struct(**args_dict)

    dataset = util.Dataset(args, split)

    loader = torch.utils.data.DataLoader(
                    dataset, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, pin_memory=False)
    return loader


def optimal_threshold_compute(labels, probs):
    thresholds = []
    for i in range(probs.shape[1]):
        p, r, t = metrics.precision_recall_curve(labels[:, i], probs[:, i])
        threshold = t[np.nanargmax(2 * p * r / (p + r))]
        thresholds.append(threshold)
    thresholds = np.array(thresholds)
    print("Optimal thresholds at ", thresholds)
    return thresholds

def predict_for_split(args_dicts, model_paths, split):
    loaders = [get_loader_on_split(args_dict, split) for args_dict in args_dicts]
    all_model_probs = [get_model_predictions(args_dict, model_path, loader) \
            for args_dict, model_path, loader in zip(args_dicts, model_paths, loaders)]
    probs = np.mean(all_model_probs, axis=0)
    # labels should be the same across all models
    labels = loaders[0].dataset.labels

    if 'valid' in split:
        thresholds = optimal_threshold_compute(labels, probs)
        auc = metrics.roc_auc_score(labels, probs)
        print("AUC", auc)
        name = str(auc) + '-' + split
    else:
        name = split
        thresholds = None

    return probs, thresholds, name


def predict(model_paths, split="valid", save=True):

    def get_model_params(model_path):
        params = json.load(open(
            os.path.dirname(model_path) + '/params.txt', 'r'))
        return params

    model_args_dicts = [get_model_params(model_path) for model_path in model_paths]

    folder_name = 'predictions/' + str(int(round(time.time() * 1000)))

    probs, thresholds, name = predict_for_split(model_args_dicts, model_paths, split)
    
    if save:
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        np.save(folder_name + '/' + name, probs)
        print("Predictions saved to ", folder_name)

        params = {"model_paths": model_paths,
                  "num_models" : len(model_paths)}
        with open(folder_name + '/params.json', 'w') as outfile:
            json.dump(params, outfile)

        return folder_name + '/' + name + ".npy"

    else:

        return probs, thresholds


if __name__ == "__main__":
    """
    Usage
        python predict.py model1 model2 ... modelN
    """

    parser = util.get_parser()
    parser.add_argument(
        'model_paths',
        nargs='+',
        help="path to models")
    args = parser.parse_args()

    predict(args.model_paths, save=True)

