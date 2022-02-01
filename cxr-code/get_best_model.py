from __future__ import print_function
import glob
import os
import sys
import argparse
import json
import itertools

def print_helper(seq, verbose, n_print, reverse=False):
    for i, model_info in enumerate(sorted(seq, reverse=reverse)):
        if verbose:
            print(model_info)
        else:
            print(model_info[1], end=" ")
        if i == n_print:
            break


def get_best_predictions(path, n_print, verbose):
    predictions = []
    for timestamp in glob.glob(path + "/*"):
        for validation_file in glob.glob(timestamp + "/*-valid.npy"):
            basename = os.path.basename(validation_file)
            dirname = os.path.dirname(validation_file)
            with open(dirname + '/params.json') as json_data:
                params = json.load(json_data)
                num_models = params["num_models"]
            end_index = basename.find("-valid")
            AUC = float(basename[:end_index])
            predictions.append((AUC, dirname, num_models))
    print_helper(predictions, verbose, n_print, reverse=True)


def get_best_models(path, n, verbose, ungrouped):
    # from https://stackoverflow.com/questions/3368969/find-string-between-two-substrings
    def find_between(s, first, last):
        try:
            start = s.index(first ) + len( first )
            end = s.index( last, start )
            return s[start:end]
        except ValueError:
            return ""

    models = []
    for checkpoint_path in glob.glob(path + "/**/*epoch*"):
        val_loss = float(find_between(
            checkpoint_path, 'val', '_train'))
        models.append((val_loss, checkpoint_path))

    groups = itertools.groupby(models,
                lambda x: os.path.dirname(x[1]))

    models_group_best = []
    if ungrouped is True:
        models_group_best = models
    else:
        for _, model in groups:
            models_group_best.append(sorted(list(model))[0])

    return sorted(models_group_best)[:n]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'selection_type',
        choices={'model', 'prediction'},
        help="Whether to get the best models or best predictions")
    parser.add_argument(
        'folder',
        help="path to get best from")
    parser.add_argument('-n', '--n', type=int, default=20)
    parser.add_argument('--verbose', action="store_true")
    parser.add_argument('--ungrouped', action="store_true")
    args = parser.parse_args()
    if args.selection_type == 'model':
        models_group_best = get_best_models(args.folder, args.n, args.verbose, args.ungrouped)
        print_helper(models_group_best, args.verbose, args.n, reverse=False)
    elif args.selection_type == 'prediction':
        get_best_predictions(args.folder, args.n, args.verbose)
