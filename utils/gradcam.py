from PIL import Image
from model.wrapper import GradCAMWrapper
from utils.transform import TransformFuncs

from pytorch_grad_cam import  GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from utils.train import transform_data

def get_clinical_data(dataset, df, device):

    clinical_numerical_input = torch.tensor(
        np.array(df[dataset.clinical_numerical_cols]).astype('float')).float().to(device)

    clinical_categorical_input = {}

    for col in dataset.clinical_categorical_cols:
        clinical_categorical_input[col] = torch.tensor(
            np.array(df[col])).long().to(device)

    return (clinical_numerical_input, clinical_categorical_input)


def get_df_label_pred_img_input(model, dataset, idx, device):

    transform_func = TransformFuncs(image_size= dataset.image_size)

    model.eval()

    df = pd.DataFrame([dataset.__getitem__(idx)]).reset_index()

    labels_df = df[dataset.labels_cols]

    img = Image.open(df['image_path'][0]).convert("RGB")

    tensor_img = transform_func.tensor_norm_transform(img).to(device).unsqueeze(0)

    clinical_data = get_clinical_data(dataset, df, device)

    ## get prediction here as well.

    pred_df = pd.DataFrame(model(tensor_img, clinical_data).detach().cpu().numpy(), columns=dataset.labels_cols)

    return df, labels_df, pred_df, img, (tensor_img, clinical_data)


def get_df_label_pred_img_input_loss(model, loss_fn, dataset, idx, device):

    transform_func = TransformFuncs(image_size= dataset.image_size)

    model.eval()

    data = dataset.__getitem__(idx)

    input_data = dataset.collate_fn([data])

    image_array, clinical_data, label = transform_data(input_data, device)


    df = pd.DataFrame([data]).reset_index()

    labels_df = df[dataset.labels_cols]

    img = Image.open(df['image_path'][0]).convert("RGB")

    image_array = transform_func.tensor_norm_transform(img).to(device).unsqueeze(0)

    # clinical_data = get_clinical_data(dataset, df, device)

    ## get prediction here as well.

    output= model(image_array, clinical_data)

    loss = loss_fn(output, label)

    pred_df = pd.DataFrame(output.detach().cpu().numpy(), columns=dataset.labels_cols)

    return df, labels_df, pred_df, img, loss.item(), (image_array, clinical_data)
    
    



def show_gradCAMpp_result(dataset, model, desire_label_name, img, model_input, use_full_features=True ):

    transform_func = TransformFuncs(image_size= dataset.image_size)

    model.eval()

    (tensor_img, clinical_data) = model_input

    model.decision_net.net[-1] = nn.Identity()

    # replace the last one with identity

    wrapped_model = GradCAMWrapper(
        labels=dataset.labels_cols,
        desire_label_name=desire_label_name,
        model=model,
        prepared_clinical_data=clinical_data,
    )

    if use_full_features:
        # target_layer = wrapped_model.model.image_net.model_ft.features
        target_layer = wrapped_model.model.image_net.model_ft.features[-1]
    else:
        target_layer = wrapped_model.model.image_net.model_ft.features.denseblock4.denselayer16.conv2
        # target_layer = wrapped_model.model.image_net.model_ft.features[-1]
        # target_layer = wrapped_model.model.decision_net.net[0]
        # target_layer = wrapped_model.model.image_net.model_ft.features.norm5
        # target_layer = wrapped_model.model.image_net.model_ft.features.denseblock4.denselayer16.norm2

    gardcam_pp = GradCAMPlusPlus(model= wrapped_model, target_layers=[target_layer], use_cuda=True)

    targets = [ClassifierOutputTarget(category=0)]

    grayscale_cam = gardcam_pp(input_tensor=tensor_img, targets=targets)

    image_float_np = np.float32(transform_func.display_transform(img)) / 255

    cam_img = show_cam_on_image(image_float_np , grayscale_cam[0, :], use_rgb=True)

    model.decision_net.net[-1] = nn.Sigmoid()

    return Image.fromarray(cam_img)