# design the Fully Connected layer to process
from collections import OrderedDict
from msilib.schema import Error
from turtle import forward
import torch
import torch.nn as nn
import numpy as np

from torch.autograd import Variable

# from model.densenet import densenet121
from torchvision.models import densenet121


class REFLACXClincalNet(nn.Module):
    def __init__(
        self,
        num_numerical_features,
        output_dim,
        gender_emb_dim=64,
        dim=64,
        dropout=0.1,
    ) -> None:
        super(REFLACXClincalNet, self).__init__()

        self.gender_emb = nn.Embedding(2, gender_emb_dim,)
        self.net = nn.Sequential(
            nn.Linear(gender_emb_dim + num_numerical_features, dim),
            nn.Dropout(dropout),
            nn.LayerNorm(dim),
            nn.LeakyReLU(0.05),
            nn.Linear(dim, dim * 2),
            nn.Dropout(dropout),
            nn.LayerNorm(dim * 2),
            nn.LeakyReLU(0.05),
            nn.Linear(dim * 2, dim * 2),
            nn.Dropout(dropout),
            nn.LayerNorm(dim * 2),
            nn.LeakyReLU(0.05),
            nn.Linear(dim * 2, output_dim),
        )

    def forward(self, data):
        clinical_numerical_input, clinical_categorical_input = data
        gender_emb_out = self.gender_emb(clinical_categorical_input["gender"])

        concat_input = torch.cat((clinical_numerical_input, gender_emb_out), dim=1)

        return self.net(concat_input)


class ClinicalNet(nn.Module):
    def __init__(
        self,
        num_output_features,
        numerical_cols,
        categorical_cols,
        embedding_dim_maps,
        categorical_unique_map,
        device,
        dims=[16],
        gender_emb_dim=64,
    ) -> None:
        super(ClinicalNet, self).__init__()

        # update here to only use the data we want.

        fcs = []

        for idx, dim in enumerate(dims):
            if idx == 0:
                fcs.append(
                    nn.Linear(
                        len(numerical_cols) + sum(embedding_dim_maps.values()), dim
                    )
                )

            if idx == len(dims) - 1:
                fcs.append(nn.Linear(dim, num_output_features))

            if idx != 0 and idx == len(dim) - 1:
                fcs.append(nn.Linear(dims[idx - 1], dim))

        self.net = nn.Sequential(*fcs)

        self.embs = {}

        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols

        for col in categorical_cols:
            self.embs[col] = nn.Embedding(
                categorical_unique_map[col], embedding_dim_maps[col]
            ).to(device)

    def forward(self, data):
        # perform embedding first.
        clinical_numerical_input, clinical_categorical_input = data

        emb_out = {}

        for col in self.categorical_cols:
            emb_out[col] = self.embs[col](clinical_categorical_input[col])

        concat_input = torch.cat(
            (
                clinical_numerical_input,
                *[emb_out[col] for col in self.categorical_cols],
            ),
            dim=1,
        )

        return self.net(concat_input)


class ImageDenseNet(nn.Module):
    def __init__(self, num_output_features, pretrained=False):
        super(ImageDenseNet, self).__init__()
        self.model_ft = densenet121(pretrained=pretrained, drop_rate=0)
        num_ftrs = self.model_ft.classifier.in_features
        self.model_ft.classifier = nn.Linear(num_ftrs, num_output_features)

    def forward(self, x):
        return self.model_ft(x)


class DecisionNet(nn.Module):
    def __init__(
        self, num_input_features, num_output_features, dim, dropout=0.1
    ) -> None:
        super(DecisionNet, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(num_input_features, dim),
            nn.Dropout(dropout),
            nn.LayerNorm(dim),
            nn.LeakyReLU(0.05, inplace=True),
            nn.Linear(dim, dim * 2),
            nn.Dropout(dropout),
            nn.LayerNorm(dim * 2),
            nn.LeakyReLU(0.05, inplace=True),
            nn.Linear(dim * 2, dim * 2),
            nn.Dropout(dropout),
            nn.LayerNorm(dim * 2),
            nn.LeakyReLU(0.05, inplace=True),
            nn.Linear(dim * 2, num_output_features),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class FusionLayer(nn.Module):
    def __init__(self, fuse_type) -> None:
        super().__init__()
        self.fuse_type = fuse_type

    def forward(self, x, y):
        if self.fuse_type == "add":
            return x + y
        elif self.fuse_type == "concat":
            return torch.cat((x, y), dim=-1)
        else:
            raise Error("Not supported fusion type")


class AddFusionLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, y):
        return x + y


class ConcateFusionLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, y):
        return torch.cat((x, y), dim=-1)


class XAMIMultiModalSum(nn.Module):
    def __init__(
        self,
        reflacx_dataset,
        device,
        embeding_dim=64,
        joint_feature_size=64,
        model_dim=128,
        use_clinical=True,
        dropout=0.1,
        pretrained=True,
    ) -> None:
        super(XAMIMultiModalSum, self).__init__()

        self.device = device

        self.model_dim = model_dim

        categorical_unique_map = {}
        for col in reflacx_dataset.clinical_categorical_cols:
            categorical_unique_map[col] = torch.tensor(
                len(reflacx_dataset.df[col].unique())
            )

        # self.clinical_net = ClinicalNet(
        #     num_output_features=joint_feature_size,
        #     device=self.device,
        #     dims=[model_dim],
        #     numerical_cols=reflacx_dataset.clinical_numerical_cols,
        #     categorical_cols=reflacx_dataset.clinical_categorical_cols,
        #     embedding_dim_maps=ohe_dim_map,  # define embedding dim here.
        #     categorical_unique_map=categorical_unique_map
        # )

        self.image_net = ImageDenseNet(
            num_output_features=joint_feature_size, pretrained=pretrained,
        )

        self.use_clinical = use_clinical

        if self.use_clinical:
            self.fuse_layer = AddFusionLayer()
            self.clinical_net = REFLACXClincalNet(
                dropout=dropout,
                dim=model_dim,
                gender_emb_dim=embeding_dim,
                num_numerical_features=len(reflacx_dataset.clinical_numerical_cols),
                output_dim=joint_feature_size,
            )

        self.decision_net = DecisionNet(
            num_input_features=joint_feature_size,
            num_output_features=len(reflacx_dataset.labels_cols),
            dim=model_dim,
            dropout=dropout,
        )

    def forward(self, image, clincal_data):
        image_out = self.image_net(image)

        if self.use_clinical:
            clinical_out = self.clinical_net(clincal_data)
            fused_representation = self.fuse_layer(clinical_out, image_out)
        else:
            fused_representation = image_out

        decision_out = self.decision_net(fused_representation)
        return decision_out

    def num_all_params(self,) -> int:
        """
        return how many parameters in the model
        """
        return sum([param.nelement() for param in self.parameters()])


class XAMIMultiCocatModal(nn.Module):
    def __init__(
        self,
        reflacx_dataset,
        device,
        embeding_dim=64,
        joint_feature_size=64,
        model_dim=128,
        use_clinical=True,
        use_image=True,
        dropout=0.1,
        pretrained=True,
        detach_image=False,
    ) -> None:
        super(XAMIMultiCocatModal, self).__init__()

        self.device = device

        self.model_dim = model_dim

        categorical_unique_map = {}

        for col in reflacx_dataset.clinical_categorical_cols:
            categorical_unique_map[col] = torch.tensor(
                len(reflacx_dataset.df[col].unique())
            )

        # self.clinical_net = ClinicalNet(
        #     num_output_features=joint_feature_size,
        #     device=self.device,
        #     dims=[model_dim],
        #     numerical_cols=reflacx_dataset.clinical_numerical_cols,
        #     categorical_cols=reflacx_dataset.clinical_categorical_cols,
        #     embedding_dim_maps=ohe_dim_map,  # define embedding dim here.
        #     categorical_unique_map=categorical_unique_map
        # )

        decision_input_size = 0

        self.use_image = use_image
        if use_image:
            self.image_net = ImageDenseNet(
                num_output_features=joint_feature_size, pretrained=pretrained,
            )
            decision_input_size += joint_feature_size

        self.use_clinical = use_clinical
        if self.use_clinical:
            self.fuse_layer = ConcateFusionLayer()
            self.clinical_net = REFLACXClincalNet(
                dropout=dropout,
                dim=model_dim,
                gender_emb_dim=embeding_dim,
                num_numerical_features=len(reflacx_dataset.clinical_numerical_cols),
                output_dim=joint_feature_size,
            )
            decision_input_size += joint_feature_size

        self.detach_image = detach_image
        self.decision_net = DecisionNet(
            num_input_features=decision_input_size,
            num_output_features=len(reflacx_dataset.labels_cols),
            dim=model_dim,
            dropout=dropout,
        )

    def forward(self, image, clincal_data):

        if self.use_clinical and self.use_image:
            clinical_out = self.clinical_net(clincal_data)
            image_out = self.image_net(image)

            if self.detach_image:
                image_out = image_out.detach()

            fused_representation = self.fuse_layer(clinical_out, image_out)
        elif self.use_image:
            image_out = self.image_net(image)
            fused_representation = image_out
        elif self.use_clinical:
            clinical_out = self.clinical_net(clincal_data)
            fused_representation = clinical_out
        else:
            raise Error("Not modality is included.")

        decision_out = self.decision_net(fused_representation)
        return decision_out

    def num_all_params(self,) -> int:
        """
        return how many parameters in the model
        """
        return sum([param.nelement() for param in self.parameters()])
