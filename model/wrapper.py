import torch.nn as nn

class GradCAMWrapper(nn.Module):
    def __init__(self,
                 prepared_clinical_data,
                 model,
                 desire_label_name,
                 labels=[
                     #  "Support devices",
                     "Enlarged cardiac silhouette",
                     "Atelectasis",
                     "Pleural abnormality",
                     "Consolidation",
                     "Pulmonary edema",
                 ],
                 ):
        super(GradCAMWrapper, self).__init__()

        # we prepare the clinical data before feeding the image to the model.
        self.prepared_clinical_data = prepared_clinical_data
        self.labels = labels
        self.desire_label_name = desire_label_name
        self.desire_label_idx = labels.index(desire_label_name)
        self.model = model

    def forward(self, input_image):
        return self.model(input_image, self.prepared_clinical_data)[:, self.desire_label_idx][None, :]
        
