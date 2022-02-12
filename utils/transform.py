import torchvision.transforms as transforms
from torch.autograd import Variable

class TransformFuncs:

    def __init__(self, image_size) -> None:
        normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        tensor_norm_transform_list = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]

        self.tensor_norm_transform = transforms.Compose(tensor_norm_transform_list)

        display_transform_list = [
            transforms.Resize((image_size, image_size)),
        ]

        self.display_transform = transforms.Compose(display_transform_list)


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