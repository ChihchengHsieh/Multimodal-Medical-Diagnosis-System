import torchvision.transforms as transforms

class TransformFuncs:
    
    normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    tensor_norm_transform_list = [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        normalize,
    ]

    tensor_norm_transform = transforms.Compose(tensor_norm_transform_list)

    display_transform_list = [
        transforms.Resize((256, 256)),
    ]

    display_transform = transforms.Compose(display_transform_list)