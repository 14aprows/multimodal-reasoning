from torchvision import transforms

def get_image_transform(image_size=224):
    if isinstance(image_size, int):
        image_size = (image_size, image_size)

    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def get_train_transform(image_size=224):
    return get_image_transform(image_size)

def get_test_transform(image_size=224):
    return get_image_transform(image_size)
