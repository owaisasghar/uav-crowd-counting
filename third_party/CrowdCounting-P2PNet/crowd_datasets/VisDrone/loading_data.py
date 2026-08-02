import torchvision.transforms as transforms

from .VisDrone import VisDronePoints


def loading_data(data_root, args):
    train_transform = transforms.Compose(
        [
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    validation_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    train_set = VisDronePoints(
        data_root,
        train=True,
        transform=train_transform,
        patch_size=args.patch_size,
        num_patches=args.num_patches,
        flip=True,
    )
    validation_set = VisDronePoints(
        data_root,
        train=False,
        transform=validation_transform,
        patch_size=args.patch_size,
        num_patches=1,
        flip=False,
    )
    return train_set, validation_set
