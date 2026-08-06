import albumentations as A


def get_train_augmentations(image_size: int = 224):
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(p=0.2),
            A.Rotate(limit=10, p=0.3),
            A.Resize(image_size, image_size),
        ]
    )


def get_val_augmentations(image_size: int = 224):
    return A.Compose([A.Resize(image_size, image_size)])
