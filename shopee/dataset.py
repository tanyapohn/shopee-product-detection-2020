from typing import Tuple

import albumentations as A
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from shopee.utils import get_image


def get_transform(image_size, normalize=True, train=True):
    if train:
        transforms = [
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30,
                                     val_shift_limit=30, p=0.6),
                A.RandomBrightnessContrast(brightness_limit=0.2,
                                           contrast_limit=0.2, p=0.6),
            ], p=0.9),
            A.Transpose(p=0.5),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            # A.CoarseDropout(
            #     max_holes=8, max_height=22,
            #     max_width=22, fill_value=255, p=0.7),
        ]
    else:
        transforms = [
        ]

    if normalize:
        transforms.append(A.Normalize())

    transforms.extend([
        ToTensorV2(),
    ])
    return A.Compose(transforms)


class ShopeeDataset(Dataset):
    def __init__(
            self, df: pd.DataFrame, image_dir: str,
            transform=None,
    ):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx) -> Tuple:
        item = self.df.iloc[idx]
        image = get_image(item, self.image_dir)

        # get label
        label = item.category
        data = {
            'image': image,
            'labels': label,
        }

        if self.transform is not None:
            data = self.transform(**data)

        image = data['image']
        label = torch.tensor(data['labels'], dtype=torch.long)

        return image, label
