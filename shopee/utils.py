import pandas as pd
import os
import cv2
from typing import Tuple


def get_image(item: pd.Series, image_dir: str):
    # image_path = os.path.join(image_dir, f'{item.category:02}', item.filename)
    image_path = os.path.join(image_dir, item.filename)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def load_train_df(data_dir: str, train) -> pd.DataFrame:
    if train:
        df_path = os.path.join(data_dir, 'train_with_folds.csv')
    else:
        df_path = os.path.join(data_dir, 'test.csv')
    return pd.read_csv(df_path)


def load_train_valid_df(data_dir: str, fold: int, train=True) -> Tuple:

    df = load_train_df(data_dir, train)
    n_classes = int((df.nunique())[1:2])
    mask = df['fold'] == fold
    train = df[~mask]
    valid = df[mask]
    return train, valid, n_classes
