import argparse
import os
import random
import time

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from warmup_scheduler import GradualWarmupScheduler

from shopee.dataset import ShopeeDataset, get_transform
from shopee.engine import train_epoch, val_epoch
from shopee.loss import FocalLoss, arc_criterion
from shopee.model import build_model
from shopee.models.enet import enet_arcface_v2, Mish_module, Swish_module
from shopee.utils import load_train_valid_df


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg('--backbone', default='resnext50_32x4d_ssl', type=str)
    arg('--debug', type=int, default=0)
    # Data path params
    arg('--data-path', type=str, help='data folder')
    arg('--image-train-dir', type=str)
    arg('--image-test-dir', type=str)
    arg('--image-size', type=int)

    # Training params
    arg('--batch-size', type=int, default=64)
    arg('--fold', type=int)
    arg('--use-neck', type=int)
    arg('--criterion', type=str, default='ce')
    arg('--lr', default=3e-4, type=float, help='initial learning rate')
    arg('--warmup-factor', default=1, type=int)
    arg('--warmup-epoch', default=5, type=int)
    arg('--min-lr', default=1e-7, type=float)
    arg('--optimizer', default='sgd', type=str)
    arg('--momentum', default=0.9, type=float)
    arg('--weight-decay', default=1e-4, type=float, help='weight decay')
    arg('--scheduler', default='cosine', type=str)
    arg('--epochs', type=int)
    arg('--device', default=0, type=int)
    arg('--fp16', default=True, type=bool)

    # Save path param
    arg('--output-dir', type=str)

    args = parser.parse_args()
    print('Loading data')

    def seed_everything(seed=44):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    seed_everything()

    train_df, valid_df, n_classes = load_train_valid_df(args.data_path, args.fold)
    if args.debug:
        train_df = train_df.iloc[:10000]
        valid_df = valid_df.iloc[:10000]

    train_dataset = ShopeeDataset(
        train_df, image_dir=args.image_train_dir,
        transform=get_transform(image_size=args.image_size, train=True),
    )
    valid_dataset = ShopeeDataset(
        valid_df, image_dir=args.image_train_dir,
        transform=get_transform(image_size=args.image_size, train=False),
    )

    data_train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=RandomSampler(train_dataset),

    )
    data_valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size,
        sampler=SequentialSampler(valid_dataset),
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(
        backbone=args.backbone, n_classes=n_classes,
        use_neck=args.use_neck,
    )
    # model = enet_arcface_v2(
    #     backbone=args.backbone,
    #     out_dim=n_classes, act_fn=Swish_module(),
    # )
    model = model.to(device)

    optimizer = Adam(model.parameters(), lr=args.lr / args.warmup_factor)
    # scheduler_cosine = CosineAnnealingLR(optimizer, args.epochs - args.warmup_epoch)
    scheduler_cosine = StepLR(optimizer, step_size=3, gamma=0.8,)
    scheduler = GradualWarmupScheduler(
        optimizer, multiplier=args.warmup_factor,
        total_epoch=args.warmup_epoch,
        after_scheduler=scheduler_cosine
    )
    if args.fp16:
        from apex import amp
        model, optimizer = amp.initialize(
            model, optimizer, opt_level="O1",
        )

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    is_arc = False
    if args.criterion == 'ce':
        criterion_fn = nn.CrossEntropyLoss()
    elif args.criterion == 'focal':
        criterion_fn = FocalLoss(n_classes)
    elif args.criterion == 'arc':
        criterion_fn = arc_criterion
        is_arc = True

    acc_max = 0.
    best_file = f'best_fold{args.fold}.pth'

    optimizer.zero_grad()
    optimizer.step()
    for epoch in range(1, args.epochs + 1):
        print(time.ctime(), 'Epoch:', epoch)
        scheduler.step(epoch - 1)
        train_loss = train_epoch(
            data_train_loader, model,
            device, criterion_fn, optimizer,
            args.fp16, n_classes, is_arc,
        )
        val_loss, acc = val_epoch(
            data_valid_loader, model,
            device, criterion_fn,
            n_classes, is_arc,
        )

        content = (
                time.ctime()
                + ' '
                + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, \
                train loss: {np.mean(train_loss):.5f}, \
                val loss: {np.mean(val_loss):.5f}, top-1 acc: {acc:.5f}'
        )
        print(content)
        with open(os.path.join(args.output_dir, f'log_fold{args.fold}.txt'), 'a') as appender:
            appender.write(content + '\n')

        if acc > acc_max:
            print(f'score2 ({acc_max:.5f} --> {acc:.5f}).  Saving model ...')
            torch.save(model.state_dict(), os.path.join(args.output_dir, best_file))
            acc_max = acc

    torch.save(model.state_dict(), os.path.join(args.output_dir, f'final_fold{args.fold}.pth'))


if __name__ == '__main__':
    main()

