import torch
from tqdm import tqdm
from shopee.meters import TopKAccuracy
from shopee.cutmix import cutmix, cutmix_criterion
import numpy as np


def train_epoch(
        loader, model, device,
        criterion, optimizer, fp16,
        n_classes, is_arc, mixed_prob=0.5,
):
    model.train()
    train_loss = []
    bar = tqdm(loader)
    for (data, target) in bar:
        data, target = data.to(device, dtype=torch.float), target.to(device)
        optimizer.zero_grad()

        r = np.random.rand()
        alpha = np.random.uniform(low=0.8, high=1.)

        # if MIXED:
        if r < mixed_prob:
            input_image, targets = cutmix(
                data, target, alpha=alpha,
            )
            preds = model(input_image)
            loss = cutmix_criterion(preds, targets)
        else:
            if is_arc:
                preds, metric_logits = model(data)
                loss = criterion(preds, metric_logits, target, n_classes)
            else:
                preds = model(data)
                loss = criterion(preds, target)

        if fp16:
            from apex import amp
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        loss_np = loss.detach().cpu().numpy()
        train_loss.append(loss_np)
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
        bar.set_description(f'loss: {loss_np:.5f}, smth: {smooth_loss:.5f}')
    return train_loss


def val_epoch(loader, model, device, criterion, n_classes, is_arc):

    val_loss = 0
    acc = 0

    top_acc = TopKAccuracy(k=1)

    model.eval()
    with torch.no_grad():
        for (data, targets) in tqdm(loader):
            data, targets = data.to(device, dtype=torch.float), targets.to(device)

            if is_arc:
                preds, metric_logits = model(data)
                loss = criterion(preds, metric_logits, targets, n_classes, is_val=True)
            else:
                preds = model(data)
                loss = criterion(preds, targets)

            val_loss += loss.item()
            acc += (preds.argmax(1) == targets).float().mean().item()

            top_acc.update(targets, preds)

    n_total = len(loader)
    val_loss = val_loss / n_total
    acc = acc / n_total
    top_acc = top_acc.compute()

    print('acc', acc, 'top-1 acc', top_acc,)
    return val_loss, top_acc
