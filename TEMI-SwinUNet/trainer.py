import argparse
import logging
import os
import random
import sys
import time
from sched import scheduler

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
from utils import test_single_volume
#滑动窗口版
def sliding_window(image, window_size, stride):
    windows = []
    positions = []
    img_h, img_w = image.shape[:2]
    window_h, window_w = window_size
    stride_h, stride_w = stride

    for i in range(0, img_h, stride_h):
        for j in range(0, img_w, stride_w):
            # 截取窗口，考虑图像的边界
            window = image[i:min(i + window_h, img_h), j:min(j + window_w, img_w)]
            windows.append(window)
            positions.append((i, j))  # 记录窗口的位置
    return windows, positions

def calculate_mdice(outputs, labels, num_classes, include_background=False):
    dice_per_class = []
    smooth = 1e-5

    # 确定类别范围：如果 include_background 为 True，包含背景类（类0），否则从1开始
    class_range = range(num_classes) if include_background else range(1, num_classes)

    for i in class_range:
        # 获取预测类别和标签中的类别i的像素
        intersection = ((outputs == i) & (labels == i)).sum().float()
        union = ((outputs == i) | (labels == i)).sum().float()

        # 计算 Dice 系数
        dice = (2. * intersection + smooth) / (union + smooth)
        dice_per_class.append(dice)

    # 返回平均的 mDice
    return sum(dice_per_class) / len(dice_per_class)
def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator,RandomGenerator1
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))
    db_val = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="test",
                             transform=transforms.Compose(
                                 [RandomGenerator1(
                                     output_size=[args.img_size, args.img_size])]))  # 可以去掉 RandomGenerator 或使用其他预处理
    print("The length of validation set is: {}".format(len(db_val)))
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=20, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    validloader = DataLoader(db_val, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train() 
    class_weights = torch.tensor([1.0, 2.0, 2.0, 3, 3])
    ce_loss = nn.CrossEntropyLoss(weight=class_weights, reduction='mean').cuda()
    #ce_loss = CrossEntropyLoss
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    epoch_loss = 0.0
    best_loss = float('inf')
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        model.train()
        for i_batch, sampled_batch in enumerate(trainloader):
            windows_data = sampled_batch['windows']
            labels_data = sampled_batch['labels']
            #image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            #image_batch, label_batch = windows_data.cuda(), labels_data.cuda()
            loss = 0.0
            for window, label in zip(windows_data, labels_data):
                window = window.cuda()
                label = label.cuda()
                outputs = model(window.unsqueeze(0))  # 增加batch维度

                loss_ce = ce_loss(outputs, label.long())
                loss_dice = dice_loss(outputs, label, softmax=True)
                loss += (0.5 * loss_ce + 0.5 * loss_dice)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))
        epoch_loss /= len(trainloader)
        logging.info(f"Epoch {epoch_num} : Average Loss: {epoch_loss}")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_loss_model_path = os.path.join(snapshot_path, 'best_loss_model.pth')
            torch.save(model.state_dict(), best_loss_model_path)
            logging.info(f"Best loss model saved with Loss: {best_loss:.4f} at epoch {epoch_num}")

        torch.cuda.empty_cache()
        # 每个 epoch 后评估 mDice，并保存最佳模型
        model.eval()
        with torch.no_grad():
            val_outputs, val_labels = [], []
            for val_batch in validloader:  # 你需要定义 val_loader
                val_image, val_label = val_batch['image'].cuda(), val_batch['label'].cuda()
                windows, positions = sliding_window(val_image.cpu().numpy(), (args.img_size, args.img_size),
                                                    (args.stride, args.stride))
                output_full = torch.zeros_like(val_label)  # 初始化全零的输出图像
                for window, (i, j) in zip(windows, positions):
                    window = torch.tensor(window).cuda()
                    output = model(window.unsqueeze(0))
                    output_full[i:i + args.img_size, j:j + args.img_size] = output.argmax(dim=1).cpu()
                val_outputs.append(output_full)
                val_labels.append(val_label)

        # 计算 mDice
        mdice = calculate_mdice(torch.cat(val_outputs), torch.cat(val_labels), num_classes)
        logging.info(f'Epoch {epoch_num} : mDice: {mdice}')

        # 保存最佳模型
        if mdice > best_performance:
            best_performance = mdice
            best_model_path = os.path.join(snapshot_path, 'epoch_' + 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"Best model saved with mDice: {best_performance} at epoch {epoch_num}")


        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"