import cv2
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from medpy import metric
import copy
from scipy.ndimage import zoom
import torch.nn as nn
from sklearn.metrics import precision_recall_curve, auc, average_precision_score
import numpy as np
import torch.nn.functional as F
class NALoss(nn.Module):
    def __init__(self, alpha=0.5, sigma=0.1, class_weights=None, noise_ratio=0.5):
        super(NALoss, self).__init__()
        self.alpha = alpha  # Single float value for all classes
        self.sigma = sigma
        self.class_weights = class_weights  # Loss weights for each class
        self.noise_ratio = noise_ratio

    def add_gaussian_noise(self, tensor):
        noise = torch.normal(mean=0, std=self.sigma, size=tensor.size()).to(tensor.device)
        return tensor + noise

    def forward(self, outputs, targets):
        batch_size, num_classes, height, width = outputs.size()

        # Convert targets to one-hot encoding if they are class indices
        if targets.ndimension() == 3:  # If targets are class indices, convert to one-hot encoding
            targets = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

        # Standard loss
        loss_clean = F.cross_entropy(outputs, targets.argmax(dim=1), reduction='none')

        # Add Gaussian noise to the ddrswinunet
        noisy_outputs = self.add_gaussian_noise(outputs)

        # Noisy loss
        loss_noise = F.cross_entropy(noisy_outputs, targets.argmax(dim=1), reduction='none')

        # Apply the same alpha to all classes
        class_loss = self.alpha * loss_noise  # Apply alpha directly

        # Final loss calculation
        final_loss = (1 - self.noise_ratio) * loss_clean + self.noise_ratio * class_loss

        # Apply class weights if provided
        if self.class_weights is not None:
            # Ensure class_weights is a tensor of shape [num_classes]
            self.class_weights = torch.tensor(self.class_weights).to(outputs.device)  # Ensure it's on the same device
            # Expand class_weights to [1, num_classes, 1, 1] for broadcasting
            class_weights_expanded = self.class_weights.unsqueeze(0).unsqueeze(2).unsqueeze(3)  # Shape [1, num_classes, 1, 1]
            final_loss = final_loss * class_weights_expanded

        return final_loss.mean()
class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_aupr(pred_probs, true_labels, classes):
    aupr_per_class = []
    for c in range(1, classes):  # 从 1 开始忽略背景类
        # 获取第 c 类的预测概率和真实标签
        pred_c = pred_probs[:, c, :, :].flatten()
        true_c = (true_labels == c).flatten()

        # 计算 Precision-Recall 曲线
        precision, recall, _ = precision_recall_curve(true_c, pred_c)

        # 计算 AUPR
        aupr = auc(recall, precision)
        aupr_per_class.append(aupr)

    # 计算平均 AUPR
    avg_aupr = np.mean(aupr_per_class)
    return aupr_per_class, avg_aupr


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1, 0
    else:
        return 0, 0


def test_single_volume_with_sliding_window(image, label, model, classes, window_size, overlap, test_save_path=None,
                                           case=None, resize_to=224):
    """
    对单个图像进行滑动窗口预测，并在输入网络前缩放到指定尺寸。
    """
    # h, w,_ = image.shape  # 输入图像的原始尺寸
    image = image.transpose(2, 0, 1)
    _, h, w = image.shape
    pred_map = np.zeros((classes, h, w), dtype=np.float32)  # 初始化预测概率图
    count_map = np.zeros((h, w), dtype=np.float32)  # 记录每个像素被预测的次数

    window_h, window_w = window_size
    stride_h = int(window_h * (1 - overlap))
    stride_w = int(window_w * (1 - overlap))

    # 滑动窗口预测
    for y in range(0, h, stride_h):
        for x in range(0, w, stride_w):
            y1, y2 = y, min(y + window_h, h)
            x1, x2 = x, min(x + window_w, w)

            # 裁剪窗口
            window = image[:, y1:y2, x1:x2]

            # 如果窗口不足 1024，则填充到目标大小
            padded_window = np.zeros((image.shape[0], window_h, window_w), dtype=window.dtype)
            padded_window[:, :y2 - y1, :x2 - x1] = window

            # 缩放到 224
            resized_window = cv2.resize(padded_window.transpose(1, 2, 0), (resize_to, resize_to),
                                        interpolation=cv2.INTER_LINEAR)
            resized_window = resized_window.transpose(2, 0, 1)  # 调整为 (C, H, W)

            # 转为 Tensor 并送入模型
            resized_window_tensor = torch.tensor(resized_window, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                preds = model(resized_window_tensor)  # 模型预测，形状为 (B, C, H, W)
                preds = torch.softmax(preds, dim=1).cpu().numpy()[0]  # 取 softmax 并去掉 batch 维度

            # 将预测结果还原到原始窗口大小
            preds_resized = cv2.resize(preds.transpose(1, 2, 0), (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
            preds_resized = preds_resized.transpose(2, 0, 1)  # 调整回 (C, H, W)

            # 累加到整体预测图
            pred_map[:, y1:y2, x1:x2] += preds_resized
            count_map[y1:y2, x1:x2] += 1

    # 平滑预测图
    pred_map /= count_map

    # 取每个像素点的最大概率类别作为预测结果
    pred_label = np.argmax(pred_map, axis=0)
    aupr_per_class = []
    pred_map_flat = pred_map.reshape(classes, -1).T  # (total_pixels, num_classes)
    aupr_list = []  # 用于存储每个类别的 AUPR
    for i in range(1, classes):  # 遍历每个类别
        # 取出当前类别的真实标签和预测概率
        y_true_class = (label == i).astype(int)  # 将当前类别的标签转换为二分类格式
        y_score_class = pred_map_flat[:, i]  # 当前类别的预测概率
        if np.sum(y_true_class) == 0:
            aupr_list.append(1)
            continue
        # 计算 Precision-Recall 曲线
        precision, recall, _ = precision_recall_curve(y_true_class.ravel(), y_score_class.ravel())

        # 计算 AUPR
        aupr = np.trapz(recall, precision)
        aupr_list.append(aupr)

    # 计算 Dice 系数等指标
    iou_list = []
    metric_list = []
    for i in range(1, classes):  # 从 1 开始忽略背景
        pred_i = (pred_label == i)
        label_i = (label == i)

        intersection = np.logical_and(pred_i, label_i).sum()
        union = pred_i.sum() + label_i.sum()
        dice = (2. * intersection) / (union + 1e-5)
        metric_list.append(dice)

        # 计算 IoU
        #union_area = np.logical_or(pred_i, label_i).sum()
        #iou = intersection / (union_area + 1e-5)
        #iou_list.append(iou)
        plt.step(recall, precision, where='post' )

        # 设置图像标题和标签
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves for All Classes')
        #plt.legend(loc='best')
        plt.grid()

    # 保存或显示图像
        if test_save_path:
            plt.savefig(f"{test_save_path}/PR_curve_class_{i}_case_{case}.png")
        plt.close()
    if test_save_path is not None:
        # 保存结果图像的代码
        a1 = copy.deepcopy(pred_label)
        a2 = copy.deepcopy(pred_label)
        a3 = copy.deepcopy(pred_label)

        a1[a1 == 1] = 255
        a1[a1 == 2] = 0
        a1[a1 == 3] = 255
        a1[a1 == 4] = 20

        a2[a2 == 1] = 255
        a2[a2 == 2] = 255
        a2[a2 == 3] = 0
        a2[a2 == 4] = 10

        a3[a3 == 1] = 255
        a3[a3 == 2] = 77
        a3[a3 == 3] = 0
        a3[a3 == 4] = 120

        a1 = Image.fromarray(np.uint8(a1)).convert('L')
        a2 = Image.fromarray(np.uint8(a2)).convert('L')
        a3 = Image.fromarray(np.uint8(a3)).convert('L')
        prediction = Image.merge('RGB', [a1, a2, a3])
        prediction.save(test_save_path + '/' + case + '.png')
    return metric_list, aupr_list


def test_single_volume(image, label, net, classes, patch_size=[224, 224], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()

    image = image.transpose(2, 0, 1)
    _, x, y = image.shape

    # 如果图像尺寸不匹配，进行缩放
    if x != patch_size[0] or y != patch_size[1]:
        image = zoom(image, (1, patch_size[0] / x, patch_size[1] / y), order=3)

    # 将图像转换为 PyTorch 张量
    input = torch.from_numpy(image).unsqueeze(0).float()

    net.eval()
    with torch.no_grad():
        out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
        out = out.cpu().detach().numpy()
        if x != patch_size[0] or y != patch_size[1]:
            # 缩放图像至原始大小
            prediction = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
        else:
            prediction = out

        out_prob = torch.softmax(net(input), dim=1)
        # 将概率分布转换为 NumPy 数组，shape 为 (num_classes, H, W)
        out_prob = out_prob.squeeze(0).cpu().detach().numpy()  # 移除 batch 维度

        # 如果图像尺寸不匹配，缩放回原始尺寸
        if x != patch_size[0] or y != patch_size[1]:
            prediction_prob = zoom(out_prob, (1, x / patch_size[0], y / patch_size[1]), order=3)
        else:
            prediction_prob = out_prob

    # 存储各类的指标
    metric_list = []
    dice_list = []  # 用于存储每类的 Dice 系数
    iou_list = []   # 用于存储每类的 IoU
    aupr_list = []  # 用于存储每类的 AUPR

    prediction_prob_flat = prediction_prob.transpose(1, 2, 0).reshape(-1, classes)  # 形状变为 (num_pixels, num_classes)

    for i in range(1, classes):  # 忽略背景类（假设背景为0）
        # 计算当前类的 Dice 系数
        metrics = calculate_metric_percase(prediction == i, label == i)
        metric_list.append(metrics)
        dice_list.append(metrics[0])  # 假设 metrics[0] 为 Dice 系数

        # 计算 IoU
        intersection = np.logical_and(prediction == i, label == i).sum()
        union = np.logical_or(prediction == i, label == i).sum()
        iou = intersection / union if union != 0 else 0.0
        iou_list.append(iou)

        # 计算 AUPR 并绘制 PR 曲线
        y_true = (label == i).astype(int)
        y_score = prediction_prob_flat[:, i]
        if np.sum(y_true) == 0:
            aupr_list.append(0.5)
            continue
        aupr = average_precision_score(y_true.ravel(), y_score)
        aupr_list.append(aupr)

        # 绘制 Precision-Recall 曲线
        precision, recall, _ = precision_recall_curve(y_true.ravel(), y_score)
        plt.figure()
        #plt.step(recall, precision, where='post', label=f'Class {i} (AUPR={aupr:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        #plt.title(f'Precision-Recall Curve for Class {i}')
        #plt.legend(loc='best')
        plt.grid()

        # 保存图像
        if test_save_path:
            plt.savefig(f"{test_save_path}/PR_curve_class_{i}_case_{case}.png")
        plt.close()

    if test_save_path is not None:
        # 保存结果图像
        a1 = copy.deepcopy(prediction)
        a2 = copy.deepcopy(prediction)
        a3 = copy.deepcopy(prediction)

        a1[a1 == 1] = 255
        a1[a1 == 2] = 0
        a1[a1 == 3] = 255
        a1[a1 == 4] = 20

        a2[a2 == 1] = 255
        a2[a2 == 2] = 255
        a2[a2 == 3] = 0
        a2[a2 == 4] = 10

        a3[a3 == 1] = 255
        a3[a3 == 2] = 77
        a3[a3 == 3] = 0
        a3[a3 == 4] = 120

        a1 = Image.fromarray(np.uint8(a1)).convert('L')
        a2 = Image.fromarray(np.uint8(a2)).convert('L')
        a3 = Image.fromarray(np.uint8(a3)).convert('L')
        prediction = Image.merge('RGB', [a1, a2, a3])
        prediction.save(test_save_path + '/' + case + '.png')

    return dice_list, aupr_list,iou_list  # 返回每个类别的Dice系数

