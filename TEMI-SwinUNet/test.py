import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset_synapse import Synapse_dataset
from utils import test_single_volume,test_single_volume_with_sliding_window
from networks.vision_transformer import SwinUnet as ViT_seg
from trainer import trainer_synapse
from config import get_config

parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='./DDR/', help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--dataset', type=str,
                    default='DDR', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=5, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_ddr', help='list dir')
parser.add_argument('--output_dir', type=str, default='./output',  help='output dir')
parser.add_argument('--max_iterations', type=int,default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--is_savenii', action="store_true",default=True,  help='whether to save results during inference')
parser.add_argument('--test_save_dir', type=str, default='./ddrsqqwanzheng', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                            'full: cache all data, '
                            'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')

args = parser.parse_args()
if args.dataset == "Synapse":
    args.volume_path = os.path.join(args.volume_path, "test_vol_h5")
config = get_config(args)

def inference(args, model, test_save_path=None):
    db_test = args.Dataset(base_dir=args.volume_path, split="test", list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=0)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()

    aupr_sums = np.zeros(args.num_classes - 1)  # 用于累加每个类别的Dice系数
    dice_sums = np.zeros(args.num_classes - 1)  # 用于累加每个类别的Dice系数
    iou_sums = np.zeros(args.num_classes - 1)  # 用于累加每个类别的Dice系数
    total_cases = len(testloader)

    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        #dice_list, aupr_list= test_single_volume(image, label, model, classes=args.num_classes,patch_size=[args.img_size, args.img_size],z_spacing=args.z_spacing)
        dice_list , aupr_list,iou_list= test_single_volume(image, label, model, classes=args.num_classes,
                                                  test_save_path=test_save_path, case=case_name,)
        dice_sums += np.array(dice_list)  # 累加每个类别的Dice系数
        aupr_sums += np.array(aupr_list)  # 累加每个类别的Dice系数
        iou_sums += np.array(iou_list)  # 累加每个类别的Dice系数
        logging.info(f'Case {case_name}, iou per class: {iou_list}')
        logging.info(f'Case {case_name}, Dice per class: {dice_list}')
        logging.info(f'Case {case_name}, AUPR per class: {aupr_list}')

        # 计算每个类的平均Dice
    mean_dice_per_class = dice_sums / total_cases
    mean_aupr_per_class = aupr_sums / total_cases
    mean_iou_per_class = iou_sums / total_cases
    for i, iou in enumerate(mean_iou_per_class, 1):
        logging.info(f'Mean iou for class {i}: {iou:.4f}')
    for i, dice in enumerate(mean_dice_per_class, 1):
        logging.info(f'Mean Dice for class {i}: {dice:.4f}')
    for i, aupr in enumerate(mean_aupr_per_class, 1):
        logging.info(f'Mean aupr for class {i}: {aupr:.4f}')
    # 计算总体平均Dice
    mean_dice = np.mean(mean_dice_per_class)
    mean_aupr = np.mean(mean_aupr_per_class)
    mean_iou = np.mean(mean_iou_per_class)
    logging.info(f'Mean Dice across all classes: {mean_iou:.4f}')
    logging.info(f'Mean Dice across all classes: {mean_dice:.4f}')
    logging.info(f'Mean aupr across all classes: {mean_aupr:.4f}')

    return "Testing Finished!"

if __name__ == "__main__":

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


    dataset_config = {
        'DDR': {
            'Dataset': Synapse_dataset,
            'volume_path': args.volume_path,
            'list_dir': './lists/lists_ddr',
            'num_classes': 5,
            'z_spacing': 1,
        },
    }
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.is_pretrain = True

    net = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes)

    snapshot = 'ddrlog/sqq/1/epoch_best_model.pth'
    net.load_state_dict(torch.load(snapshot, map_location=torch.device('cpu')))

    snapshot_name = snapshot.split('/')[-1]

    log_folder = './test_log/test_log_'
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    if args.is_savenii:
        args.test_save_dir = os.path.join(args.output_dir, "./ddrsqqwanzheng")
        test_save_path = args.test_save_dir
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    inference(args, net, test_save_path)
    #inference_with_sliding_window(args, net, test_save_path)
