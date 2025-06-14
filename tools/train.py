# ------------------------------------------------------------------------------  
# Copyright (c) Microsoft  
# Licensed under the MIT License.  
# Written by Bin Xiao (Bin.Xiao@microsoft.com)  
# ------------------------------------------------------------------------------  

from __future__ import absolute_import  
from __future__ import division  
from __future__ import print_function  

import argparse  
import os  
import pprint  
import shutil  

import torch  
import torch.nn.parallel  
import torch.backends.cudnn as cudnn  
import torch.optim  
import torch.utils.data  
import torchvision.transforms as transforms  
from tensorboardX import SummaryWriter  

import _init_paths  
from config import cfg  
from config import update_config  
from core.loss import JointsMSELoss  
from core.function import train  
from core.function import validate  
from utils.utils import get_optimizer  
from utils.utils import save_checkpoint  
from utils.utils import create_logger  
from utils.utils import get_model_summary  

# Import our new minimal dataset class  
from dataset.minimal_coco import MinimalCOCODataset  

# Import Subset for debugging
from torch.utils.data import Subset



def parse_args():  
    parser = argparse.ArgumentParser(description='Train keypoints network')  
    # general  
    parser.add_argument('--cfg',  
                        help='experiment configure file name',  
                        required=True,  
                        type=str)  

    parser.add_argument('opts',  
                        help="Modify config options using the command-line",  
                        default=None,  
                        nargs=argparse.REMAINDER)  

    # philly  
    parser.add_argument('--modelDir',  
                        help='model directory',  
                        type=str,  
                        default='')  
    parser.add_argument('--logDir',  
                        help='log directory',  
                        type=str,  
                        default='')  
    parser.add_argument('--dataDir',  
                        help='data directory',  
                        type=str,  
                        default='')  
    parser.add_argument('--prevModelDir',  
                        help='prev Model directory',  
                        type=str,  
                        default='')  

    args = parser.parse_args()  
    return args  


def main():  
    args = parse_args()  
    update_config(cfg, args)  

    logger, final_output_dir, tb_log_dir = create_logger(  
        cfg, args.cfg, 'train')  

    logger.info(pprint.pformat(args))  
    logger.info(cfg)  

    # cudnn related setting  
    cudnn.benchmark = cfg.CUDNN.BENCHMARK  
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC  
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED  

    # Build model  
    model = __import__('models.' + cfg.MODEL.NAME, fromlist=['get_pose_net']).get_pose_net(  
        cfg, is_train=True  
    )  

    # Copy model file into output dir for reproducibility  
    this_dir = os.path.dirname(__file__)  
    shutil.copy2(  
        os.path.join(this_dir, '../lib/models', cfg.MODEL.NAME + '.py'),  
        final_output_dir)  

    writer_dict = {  
        'writer': SummaryWriter(log_dir=tb_log_dir),  
        'train_global_steps': 0,  
        'valid_global_steps': 0,  
    }  

    dump_input = torch.rand(  
        (1, 3, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0])  
    )  
    writer_dict['writer'].add_graph(model, (dump_input, ))  

    logger.info(get_model_summary(model, dump_input))  

    # Wrap model for multi-GPU, using whatever GPUs are available  
    device_ids = list(range(torch.cuda.device_count()))  
    model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()  

    # Define loss function and optimizer  
    criterion = JointsMSELoss(  
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT  
    ).cuda()  

    # Data loading code: use MinimalCOCODataset instead of COCODataset  
    normalize = transforms.Normalize(  
        mean=[0.485, 0.456, 0.406],  
        std=[0.229, 0.224, 0.225]  
    )  

    # Construct paths for training images and annotation JSON  
    train_images_dir = os.path.join(  
        cfg.DATASET.ROOT, 'images'  
    )  
    train_ann_file   = os.path.join(  
        cfg.DATASET.ROOT, 'annotations', f'{cfg.DATASET.TRAIN_SET}.json'  
    )  

    valid_images_dir = os.path.join(  
        cfg.DATASET.ROOT, 'images'  
    )  
    valid_ann_file   = os.path.join(  
        cfg.DATASET.ROOT, 'annotations', f'{cfg.DATASET.TEST_SET}.json'  
    )  

    # Sanity check: ensure those files/folders exist  
    if not os.path.isdir(train_images_dir):  
        raise FileNotFoundError(f"Training image folder not found: {train_images_dir}")  
    if not os.path.isfile(train_ann_file):  
        raise FileNotFoundError(f"Training annotation JSON not found: {train_ann_file}")  
    if not os.path.isdir(valid_images_dir):  
        raise FileNotFoundError(f"Validation image folder not found: {valid_images_dir}")  
    if not os.path.isfile(valid_ann_file):  
        raise FileNotFoundError(f"Validation annotation JSON not found: {valid_ann_file}")  

    # Instantiate our minimal COCO-based datasets  
    train_dataset = MinimalCOCODataset(  
        cfg,  
        root=train_images_dir,  
        ann_file=train_ann_file,  
        image_set=cfg.DATASET.TRAIN_SET,  
        is_train=True,  
        transform=transforms.Compose([  
            transforms.ToTensor(),  
            normalize,  
        ])  
    )  
    valid_dataset = MinimalCOCODataset(  
        cfg,  
        root=valid_images_dir,  
        ann_file=valid_ann_file,  
        image_set=cfg.DATASET.TEST_SET,  
        is_train=False,  
        transform=transforms.Compose([  
            transforms.ToTensor(),  
            normalize,  
        ])  
    )  
    N = len(train_dataset)
    train_dataset = Subset(train_dataset, range(N))

    train_loader = torch.utils.data.DataLoader(  
        train_dataset,  
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU * len(device_ids),  
        shuffle=cfg.TRAIN.SHUFFLE,  
        num_workers=cfg.WORKERS,  
        pin_memory=cfg.PIN_MEMORY
    )  

    valid_loader = torch.utils.data.DataLoader(  
        valid_dataset,  
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(device_ids),  
        shuffle=False,  
        num_workers=cfg.WORKERS,  
        pin_memory=cfg.PIN_MEMORY
    )  

    best_perf = 0.0  
    best_model = False  
    last_epoch = -1  
    optimizer = get_optimizer(cfg, model)  
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH  
    checkpoint_file = os.path.join(final_output_dir, 'checkpoint.pth')  

    # Auto-resume if a checkpoint exists  
    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):  
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))  
        checkpoint = torch.load(checkpoint_file, weights_only=False)  
        begin_epoch = checkpoint['epoch']  
        best_perf = checkpoint['perf']  
        last_epoch = checkpoint['epoch']  
        model.load_state_dict(checkpoint['state_dict'])  
        optimizer.load_state_dict(checkpoint['optimizer'])  
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(  
            checkpoint_file, checkpoint['epoch']))  

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(  
        optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,  
        last_epoch=last_epoch  
    )  

    # Main training loop  
    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):  
        lr_scheduler.step()  

        # train for one epoch  
        train(cfg, train_loader, model, criterion, optimizer, epoch,  
              final_output_dir, tb_log_dir, writer_dict)  
        # Determine correct dataset reference for validation
        if isinstance(valid_dataset, Subset):
            val_dataset_for_eval = valid_dataset.dataset
        else:
            val_dataset_for_eval = valid_dataset
        # evaluate on validation set  
        perf_indicator = validate(  
            cfg, valid_loader, val_dataset_for_eval, model, criterion,  
            final_output_dir, tb_log_dir, writer_dict  
        )  

        if perf_indicator >= best_perf:  
            best_perf = perf_indicator  
            best_model = True  
        else:  
            best_model = False  

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))  
        save_checkpoint({  
            'epoch': epoch + 1,  
            'model': cfg.MODEL.NAME,  
            'state_dict': model.state_dict(),  
            'best_state_dict': model.module.state_dict(),  
            'perf': perf_indicator,  
            'optimizer': optimizer.state_dict(),  
        }, best_model, final_output_dir)  

    final_model_state_file = os.path.join(  
        final_output_dir, 'final_state.pth'  
    )  
    logger.info('=> saving final model state to {}'.format(  
        final_model_state_file)  
    )  
    torch.save(model.module.state_dict(), final_model_state_file)  
    writer_dict['writer'].close()  


if __name__ == '__main__':  
    main()  
