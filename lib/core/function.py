# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import time
import logging
import os

import numpy as np
import torch

from core.evaluate import accuracy
from core.inference import get_final_preds
from utils.transforms import flip_back
from utils.vis import save_debug_images


logger = logging.getLogger(__name__)


def train(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        outputs = model(input)

        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)

        if isinstance(outputs, list):
            loss = criterion(outputs[0], target, target_weight)
            for output in outputs[1:]:
                loss += criterion(output, target, target_weight)
        else:
            output = outputs
            loss = criterion(output, target, target_weight)

        # loss = criterion(output, target, target_weight)

        # ----- RIGID BODY LOSS -----
        # preds_np, _ = get_final_preds(
        #     config, 
        #     output.detach().cpu().numpy(), 
        #     meta['center'].numpy(), 
        #     meta['scale'].numpy()
        # )  # shape: (N, K, 2)
        # preds = torch.from_numpy(preds_np).to(output.device)        # (N, K, 2)

        # # Extract ground‐truth (x,y) from meta['joints'] (shape: [N, K, 3])
        # # THESE ARE RESIZED!
        # gt_coords = meta['joints'][:, :, :2].to(output.device)      # (N, K, 2)

        # # Compute rigid‐geometry loss over each corner‐pair
        # loss_geom = 0.0
        # rigid_pairs = config.DATASET.RIGID_PAIRS # edges to keep at a constant distance
        # for (i, j) in rigid_pairs:
        #     d_pred = torch.norm(preds[:, i] - preds[:, j], dim=1)   # (N,)
        #     d_gt   = torch.norm(gt_coords[:, i] - gt_coords[:, j], dim=1)
        #     loss_geom += ((d_pred - d_gt)**2).mean()
        
        # # Hyperparameter lambda_geom for rigid body keypoints
        # lambda_geom = 0.01

        # # Combine with the original heatmap MSE
        # loss = loss + lambda_geom * loss_geom

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, input, meta, target, pred*4, output,
                              prefix)


def validate(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses     = AverageMeter()
    acc        = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)  # total size of underlying dataset
    all_preds   = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes   = np.zeros((num_samples, 6))
    image_path  = []
    all_image_ids = []
    filenames   = []
    imgnums     = []
    idx         = 0

    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, meta) in enumerate(val_loader):
            # -------- forward pass --------
            outputs = model(input)
            if isinstance(outputs, list):
                output = outputs[-1]
            else:
                output = outputs

            # -------- flip‐test (only if enabled) --------
            if config.TEST.FLIP_TEST:
                # flip input, run model, unflip & average
                input_flipped = np.flip(input.cpu().numpy(), axis=3).copy()
                input_flipped = torch.from_numpy(input_flipped).cuda()
                outputs_flipped = model(input_flipped)

                if isinstance(outputs_flipped, list):
                    output_flipped = outputs_flipped[-1]
                else:
                    output_flipped = outputs_flipped

                output_flipped = flip_back(
                    output_flipped.cpu().numpy(),
                    val_dataset.flip_pairs
                )
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = output_flipped.clone()[:, :, :, 0:-1]

                output = (output + output_flipped) * 0.5

            # -------- compute loss & accuracy --------
            target        = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)
            loss = criterion(output, target, target_weight)

            num_images = input.size(0)
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(
                output.cpu().numpy(),
                target.cpu().numpy()
            )
            acc.update(avg_acc, cnt)

            # -------- timing --------
            batch_time.update(time.time() - end)
            end = time.time()

            # -------- get final keypoints --------
            c     = meta['center'].numpy()    # shape: (batch_size, 2)
            s     = meta['scale'].numpy()     # shape: (batch_size, 2)
            score = meta['score'].numpy()     # shape: (batch_size,)

            preds, maxvals = get_final_preds(
                config,
                output.clone().cpu().numpy(),
                c, s
            )

            # -------- write into all_preds / all_boxes --------
            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals

            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4]    = np.prod(s * 200, axis=1)
            all_boxes[idx:idx + num_images, 5]    = score

            image_path.extend(meta['image'])

            # -------- collect image IDs (one per sample) --------
            if 'imgnum' in meta:
                batch_ids = meta['imgnum'].tolist()     # e.g. [5708, 10331, …]
            else:
                batch_ids = meta['image_id'].tolist()

            # each element in batch_ids is already a Python number or convertible to int
            all_image_ids.extend([int(x) for x in batch_ids])

            # advance idx by how many images this batch had
            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, acc=acc)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                save_debug_images(config, input, meta, target, pred * 4, output,
                                  prefix)

        # ─── After the loop: slice to exactly how many samples we saw ───
        all_preds = all_preds[:idx]
        all_boxes = all_boxes[:idx]

        # (Optional debug)
        # print("Processed samples (idx):", idx)
        # print("len(all_image_ids):", len(all_image_ids))
        # assert idx == len(all_image_ids), "Still a mismatch: {} vs {}".format(idx, len(all_image_ids))

        # ─── build COCO‐format results only up to `idx` ───
        coco_preds = []
        for i in range(idx):
            keypoints = all_preds[i].reshape(-1).tolist()
            image_id  = int(all_image_ids[i])
            coco_preds.append({
                "image_id":  image_id,
                "category_id": 1,
                "keypoints":  keypoints,
                "score":      float(all_boxes[i, 5]) if all_boxes is not None else 1.0
            })

        # ─── finally run the dataset's evaluate() ───
        name_values, perf_indicator = val_dataset.evaluate(
            config, coco_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer       = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar('valid_loss', losses.avg, global_steps)
            writer.add_scalar('valid_acc',  acc.avg,   global_steps)
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars('valid', dict(name_value), global_steps)
            else:
                writer.add_scalars('valid', dict(name_values), global_steps)

            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator



# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
