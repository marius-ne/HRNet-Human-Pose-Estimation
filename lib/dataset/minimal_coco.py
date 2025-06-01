import copy
import random
import cv2
import torch
import numpy as np
import os
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from utils.transforms import get_affine_transform, affine_transform, fliplr_joints
from collections import defaultdict
from dataset.JointsDataset import JointsDataset
import json

class MinimalCOCODataset(JointsDataset):
    def __init__(self, cfg, root, ann_file, image_set, is_train, transform=None):
        super().__init__(cfg, root, image_set, is_train, transform)
        self.cfg = cfg
        self.root = root
        self.is_train = is_train
        self.transform = transform

        # Store parameters that __getitem__ needs
        self.num_joints = cfg.MODEL.NUM_JOINTS
        self.num_joints_half_body = cfg.DATASET.NUM_JOINTS_HALF_BODY
        self.prob_half_body = cfg.DATASET.PROB_HALF_BODY
        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP
        self.flip_pairs = cfg.DATASET.FLIP_PAIRS
        self.image_size = np.array(
            [cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1]], dtype=np.int32
        )
        self.color_rgb = cfg.DATASET.COLOR_RGB

        with open(ann_file, 'r') as f:
            raw_data = json.load(f)

        # raw_data is a dict with keys "images", "annotations", etc.
        annotations = raw_data['annotations']

        anns_per_image = defaultdict(list)
        for ann in annotations:
            img_id = int(ann['image_id'])
            anns_per_image[img_id].append(ann)

        # Now set image_ids to the keys that actually exist in the JSON
        self.image_ids = list(anns_per_image.keys())

        # Build a minimal “db” with exactly the fields __getitem__ expects
        self.db = []
        for img_id in self.image_ids:
            anns = anns_per_image[img_id]

            # Use first annotation per image (drop the rest)
            ann = None
            for a in anns:
                if a.get('num_keypoints', 0) > 0:
                    ann = a
                    break
            if ann is None:
                continue  # no keypoints -> skip this image

            # image_filepath is relative to self.root
            filename = os.path.basename(ann['image_filepath'])
            image_path = os.path.join(self.root, filename)

            # Keypoints array (reshape into K×3)
            kp = np.array(ann['keypoints'], dtype=np.float32).reshape(-1, 3)
            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float32)
            joints_3d_vis = np.zeros((self.num_joints, 3), dtype=np.float32)
            for j in range(self.num_joints):
                joints_3d[j, 0:2] = kp[j, 0:2]
                v = kp[j, 2]
                joints_3d_vis[j, 0] = v
                joints_3d_vis[j, 1] = v

            # Center & scale from bbox
            x, y, w, h = ann['bbox']
            center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)
            aspect = self.image_size[0] * 1.0 / self.image_size[1]
            if w > aspect * h:
                h = w * 1.0 / aspect
            elif w < aspect * h:
                w = h * aspect
            scale = np.array([w / 200.0, h / 200.0], dtype=np.float32) * 1.25

            self.db.append({
                'image': image_path,
                'filename': filename,
                'imgnum': img_id,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
                'center': center,
                'scale': scale,
                'score': 1.0
            })

    def __len__(self):
        return len(self.db)

    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])

        image_file = db_rec['image']
        filename = db_rec.get('filename', '')
        imgnum = db_rec.get('imgnum', '')

        # Read image (assuming no zip format)
        data_numpy = cv2.imread(image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if self.color_rgb:
            data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

        if data_numpy is None:
            raise ValueError(f'Fail to read {image_file}')

        joints = db_rec['joints_3d']
        joints_vis = db_rec['joints_3d_vis']

        c = db_rec['center']
        s = db_rec['scale']
        score = db_rec.get('score', 1)
        r = 0

        if self.is_train:
            if (np.sum(joints_vis[:, 0]) > self.num_joints_half_body
                    and np.random.rand() < self.prob_half_body):
                c_half, s_half = self.half_body_transform(joints, joints_vis)
                if c_half is not None and s_half is not None:
                    c, s = c_half, s_half

            # Scale and rotation augment
            sf = self.scale_factor
            rf = self.rotation_factor
            s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
            r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) if random.random() <= 0.6 else 0

            # Horizontal flip
            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]
                joints, joints_vis = fliplr_joints(joints, joints_vis, data_numpy.shape[1], self.flip_pairs)
                c[0] = data_numpy.shape[1] - c[0] - 1

        trans = get_affine_transform(c, s, r, self.image_size)
        input_img = cv2.warpAffine(
            data_numpy,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR
        )

        if self.transform:
            input_tensor = self.transform(input_img)  # a Tensor
        else:
            # Convert BGR→RGB if needed, then HWC→CHW, then to Tensor
            input_tensor = torch.from_numpy(input_img.transpose(2, 0, 1)).float().div(255.0)

        # Transform joint coordinates
        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)

        # Generate target heatmap and weight (same as original)
        target, target_weight = self.generate_target(joints, joints_vis)
        target = torch.from_numpy(target)
        target_weight = torch.from_numpy(target_weight)

        meta = {
            'image': image_file,
            'filename': filename,
            'imgnum': imgnum,
            'joints': joints,
            'joints_vis': joints_vis,
            'center': c,
            'scale': s,
            'rotation': r,
            'score': score
        }

        return input_tensor, target, target_weight, meta



    def evaluate(self, predictions, output_dir):
        """
        predictions: a list of dicts, each dict in COCO keypoint result format:
            {
              "image_id": int,
              "category_id": 1,
              "keypoints": [x1,y1,v1, x2,y2,v2, ..., x17,y17,v17],
              "score": float
            }
        output_dir: directory where to write the results JSON and evaluation logs.

        Writes predictions to COCO‐style JSON and runs COCOeval on keypoints.
        """
        os.makedirs(output_dir, exist_ok=True)
        res_file = os.path.join(output_dir, "keypoint_results.json")
        with open(res_file, 'w') as f:
            json.dump(predictions, f)

        coco_dt = self.coco.loadRes(res_file)
        coco_eval = COCOeval(self.coco, coco_dt, iouType='keypoints')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        # Returns a dict of the main statistics if needed
        stats_names = ['AP', 'Ap .5', 'Ap .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']
        return {name: coco_eval.stats[i] for i, name in enumerate(stats_names)}
