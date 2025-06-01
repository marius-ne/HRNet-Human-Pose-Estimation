import os
import json
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
from PIL import Image

class MinimalCOCODataset(Dataset):
    """
    A minimal PyTorch Dataset for COCO keypoint training and evaluation.
    Only loads image paths and raw keypoint annotations. Provides a basic
    `evaluate` method that writes predictions in COCO format and runs COCOeval.
    """
    def __init__(self, root, ann_file, is_train=True, transform=None):
        """
        Args:
            root (str): Directory where images are stored (COCO image folder).
            ann_file (str): Path to the COCO annotation JSON (e.g., person_keypoints_train2017.json).
            is_train (bool): If True, loads training IDs; otherwise loads validation/test IDs.
            transform (callable, optional): A function/transform that takes in a dict
                with keys {'image': PIL.Image, 'keypoints': np.ndarray} and returns a transformed dict.
        """
        super().__init__()
        self.root = root
        self.coco = COCO(ann_file)
        # Use all image IDs (for train or val, annotation file should correspond)
        self.ids = list(self.coco.getImgIds())
        self.transform = transform
        self.is_train = is_train

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        """
        Returns a dict with:
            'image': PIL.Image loaded from disk,
            'keypoints': (K, 3) array where K = # keypoints in that image,
                         each keypoint is (x, y, v) with v ∈ {0,1,2} as in COCO.
        """
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root, img_info['file_name'])
        image = Image.open(img_path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)

        # Collect all keypoints for this image into an (N_instances, K, 3) array
        keypoints = []
        for ann in anns:
            if 'keypoints' in ann and np.sum(ann['keypoints']) > 0:
                kpt = np.array(ann['keypoints'], dtype=np.float32).reshape(-1, 3)
                keypoints.append(kpt)
        if len(keypoints) > 0:
            keypoints = np.stack(keypoints, axis=0)  # shape: (num_instances, num_joints, 3)
        else:
            keypoints = np.zeros((0, 17, 3), dtype=np.float32)

        sample = {'image': image, 'keypoints': keypoints, 'img_id': img_id}

        if self.transform:
            sample = self.transform(sample)

        return sample

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
