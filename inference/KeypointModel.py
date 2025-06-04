import os
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import training.tools._init_paths
from training.lib.config import cfg, update_config
from training.lib.utils.utils import get_model_summary
from training.lib.dataset.minimal_coco import MinimalCOCODataset
from training.lib.core.inference import get_final_preds  # HRNet’s heatmap → keypoint decoder


class KeypointEvaluator:
    def __init__(self, cfg_file, checkpoint_file, labels_file, opts=None):
        """
        - cfg_file: path to your YAML config (e.g. "experiments/hrnet_w32.yaml")
        - checkpoint_file: .pth file with model weights
        - labels_file: COCO‐format JSON (e.g. "annotations/val2017.json")
        - opts: optional list of “KEY VALUE” overrides for cfg (same as argparse’s opts)
        """
        # 1) Load config
        if opts is None:
            opts = []
        update_config(cfg, [cfg_file] + opts)

        # 2) Build model and load weights
        cudnn.benchmark = cfg.CUDNN.BENCHMARK
        torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
        torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

        model = __import__('models.' + cfg.MODEL.NAME, fromlist=['get_pose_net']).get_pose_net(
            cfg, is_train=False
        )
        device_ids = list(range(torch.cuda.device_count()))
        self.model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()

        ckpt = torch.load(checkpoint_file, map_location='cpu')
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # 3) Prepare dataset (so we can compute center/scale & transforms)
        data_root = os.path.dirname(os.path.dirname(labels_file))
        images_dir = os.path.join(data_root, 'images')
        image_set_name = os.path.splitext(os.path.basename(labels_file))[0]

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        self.dataset = MinimalCOCODataset(
            cfg,
            root=images_dir,
            ann_file=labels_file,
            image_set=image_set_name,
            is_train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        )

        # Build a lookup: image_id → index in self.dataset.db
        self.id_to_index = {
            rec['imgnum']: idx
            for idx, rec in enumerate(self.dataset.db)
        }

    def eval(self, image_id=None, filename=None):
        """
        Returns a list of [x, y, confidence] for each keypoint in the given image.
        You must supply either image_id (int) or filename (str).
          - image_id: the integer COCO image ID
          - filename: the basename string (e.g. "000123.jpg")
        """
        # 1) Find the dataset index
        if image_id is not None:
            if image_id not in self.id_to_index:
                raise KeyError(f"Image ID {image_id} not found in labels.")
            idx = self.id_to_index[image_id]
        elif filename is not None:
            # search for matching 'filename' in dataset.db
            idx = next(
                (i for i, rec in enumerate(self.dataset.db) if rec['filename'] == filename),
                None
            )
            if idx is None:
                raise KeyError(f"Filename '{filename}' not found in labels.")
        else:
            raise ValueError("Call eval() with either image_id or filename.")

        # 2) Load preprocessed input & metadata from dataset
        input_tensor, _, _, meta = self.dataset[idx]
        # meta contains: 'center' [2], 'scale' [2], 'imgnum' (image_id), etc.

        # 3) Run model forward
        with torch.no_grad():
            x = input_tensor.unsqueeze(0).cuda()  # [1, 3, H, W]
            outputs = self.model(x)
            heatmaps = outputs[-1] if isinstance(outputs, (list, tuple)) else outputs  # [1, J, Hh, Wh]

            hm_numpy = heatmaps.cpu().numpy()
            center = meta['center'].reshape(1, 2)
            scale = meta['scale'].reshape(1, 2)

            # 4) Decode heatmaps → (x, y, confidence)
            preds, maxvals = get_final_preds(cfg, hm_numpy, center, scale)
            # preds: shape [1, num_joints, 2]; maxvals: [1, num_joints, 1]

            num_joints = preds.shape[1]
            results = []
            for j in range(num_joints):
                x_j = float(preds[0, j, 0])
                y_j = float(preds[0, j, 1])
                c_j = float(maxvals[0, j, 0])
                results.append([x_j, y_j, c_j])

        return results

