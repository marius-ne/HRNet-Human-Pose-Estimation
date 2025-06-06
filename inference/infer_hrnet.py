# repo/training/inference/infer_hrnet.py

import cv2
import torch
import numpy as np
from argparse import Namespace

from training.lib.config import cfg, update_config
from training.lib.core.inference import get_final_preds
from training.lib.utils.transforms import get_affine_transform
import torchvision.transforms as transforms


def setup_model(cfg_file: str, weights_file: str, device: str = "cuda") -> torch.nn.Module:
    """
    Load the HRNet (or whichever get_pose_net cfg specifies) and its weights.
    Returns a model in eval mode on the requested device.
    """
    args = Namespace(
        cfg=cfg_file,
        opts=[],
        modelDir="",
        logDir="",
        dataDir="",
        prevModelDir=""
    )
    update_config(cfg, args)

    get_pose_net = __import__(
        f"training.lib.models.{cfg.MODEL.NAME}", fromlist=["get_pose_net"]
    ).get_pose_net
    model = get_pose_net(cfg, is_train=False)

    checkpoint = torch.load(weights_file, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model


def run_inference(
    model: torch.nn.Module,
    cfg=cfg,
    image: np.ndarray | str = None,
    device: str = "cuda",
    bbox: list[float] = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run a forward pass on one image, given a ground‐truth or detected bbox,
    and return:
      - preds:         shape (K, 2) in original-image coords
      - confidences:   shape (K, 1)
      - orig_img:      the original BGR image (H×W×3)
      - orig_heatmaps: a float32 array of shape (K, H, W), each joint's heatmap in original size

    Args:
        model:   The loaded pose‐estimation network (in eval mode).
        cfg:     The same config object that was loaded in setup_model.
        image:   Either a filepath (str) or a BGR np.ndarray (H×W×3).
        device:  "cuda" or "cpu".
        bbox:    [x, y, w, h] bounding box in the original image (must be provided).

    Returns:
        preds         (np.ndarray[K,2])
        confidences   (np.ndarray[K,1])
        orig_img      (np.ndarray[H,W,3], dtype=uint8, BGR)
        orig_heatmaps (np.ndarray[K,H,W], dtype=float32)
    """
    if bbox is None:
        raise ValueError("You must supply a bbox=[x,y,w,h] when calling run_inference.")

    # 1) Load the original image if a filepath is given
    if isinstance(image, str):
        orig_img = cv2.imread(image, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if orig_img is None:
            raise FileNotFoundError(f"Could not read image: {image}")
    else:
        orig_img = image.copy()
    orig_h, orig_w = orig_img.shape[:2]

    # 2) Compute the same center & scale that MinimalCOCODataset used during training
    x, y, w, h = bbox
    center = np.array([x + 0.5 * w, y + 0.5 * h], dtype=np.float32)

    # Keep the aspect ratio of the network’s input
    aspect = cfg.MODEL.IMAGE_SIZE[0] / cfg.MODEL.IMAGE_SIZE[1]  # e.g. 288/384
    if w > aspect * h:
        h = w / aspect
    elif w < aspect * h:
        w = h * aspect

    scale = np.array([w / 200.0, h / 200.0], dtype=np.float32) * 1.25

    # 3) Compute the affine transform from object‐crop → network input
    inp_w, inp_h = cfg.MODEL.IMAGE_SIZE  # (width, height) = (288, 384)
    trans = get_affine_transform(center, scale, 0, np.array([inp_w, inp_h], dtype=np.int32))

    # 4) Warp just the object‐centric crop (288×384) from the original image
    inp_img = cv2.warpAffine(
        orig_img,
        trans,
        (int(inp_w), int(inp_h)),
        flags=cv2.INTER_LINEAR,
    )

    # 5) Convert BGR→RGB if needed, then ToTensor + Normalize
    if cfg.DATASET.COLOR_RGB:
        inp_rgb = cv2.cvtColor(inp_img, cv2.COLOR_BGR2RGB)
    else:
        inp_rgb = inp_img.copy()

    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    inp_tensor = normalize(to_tensor(inp_rgb)).unsqueeze(0).to(device)  # shape (1,3, H, W)

    # 6) Forward pass
    with torch.no_grad():
        output = model(inp_tensor)
        if isinstance(output, (list, tuple)):
            output = output[-1]
        heatmaps = output.cpu().numpy()[0]  # shape: (K, H_out, W_out)

    # 7) Decode peaks → coordinates in cropped 288×384, then map back to full image
    centers_np = np.array([center], dtype=np.float32).reshape((1, 2))
    scales_np = np.array([scale], dtype=np.float32).reshape((1, 2))
    preds_all, maxvals_all = get_final_preds(cfg, heatmaps[None, ...], centers_np, scales_np)
    preds = preds_all[0]        # (K,2) in full‐image coords
    confidences = maxvals_all[0]  # (K,1)

    # 8) Build per‐joint heatmaps in the original image size
    num_joints, h_out, w_out = heatmaps.shape
    orig_heatmaps = np.zeros((num_joints, orig_h, orig_w), dtype=np.float32)
    inv_trans = cv2.invertAffineTransform(trans)
    for j in range(num_joints):
        hmap = heatmaps[j]  # (H_out, W_out)
        # Upsample to network‐input size (288×384):
        hmap_resized = cv2.resize(hmap, (inp_w, inp_h), interpolation=cv2.INTER_CUBIC)
        # Warp back to original image coordinates:
        hmap_orig = cv2.warpAffine(
            hmap_resized,
            inv_trans,
            (orig_w, orig_h),
            flags=cv2.INTER_LINEAR,
        )
        orig_heatmaps[j] = hmap_orig

    return preds, confidences, orig_img, orig_heatmaps
