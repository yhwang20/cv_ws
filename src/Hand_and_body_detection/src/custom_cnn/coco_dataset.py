import os
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import cv2
import torch
import numpy as np

def generate_heatmap(h, w, center, sigma=2):
    """Generate a Gaussian heatmap for a single keypoint."""
    heatmap = np.zeros((h, w), dtype=np.float32)
    if center[0] < 0 or center[1] < 0:
        return heatmap
    x = np.arange(w)
    y = np.arange(h)
    xx, yy = np.meshgrid(x, y)
    heatmap = np.exp(-((xx - center[0])**2 + (yy - center[1])**2) / (2 * sigma**2))
    return heatmap

class COCOKP(Dataset):
    def __init__(self, img_dir, ann_file, input_size=256, heatmap_size=64):
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.input_size = input_size
        self.heatmap_size = heatmap_size

        # Filter only images that exist on disk
        self.ids = [img_id for img_id in self.coco.imgs.keys()
                    if os.path.exists(os.path.join(self.img_dir, self.coco.imgs[img_id]['file_name']))]

        # Optional: filter images that have at least one keypoint
        valid_ids = []
        for img_id in self.ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
            anns = self.coco.loadAnns(ann_ids)
            if any('keypoints' in ann and any(ann['keypoints']) for ann in anns):
                valid_ids.append(img_id)
        self.ids = valid_ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])

        img = cv2.imread(img_path)
        if img is None:
            # If image cannot be read, return a blank image and zero heatmap
            img = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
            heatmaps = np.zeros((17, self.heatmap_size, self.heatmap_size), dtype=np.float32)
            img_tensor = torch.tensor(img).permute(2,0,1).float() / 255.0
            return img_tensor, torch.tensor(heatmaps)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_size, self.input_size))
        img_tensor = torch.tensor(img).permute(2,0,1).float() / 255.0

        # Load first annotation (first person) for heatmap generation
        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)
        heatmaps = np.zeros((17, self.heatmap_size, self.heatmap_size), dtype=np.float32)
        for ann in anns:
            if 'keypoints' not in ann:
                continue
            kp = np.array(ann['keypoints']).reshape(-1,3)
            for i in range(17):
                if kp[i,2] > 0:  # visible
                    x = kp[i,0] * self.heatmap_size / self.input_size
                    y = kp[i,1] * self.heatmap_size / self.input_size
                    heatmaps[i] = generate_heatmap(self.heatmap_size, self.heatmap_size, (x,y))
            break  # Only take first person

        return img_tensor, torch.tensor(heatmaps, dtype=torch.float32)
