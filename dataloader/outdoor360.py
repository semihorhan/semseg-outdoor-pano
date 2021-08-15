import os
import glob
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset


class Outdoor360(Dataset):
    """Outdoor Panoramic Dataloader"""
    def __init__(self, args):
        super().__init__()
        self._base_dir = args.dataset_path
        self._image_dir = os.path.join(self._base_dir, 'test', 'rgb')
        self._mask_dir = os.path.join(self._base_dir, 'test', 'mask')
        self.args = args
        self.images = []
        self.masks = []
        self.images = glob.glob(self._image_dir+"/*.png")
        self.masks = glob.glob(self._mask_dir+"/*.png")

        assert len(self.images) == len(self.masks)
        img_check = self.images[-1].split("/")[-1]
        mask_check = self.masks[-1].split("/")[-1]
        assert img_check == mask_check

        print('Number of images in {}: {:d}'.format('test set', len(self.images)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        img = self.transform_image(img)
        mask = Image.open(self.masks[index])
        mask = self.tranform_mask(mask)
        sample = {'image': img, 'mask': mask}
        return sample

    def transform_image(self, image):
        img = image.resize((self.args.img_witdh, self.args.img_height), Image.BILINEAR)
        img = self.normalize(img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        img = torch.from_numpy(img).float()
        return img

    def tranform_mask(self, mask):
        mask = mask.resize((self.args.img_witdh, self.args.img_height), Image.NEAREST)
        mask = np.array(mask).astype(np.int32)
        mask = torch.from_numpy(mask).int()
        return mask

    def normalize(self, image, mean=None, std=None):
        img = image
        img = np.array(img).astype(np.float32)
        img /= 255.0
        img -= mean
        img /= std
        return img
