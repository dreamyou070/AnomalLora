import os
import numpy as np
from torch.utils.data import Dataset
import torch
import cv2
import glob
from data.perlin import rand_perlin_2d_np
from PIL import Image
from torchvision import transforms
from numpy.random import default_rng
import cv2
import skimage.exposure

class MVTecDRAEMTestDataset(Dataset):

    def __init__(self, root_dir, resize_shape=None):
        self.root_dir = root_dir
        self.images = sorted(glob.glob(root_dir+"/*/*.png"))
        self.resize_shape=resize_shape

    def __len__(self):
        return len(self.images)

    def transform_image(self, image_path, mask_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((image.shape[0],image.shape[1]))
        if self.resize_shape != None:
            image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
            mask = cv2.resize(mask, dsize=(self.resize_shape[1], self.resize_shape[0]))

        image = image / 255.0
        mask = mask / 255.0

        image = np.array(image).reshape((image.shape[0], image.shape[1], 3)).astype(np.float32)
        mask = np.array(mask).reshape((mask.shape[0], mask.shape[1], 1)).astype(np.float32)

        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        return image, mask

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images[idx]
        dir_path, file_name = os.path.split(img_path)
        base_dir = os.path.basename(dir_path)
        if base_dir == 'good':
            image, mask = self.transform_image(img_path, None)
            has_anomaly = np.array([0], dtype=np.float32)
        else:
            mask_path = os.path.join(dir_path, '../../ground_truth/')
            mask_path = os.path.join(mask_path, base_dir)
            mask_file_name = file_name.split(".")[0]+"_mask.png"
            mask_path = os.path.join(mask_path, mask_file_name)
            image, mask = self.transform_image(img_path, mask_path)
            has_anomaly = np.array([1], dtype=np.float32)

        sample = {'image': image, 'has_anomaly': has_anomaly,'mask': mask, 'idx': idx}

        return sample


class MVTecDRAEMTrainDataset(Dataset):

    def __init__(self,
                 root_dir,
                 anomaly_source_path,
                 resize_shape=None,
                 tokenizer=None,
                 caption: str = None,
                 use_perlin: bool = False):

        self.root_dir = root_dir
        self.resize_shape=resize_shape
        self.image_paths = sorted(glob.glob(root_dir+"/*.png"))
        self.anomaly_source_paths = sorted(glob.glob(anomaly_source_path+"/*/*.png"))
        self.caption = caption
        self.tokenizer = tokenizer
        self.transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5], [0.5]),])
        self.use_perlin = use_perlin

    def __len__(self):
        return len(self.image_paths)
    def get_input_ids(self, caption):
        tokenizer_output = self.tokenizer(caption, padding="max_length", truncation=True,return_tensors="pt")
        input_ids = tokenizer_output.input_ids
        attention_mask = tokenizer_output.attention_mask
        return input_ids, attention_mask
    def make_random_mask(self, height, width) -> np.ndarray :

        if self.use_perlin:
            perlin_scale = 6
            min_perlin_scale = 0
            perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
            perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
            noise = rand_perlin_2d_np((height, width), (perlin_scalex, perlin_scaley))
        else:
            random_generator = default_rng(seed=42)
            noise = random_generator.integers(0, 255, (height, width), np.uint8, True)
        blur = cv2.GaussianBlur(noise, (0, 0), sigmaX=15, sigmaY=15, borderType=cv2.BORDER_DEFAULT)
        stretch = skimage.exposure.rescale_intensity(blur, in_range='image', out_range=(0, 255)).astype(np.uint8)
        thresh = cv2.threshold(stretch, 175, 255, cv2.THRESH_BINARY)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask_pil = Image.fromarray(mask).convert('L')
        mask_np = np.array(mask_pil) / 255  # height, width, [0,1]
        return mask_np, mask_pil

    def load_image(self, image_path, trg_h, trg_w):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        if trg_h and trg_w:
            image = image.resize((trg_w, trg_h), Image.BICUBIC)
        img = np.array(image, np.uint8)
        return img

    def __getitem__(self, idx):

        idx = torch.randint(0, len(self.image_paths), (1,)).item()
        anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()

        # [1] base
        img = self.load_image(self.image_paths[idx], self.resize_shape[0], self.resize_shape[1])
        dtype = img.dtype
        anomal_src = self.load_image(self.anomaly_source_paths[anomaly_source_idx], self.resize_shape[0], self.resize_shape[1])

        # [2] augment ( anomaly mask white = anomal position )
        anomal_mask_np, anomal_mask_pil = self.make_random_mask(self.resize_shape[0], self.resize_shape[1]) # [512, 512], [0, 1]
        anomal_mask_np = np.where(anomal_mask_np == 0, 0, 1)  # strict anomal (0, 1
        mask = np.repeat(np.expand_dims(anomal_mask_np, axis=2), 3, axis=2).astype(dtype)
        anomal_img = (1-mask) * img + mask * anomal_src

        # [3] final
        image = self.transform(img)
        anomal_image = self.transform(anomal_img)

        # -----------------------------------------------------------------------------------------------
        anomal_img_pil = Image.fromarray((mask * 255).astype(np.uint8))
        anomal_pil = anomal_img_pil.resize((64,64)).convert('L')
        anomal_torch = torch.tensor(np.array(anomal_pil))
        anomal_mask = torch.where(anomal_torch > 0.5, 1, 0)  # strict anomal

        # [4] caption
        input_ids, attention_mask = self.get_input_ids(self.caption) # input_ids = [77]

        # [5] return
        sample = {'image': image,
                  "anomaly_mask": anomal_mask,
                  'augmented_image': anomal_image,
                  'idx': idx,
                  'input_ids': input_ids.squeeze(0),
                  'caption': self.caption,}
        return sample
