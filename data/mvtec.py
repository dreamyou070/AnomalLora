import os
import numpy as np
from torch.utils.data import Dataset
import torch
import cv2
import glob
import imgaug.augmenters as iaa
from data.perlin import rand_perlin_2d_np
from PIL import Image
from torchvision import transforms

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
                 caption: str = None,):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        print(f'root_dir: {root_dir}')
        self.resize_shape=resize_shape

        self.image_paths = sorted(glob.glob(root_dir+"/*.png"))
        print(f'len(self.image_paths): {len(self.image_paths)}')

        self.anomaly_source_paths = sorted(glob.glob(anomaly_source_path+"/*/*.png"))
        self.augmenters = [iaa.GammaContrast((0.5,2.0),per_channel=True),
                           iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30)),
                           iaa.pillike.EnhanceSharpness(),
                           iaa.AddToHueAndSaturation((-50,50),per_channel=True),
                           iaa.Solarize(0.5, threshold=(32,128)),
                           iaa.Posterize(),
                           iaa.Invert(),
                           iaa.pillike.Autocontrast(),
                           iaa.pillike.Equalize(),
                           iaa.Affine(rotate=(-45, 45))]
        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])

        self.caption = caption
        self.tokenizer = tokenizer
        self.transform = transforms.Compose([transforms.ToTensor(),
                                                                transforms.Normalize([0.5], [0.5]),])

    def __len__(self):
        return len(self.image_paths)

    def get_input_ids(self, caption):
        tokenizer_output = self.tokenizer(caption, padding="max_length", truncation=True,return_tensors="pt")
        input_ids = tokenizer_output.input_ids
        attention_mask = tokenizer_output.attention_mask
        return input_ids, attention_mask

    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]])
        return aug

    def augment_image(self, image, anomaly_source):

        # [1] get anomal mask
        perlin_scale = 6
        min_perlin_scale = 0
        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_noise = rand_perlin_2d_np((self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
        #perlin_noise = self.rot(image=perlin_noise)
        threshold = 0.5
        perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
        perlin_thr = np.expand_dims(perlin_thr, axis=2)

        # [2] synthetic image
        beta = torch.rand(1).numpy()[0] * 0.8
        augmented_image = (1 - perlin_thr) * image + \
                          (perlin_thr) * (beta * image + (1 - beta) * anomaly_source)
        return augmented_image, perlin_thr

    def load_image(self, image_path, trg_h, trg_w):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        if trg_h and trg_w:
            image = image.resize((trg_w, trg_h), Image.BICUBIC)
        img = np.array(image, np.uint8)
        return img
    def transform_image(self, image_path, anomaly_source_path):

        # ------------------------------------------------------------------------------------------------------------
        # [1] Read the image and apply general augmentation
        img = self.load_image(image_path, self.resize_shape[0], self.resize_shape[1])
        anomal_img = self.load_image(anomaly_source_path, self.resize_shape[0], self.resize_shape[1])
        augmented_image, anomaly_mask = self.augment_image(img, anomal_img)
        return img, augmented_image, anomaly_mask

    def __getitem__(self, idx):

        idx = torch.randint(0, len(self.image_paths), (1,)).item()
        anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
        image, augmented_image, anomaly_mask = self.transform_image(self.image_paths[idx],
                                                                    self.anomaly_source_paths[anomaly_source_idx])
        image = self.transform(image)
        augmented_image = self.transform(augmented_image)
        anomal_pil = Image.fromarray((np.squeeze(anomaly_mask, axis=2) * 255).astype(np.uint8)).resize((64, 64))
        anomal_torch = torch.tensor(np.array(anomal_pil))
        anomal_mask = torch.where(anomal_torch == 0, 1, 0) # strict anomal

        # -------------------------------------------------------------------------------------------------------------------
        # [2] caption
        input_ids, attention_mask = self.get_input_ids(self.caption) # input_ids = [77]
        sample = {'image': image,
                  "anomaly_mask": anomal_mask,
                  'augmented_image': augmented_image,
                  'idx': idx,
                  'input_ids': input_ids.squeeze(0),
                  'caption': self.caption,}
        return sample

