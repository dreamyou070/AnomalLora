import os
import numpy as np
from torch.utils.data import Dataset
import torch
import cv2
import random
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
                 use_perlin: bool = False,
                 num_repeat: int = 1,
                 anomal_only_on_object : bool = True,
                 anomal_training : bool = False,
                 latent_res : int = 64,
                 perlin_max_scale : int = 8,
                 kernel_size : int = 5):

        self.root_dir = root_dir
        self.resize_shape=resize_shape

        if anomaly_source_path is not None:
            self.anomaly_source_paths = []
            for ext in ["png", "jpg"]:
                self.anomaly_source_paths.extend(sorted(glob.glob(anomaly_source_path + f"/*/*/*.{ext}")))
        else :
            self.anomaly_source_paths = []

        self.caption = caption
        self.tokenizer = tokenizer
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize([0.5], [0.5]),])
        self.use_perlin = use_perlin
        self.num_repeat = num_repeat

        image_paths = sorted(glob.glob(root_dir + "/*.png"))
        self.image_paths = [image_path for image_path in image_paths for i in range(num_repeat)]
        random.shuffle(self.image_paths)
        self.anomal_only_on_object = anomal_only_on_object
        self.anomal_training = anomal_training
        self.latent_res = latent_res
        self.perlin_max_scale = perlin_max_scale
        self.kernel_size = kernel_size

    def __len__(self):
        if len(self.anomaly_source_paths) > 0 :
            return max(len(self.image_paths), len(self.anomaly_source_paths))
        else:
            return len(self.image_paths)


    def get_input_ids(self, caption):
        tokenizer_output = self.tokenizer(caption, padding="max_length", truncation=True,return_tensors="pt")
        input_ids = tokenizer_output.input_ids
        attention_mask = tokenizer_output.attention_mask
        return input_ids, attention_mask

    def augment_image(self, image, anomaly_source_path):

        # [1] org image
        anomaly_source_img = cv2.imread(anomaly_source_path)
        anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(self.resize_shape[0], self.resize_shape[1]))
        # [2] perlin noise
        perlin_scale = 6
        min_perlin_scale = 0
        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_noise = rand_perlin_2d_np((512, 512), (perlin_scalex, perlin_scaley))
        threshold = 0.5
        perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
        perlin_thr = np.expand_dims(perlin_thr, axis=2)  # [512,512,3]
        # [3] only 1 = img
        img_thr = anomaly_source_img.astype(np.float32) * perlin_thr / 255.0
        # [4] beta
        beta = torch.rand(1).numpy()[0] * 0.8  # small number
        augmented_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * (perlin_thr)  # [512,512,3]
        augmented_image = augmented_image.astype(np.float32)
        mask = (perlin_thr).astype(np.float32) # [512,512,3]
        mask = np.squeeze(mask, axis=2)
        # augmented_image = msk * augmented_image + (1 - msk) * image
        return augmented_image, mask



    def load_image(self, image_path, trg_h, trg_w,
                   type='RGB'):
        image = Image.open(image_path)
        if type == 'RGB' :
            if not image.mode == "RGB":
                image = image.convert("RGB")
        elif type == 'L':
            if not image.mode == "L":
                image = image.convert("L")
        if trg_h and trg_w:
            image = image.resize((trg_w, trg_h), Image.BICUBIC)
        img = np.array(image, np.uint8)
        return img

    def get_img_name(self, img_path):
        parent, name = os.path.split(img_path)
        class_name = os.path.split(parent)[1]
        class_name = os.path.split(class_name)[1]
        name, ext = os.path.splitext(name)
        final_name = f'{class_name}_{name}'
        return final_name

    def get_object_mask_dir(self, img_path):
        parent, name = os.path.split(img_path)
        parent, _ = os.path.split(parent)
        object_mask_dir = os.path.join(parent, f"object_mask/{name}")
        return object_mask_dir

    def make_random_gaussian_mask(self):

        end_num = self.resize_shape[0]

        # [1] back
        x = np.arange(0, end_num, 1, float)
        y = np.arange(0, end_num, 1, float)[:, np.newaxis]

        # [2] center
        x_0 = torch.randint(int(end_num / 4), int(3 * end_num / 4), (1,)).item()
        y_0 = torch.randint(int(end_num / 4), int(3 * end_num / 4), (1,)).item()

        # [3] sigma
        sigma = torch.randint(20, 50, (1,)).item()

        # [4] make kernel
        result = np.exp(-4 * np.log(2) * ((x - x_0) ** 2 + (y - y_0) ** 2) / sigma ** 2)  # 0 ~ 1
        return result


    def __getitem__(self, idx):

        # [1] base
        img_idx = idx % len(self.image_paths)
        img_path = self.image_paths[img_idx]
        img = self.load_image(img_path, self.resize_shape[0], self.resize_shape[1]) # np.array,
        dtype = img.dtype
        final_name = self.get_img_name(img_path)

        # [2] background
        parent, name = os.path.split(img_path)
        parent, _ = os.path.split(parent)
        background_dir = os.path.join(parent, f"background/{name}")
        background_img = self.load_image(background_dir, self.resize_shape[0], self.resize_shape[1])

        # [3] object mask
        object_mask_dir = self.get_object_mask_dir(img_path)
        object_img = self.load_image(object_mask_dir, self.latent_res, self.latent_res, type='L')
        object_mask_np = np.where((np.array(object_img, np.uint8) / 255) == 0, 0, 1)
        object_mask = torch.tensor(object_mask_np) # shape = [64,64], 0 = background, 1 = object

        if len(self.anomaly_source_paths) > 0:
            anomal_src_idx = idx % len(self.anomaly_source_paths)
            if not self.anomal_only_on_object:
                augmented_image, anomal_mask_np = self.augment_image(img, self.anomaly_source_paths[anomal_src_idx])  # [512,512,3]

            if self.anomal_only_on_object:
                object_img_aug = self.load_image(object_mask_dir, self.resize_shape[0], self.resize_shape[1], type='L') # [512,512]
                object_mask_np_aug = np.where((np.array(object_img_aug) / 255) == 0, 0, 1)                              # [512,512]
                # [3-1] anomal mask
                while True:
                    augmented_image, anomal_mask_np = self.augment_image(img,
                                                                         self.anomaly_source_paths[anomal_src_idx]) # [512,512,3]
                    anomal_mask_np = anomal_mask_np * object_mask_np_aug # 1 = anomal, 0 = normal
                    if anomal_mask_np.sum() > 0:
                        break
                # [3-2] hole mask
                while True:
                    hold_mask_np = self.make_random_gaussian_mask()
                    hold_mask_np = hold_mask_np * object_mask_np_aug  # 1 = hole, 0 = normal
                    if hold_mask_np.sum() > 0:
                        break

            anomal_mask = np.repeat(np.expand_dims(anomal_mask_np, axis=2), 3, axis=2).astype(dtype) # 1 = anomal, 0 = normal
            anomal_img = (1 - anomal_mask) * img + anomal_mask * augmented_image # [512,512,3]

            hole_mask = np.repeat(np.expand_dims(hold_mask_np, axis=2), 3, axis=2).astype(dtype) # 1 = hole, 0 = normal
            hole_img = (1 - hole_mask) * img + hole_mask * background_img # [512,512]
            # [3] final
            anomal_mask_pil = Image.fromarray((anomal_mask * 255).astype(np.uint8)).resize((self.latent_res,self.latent_res)).convert('L')
            anomal_mask_torch = torch.tensor(np.array(anomal_mask_pil) / 255)
            anomal_mask = torch.where(anomal_mask_torch > 0, 1, 0)  # strict anomal
            hole_mask_pil = Image.fromarray((hole_mask * 255).astype(np.uint8)).resize((self.latent_res,self.latent_res)).convert('L')
            hole_mask_torch = torch.tensor(np.array(hole_mask_pil) / 255)
            hole_mask = torch.where(hole_mask_torch > 0, 1, 0)
        else :
            anomal_img = img
            anomal_mask = object_mask # [64,64]
        input_ids, attention_mask = self.get_input_ids(self.caption) # input_ids = [77]
        if anomal_mask.sum().item() == 0 :
            raise Exception(f"no anomal on {final_name} image, check mask again")
        if hole_mask.sum().item() == 0 :
            raise Exception(f"no hole on {final_name} image, check mask again")
        sample = {'image': self.transform(img),               # original image
                  "object_mask": object_mask.unsqueeze(0),    # [1, 64, 64]
                  'augmented_image': self.transform(anomal_img),
                  "anomaly_mask": anomal_mask.unsqueeze(0),   # [1, 64, 64] ################################
                  'masked_image': self.transform(hole_img),   # masked image
                  'masked_image_mask': hole_mask.unsqueeze(0),# hold position
                  'idx': idx,
                  'input_ids': input_ids.squeeze(0),
                  'caption': self.caption,
                  'image_name' : final_name}
        return sample
