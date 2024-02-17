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
import imgaug.augmenters as iaa

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
                 anomal_only_on_object : bool = True,
                 anomal_training : bool = False,
                 latent_res : int = 64,
                 perlin_max_scale : int = 8,
                 kernel_size : int = 5,
                 beta_scale_factor : float = 0.8,
                 do_anomal_hole : bool = True,
                 bgrm_test : bool = True):

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
        self.rot_augmenters = [iaa.Affine(rotate=(0, 0)),
                               iaa.Affine(rotate=(180, 180)),
                               iaa.Affine(rotate=(90, 90)),
                               iaa.Affine(rotate=(270, 270))]
        num_repeat = len(self.rot_augmenters)

        image_paths = sorted(glob.glob(root_dir + "/*.png"))
        self.image_paths = [image_path for image_path in image_paths for i in range(num_repeat)]
        self.anomal_only_on_object = anomal_only_on_object
        self.anomal_training = anomal_training
        self.latent_res = latent_res
        self.perlin_max_scale = perlin_max_scale
        self.kernel_size = kernel_size
        self.beta_scale_factor = beta_scale_factor
        self.down_sizer = transforms.Resize(size=(self.latent_res,self.latent_res), antialias=True)
        self.do_anomal_hole = do_anomal_hole
        self.bgrm_test = bgrm_test

    def __len__(self):
        if len(self.anomaly_source_paths) > 0 :
            return max(len(self.image_paths), len(self.anomaly_source_paths))
        else:
            return len(self.image_paths)

    def torch_to_pil(self, torch_img):

        # torch_img = [3, H, W], from -1 to 1
        np_img = np.array(((torch_img + 1) / 2) * 255).astype(np.uint8).transpose(1, 2, 0)
        pil = Image.fromarray(np_img)
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
        object_mask_dir = os.path.join(parent, f"back_rm_object_mask/{name}")
        return object_mask_dir

    def randAugmenter(self, idx):
        rot_aug_ind = idx % len(self.rot_augmenters)
        rot_aug = self.rot_augmenters[rot_aug_ind]
        aug = iaa.Sequential([rot_aug])
        return aug

    def get_input_ids(self, caption):
        tokenizer_output = self.tokenizer(caption, padding="max_length", truncation=True,return_tensors="pt")
        input_ids = tokenizer_output.input_ids
        attention_mask = tokenizer_output.attention_mask
        return input_ids, attention_mask

    def augment_image(self, image, anomaly_source_img, beta_scale_factor, object_position):

        # [2] perlin noise
        while True :
            perlin_scale = 6
            min_perlin_scale = 0
            perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
            perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
            perlin_noise = rand_perlin_2d_np((self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
            threshold = 0.5
            #perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
            # 0 and more than 0.5
            perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
            # only on object
            if object_position is not None:
                total_object_pixel = np.sum(object_position)
                perlin_thr = perlin_thr * object_position
            # smoothing
            perlin_thr = cv2.GaussianBlur(perlin_thr, (5,5), 0)
            binary_2D_mask = (np.where(perlin_thr == 0, 0, 1)).astype(np.float32)  # [512,512,3]
            if np.sum(binary_2D_mask) > 0.01 * total_object_pixel :
                break
        perlin_thr = np.expand_dims(perlin_thr, axis=2)  # [512,512,3]
        # if i rase beta_scale_factor,
        beta = torch.rand(1).numpy()[0] * beta_scale_factor
        A = beta * image + (1 - beta) * anomaly_source_img.astype(np.float32) # merged
        augmented_image = (image * (1 - perlin_thr) + A * perlin_thr).astype(np.float32)
        return augmented_image, binary_2D_mask # [512,512,3], [512,512]

    def gaussian_augment_image(self, image, back_img, object_position):
        # [2] perlin noise
        while True:
            end_num = self.resize_shape[0]
            x = np.arange(0, end_num, 1, float)
            y = np.arange(0, end_num, 1, float)[:, np.newaxis]
            x_0 = torch.randint(int(end_num / 4), int(3 * end_num / 4), (1,)).item()
            y_0 = torch.randint(int(end_num / 4), int(3 * end_num / 4), (1,)).item()
            sigma = torch.randint(25, 60, (1,)).item()
            result = np.exp(-4 * np.log(2) * ((x - x_0) ** 2 + (y - y_0) ** 2) / sigma ** 2)  # 0 ~ 1
            result_thr = np.where(result < 0.5, 0, 1).astype(np.float32)
            if object_position is not None:
                total_object_pixel = np.sum(object_position)
                result_thr = (result_thr * object_position).astype(np.float32)
            blur_2D_mask = cv2.GaussianBlur(result_thr, (3,3), 0)
            binary_2D_mask = (np.where(blur_2D_mask == 0, 0, 1)).astype(np.float32)  # [512,512,3]
            if np.sum(binary_2D_mask) > 0.01 * total_object_pixel :
                break
        blur_3D_mask = np.expand_dims(blur_2D_mask, axis=2)  # [512,512,3]
        A = back_img.astype(np.float32)  # merged
        augmented_image = (image * (1 - blur_3D_mask) + A * blur_3D_mask).astype(np.float32)
        return augmented_image, binary_2D_mask  # [512,512,3], [512,512]

    def load_image(self, image_path, trg_h, trg_w, type='RGB'):
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

    def __getitem__(self, idx):

        # [0] augmenter
        aug = self.randAugmenter(idx)

        # [1] base
        img_idx = idx % len(self.image_paths)
        img_path = self.image_paths[img_idx]
        img = self.load_image(img_path, self.resize_shape[0], self.resize_shape[1]) # np.array,
        img = aug(image=img)
        final_name = self.get_img_name(img_path)

        # [2] background
        if self.bgrm_test :
            background_dir = None
        else :
            if self.do_anomal_hole :
                parent, name = os.path.split(img_path)
                parent, _ = os.path.split(parent)
                background_dir = os.path.join(parent, f"background/{name}")
            else :
                background_dir = None

        # [3] object mask
        object_mask_dir = self.get_object_mask_dir(img_path)
        object_img = self.load_image(object_mask_dir, self.latent_res, self.latent_res, type='L')
        object_img = aug(image=object_img)
        object_mask_np = np.where((np.array(object_img, np.uint8) / 255) == 0, 0, 1)
        object_mask = torch.tensor(object_mask_np) # shape = [64,64], 0 = background, 1 = object

        # [4] augment image (pseudo anomal)
        if len(self.anomaly_source_paths) > 0:
            anomal_src_idx = idx % len(self.anomaly_source_paths)
            anomal_src_path = self.anomaly_source_paths[anomal_src_idx]
            anomal_name = self.get_img_name(anomal_src_path)

            if not self.anomal_only_on_object:

                # [4.1] anomal img
                anomaly_source_img = self.load_image(self.anomaly_source_paths[anomal_src_idx], self.resize_shape[0],
                                                     self.resize_shape[1])
                augmented_image, mask = self.augment_image(img, anomaly_source_img,
                                                           beta_scale_factor=self.beta_scale_factor,
                                                           object_position=None)  # [512,512,3], [512,512]
                anomal_img = np.array(Image.fromarray(augmented_image.astype(np.uint8)), np.uint8)
                anomal_mask_torch = self.down_sizer(torch.tensor(mask).unsqueeze(0))  # [1,64,64]

                # [4.2] holed img
                if self.do_anomal_hole:
                    if self.bgrm_test:
                        background_img = (img * 0).astype(img.dtype)
                    else :
                        background_img = self.load_image(background_dir, self.resize_shape[0], self.resize_shape[1],type='RGB')
                    background_img = aug(image=background_img)
                    back_augmented_image, hole_mask = self.gaussian_augment_image(img, background_img, None)
                    back_anomal_img = np.array(Image.fromarray(back_augmented_image.astype(np.uint8)), np.uint8)
                    back_anomal_mask_torch = self.down_sizer(torch.tensor(hole_mask).unsqueeze(0))  # [1,64,64]
                    masked_image = self.transform(back_anomal_img)
                    masked_image_mask = back_anomal_mask_torch
                else:
                    masked_image = img
                    masked_image_mask = object_mask

            if self.anomal_only_on_object:
                object_img_aug = aug(image=self.load_image(object_mask_dir, self.resize_shape[0], self.resize_shape[1], type='L') )
                object_position = np.where((np.array(object_img_aug)) == 0, 0, 1)             # [512,512]
                # [4.1] anomal img
                anomaly_source_img = self.load_image(self.anomaly_source_paths[anomal_src_idx], self.resize_shape[0], self.resize_shape[1])
                augmented_image, mask = self.augment_image(img,anomaly_source_img, beta_scale_factor=self.beta_scale_factor,object_position=object_position) # [512,512,3], [512,512]
                anomal_img = np.array(Image.fromarray(augmented_image.astype(np.uint8)), np.uint8)
                anomal_mask_torch = self.down_sizer(torch.tensor(mask).unsqueeze(0)) # [1,64,64]
                # [4.2] holed img
                if self.do_anomal_hole:
                    if self.bgrm_test:
                        background_img = (img * 0).astype(img.dtype)
                    else :
                        background_img = self.load_image(background_dir, self.resize_shape[0], self.resize_shape[1],type='RGB')
                    back_augmented_image, back_mask = self.gaussian_augment_image(img, aug(image=background_img),
                                                                                  object_position = object_position)

                    back_anomal_img = np.array(Image.fromarray(back_augmented_image.astype(np.uint8)), np.uint8)
                    back_anomal_mask_torch = self.down_sizer(torch.tensor(back_mask).unsqueeze(0)) # [1,64,64]

                else :
                    back_anomal_img = img
                    back_anomal_mask_torch = torch.randn(self.latent_res, self.latent_res)
        else :
            anomal_img = img
            anomal_mask_torch = object_mask # [64,64]
            back_anomal_img = img
            back_anomal_mask_torch = torch.randn(self.latent_res, self.latent_res)

        input_ids, attention_mask = self.get_input_ids(self.caption) # input_ids = [77]


        return {'image': self.transform(img),               # original image
                "object_mask": object_mask.unsqueeze(0),    # [1, 64, 64]

                'anomal_image': self.transform(anomal_img),
                "anomal_mask": anomal_mask_torch,   # [1, 64, 64] ################################

                'bg_anomal_image': self.transform(back_anomal_img),          # masked image
                'bg_anomal_mask': back_anomal_mask_torch,

                'idx': idx,
                'input_ids': input_ids.squeeze(0),
                'caption': self.caption,
                'image_name' : final_name,
                'anomal_name' : anomal_name,}