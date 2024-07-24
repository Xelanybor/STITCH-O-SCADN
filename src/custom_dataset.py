import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import yaml
from PIL import Image
from torchvision import transforms
import json
from torch import from_numpy

class StitchoDataset(Dataset):
    def __init__(
        self,
        meta_file,
        transform_fn,
        resize_dim=None, 
        noise_factor=0, 
        p=0
    ):
        self.meta_file = meta_file
        self.transform_fn = transform_fn
        self.resize_dim = resize_dim
        self.noise_factor = noise_factor
        self.p = p

        # construct metas
        with open(meta_file, "r") as f_r:
            self.metas = []
            for line in f_r:
                meta = json.loads(line)
                self.metas.append(meta)

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, index):
        input = {}
        meta = self.metas[index]

        # read image
        filename = os.path.join("dataset/stitcho", meta["filename"])
        label = meta["label"]
        if os.path.exists(filename):
            image = np.load(filename)
        else:
            print(f"File {filename} does not exist.")
            exit()

        # print(image.shape)
        # exit()

        if self.resize_dim:
            image = cv2.resize(image, self.resize_dim)
        
        input.update(
            {
                "filename": filename,
                "label": label,
            }
        )

        if meta.get("clsname", None):
            input["clsname"] = meta["clsname"]
        else:
            input["clsname"] = filename.split("/")[-4]

        # print(image.shape)
        # exit()

        image = from_numpy(image).float().permute(2, 0, 1)
        # image = from_numpy(image).float()[None, :, :]

        # print(image.shape)
        # exit()

        if self.transform_fn:
            image = self.transform_fn(image)
        
        # if "mean" in meta and "std" in meta:
        #     normalize_fn = transforms.Normalize(mean=meta["mean"][0], std=meta["std"][0])
        #     # print(image.shape)
        #     image = normalize_fn(image)
            
            
        # # duplicate channels of image to 3
        # if image.size(0) == 1:
        #     image = image.expand(3, -1, -1)
            
        input.update({"image": image})
        # noisy_image = add_noise(image, self.noise_factor, self.p)
        # input.update({"noisy_image": noisy_image})
        
        return [input['image'], input['image'], input['label']]