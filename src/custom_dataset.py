import glob
import random
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
        
        return input['image'], input['label']
    
def load_data(opt):

    

    train_metadata = os.path.join(opt.dataroot, 'metadata/train_metadata.json')
    test_metadata = os.path.join(opt.dataroot, 'metadata/test_metadata.json')

    splits = ['train', 'test', 'train4val']

    masks = AddMask(opt)
    collate = {'train': masks.append_mask, 'test': masks.append_mask, 'train4val': masks.append_mask} # collate functions for dataloader
    # collate = {x: None for x in splits}

    splits2metadata = {'train': train_metadata, 'test': test_metadata, 'train4val': train_metadata}
    drop_last_batch = {'train': True, 'test': False, 'train4val': False}
    shuffle = {'train': True, 'test': False, 'train4val': False}

    dataset = {x: StitchoDataset(meta_file=splits2metadata[x], transform_fn=None, resize_dim=(512, 512)) for x in splits}
    # dataset = {x: get_custom_anomaly_dataset(dataset[x], opt.normal_class) for x in dataset.keys()}

    dataloader = {}

    for x in splits:
        if collate[x] is not None:
            dataloader[x] = torch.utils.data.DataLoader(dataset=dataset[x],
                                                        batch_size=opt.BATCH_SIZE,
                                                        shuffle=shuffle[x],
                                                        num_workers=int(opt.workers),
                                                        drop_last=drop_last_batch[x],
                                                        collate_fn=collate[x])
        else:
            dataloader[x] = torch.utils.data.DataLoader(dataset=dataset[x],
                                                        batch_size=opt.BATCH_SIZE,
                                                        shuffle=shuffle[x],
                                                        num_workers=int(opt.workers),
                                                        drop_last=drop_last_batch[x])
    return dataloader

    


class AddMask():
    def __init__(self, config):
        self.flist = list(glob.glob(config.TRAIN_MASK_FLIST + '/*.jpg')) + list(glob.glob(config.TRAIN_MASK_FLIST + '/*.png'))
        self.flist.sort()
        self.mask_set = []
        self.mask_type = config.MASK_TYPE
        for scale in config.SCALES:
            for mask_index in range(scale*4, (scale+1)*4):
                mask = Image.open(self.flist[mask_index])
                mask = transforms.Resize(config.INPUT_SIZE, interpolation=Image.NEAREST)(mask)
                # mask = (mask > 0).astype(np.uint8) * 255
                self.mask_set.append(transforms.ToTensor()(mask))

    def append_mask(self, batch):
        masks = []
        imgs = []
        label = []
        for i in range(len(batch)):
            img, target = batch[i]
            imgs.append(img)
            masks.append(random.choice(self.mask_set))
            label.append(target)
        imgs = torch.stack(imgs, dim=0)
        mask_batch = torch.stack(masks, dim=0)
        label = torch.FloatTensor(label)
        if self.mask_type == 0:
            mask_batch = None
        return imgs, mask_batch, label
    
def get_custom_anomaly_dataset(subset, nrm_cls):
    nrm_cls_idx = subset.class_to_idx[nrm_cls]
    idx_to_class = {v: k for k, v in subset.class_to_idx.items()}
    new_targets = [0 if x == nrm_cls_idx else 1 for x in subset.targets]
    new_samples = [(x[0], 0 if x[1] == nrm_cls_idx else 1) for x in subset.samples]
    subset.class_name = [idx_to_class[x] for x in subset.targets]
    subset.targets = new_targets
    subset.samples = new_samples
    subset.imgs = new_samples
    return subset