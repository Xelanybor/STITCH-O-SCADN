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

class custom_standardize_transform(torch.nn.Module):
    """
    A custom transformation to normalize the input data (using mean and standard deviation for each channel) and then convert the .npy files to pytorch tensors.
    """
    def __init__(self, mean : np.array, std : np.array, device : str = 'cpu'):
        super().__init__()
        self.device = device
        self.mean = torch.from_numpy(mean).to(device)
        self.std = torch.from_numpy(std).to(device)

    def forward(self, np_patch):
        # convert patch to tensor
        tensor_patch = torch.from_numpy(np_patch).float().to(self.device)

        if tensor_patch.dim() == 2:
            tensor_patch = tensor_patch.unsqueeze(2)
        
        # change from (H, W, C) to (C, H, W)
        tensor_patch = tensor_patch.permute(2, 0, 1)

        # apply channel-wise normalization
        for c in range(tensor_patch.shape[0]):
            tensor_patch[c] = (tensor_patch[c] - self.mean[c]) / self.std[c]

        return tensor_patch

class custom_pscaling_transform(torch.nn.Module):
    """
    A custom transformation to scale the input data using the 1st and 99th percentiles for each channel.
    """
    def __init__(self, percentiles: np.ndarray, device: str = 'cpu'):
        super().__init__()
        self.device = device
        self.percentiles = torch.from_numpy(percentiles).float().to(device)

    def forward(self, np_patch):
        # convert patch to tensor
        tensor_patch = torch.from_numpy(np_patch).float().to(self.device)

        if tensor_patch.dim() == 2:
            tensor_patch = tensor_patch.unsqueeze(2)

        # change from (H, W, C) to (C, H, W)
        tensor_patch = tensor_patch.permute(2, 0, 1)

        # apply channel-wise scaling
        for c in range(tensor_patch.shape[0]):
            c_min, c_max = self.percentiles[c]
            tensor_patch[c] = (tensor_patch[c] - c_min) / (c_max - c_min)

        # clip values to be between 0 and 1 (might be an optional step)
        tensor_patch = torch.clamp(tensor_patch, 0, 1)

        return tensor_patch


class CustomDataset(Dataset):
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
        filename = os.path.join("chunks", meta["filename"])
        label = meta["label"]
        if os.path.exists(filename):
            image = np.load(filename)
        else:
            print(f"File {filename} does not exist.")
            exit()

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

        image = from_numpy(image).float().permute(2, 0, 1)        

        if self.transform_fn:
            image = self.transform_fn(image)
        
        if "mean" in meta and "std" in meta:
            normalize_fn = transforms.Normalize(mean=meta["mean"], std=meta["std"])
            image = normalize_fn(image)
            
            
        # # duplicate channels of image to 3
        # if image.size(0) == 1:
        #     image = image.expand(3, -1, -1)
            
        input.update({"image": image})
        noisy_image = add_noise(image, self.noise_factor, self.p)
        input.update({"noisy_image": noisy_image})
        
        return input

def test_loaders(train_data_loader, test_data_loader, num_images=4):
    '''
    Tests the data loaders by visualizing some images.

    Parameters:
    ----------
    train_data_loader : DataLoader, optional
        The DataLoader for the training data.
    test_data_loader : DataLoader, optional
        The DataLoader for the test data.
    num_images : int, optional
        The number of images to visualize (default is 4).
        
    '''
    if train_data_loader:
        # Loop through the 7 layers and print them on a grid in plt
        count = 0
        for image, label in train_data_loader:
            image, label = image[0], label[0]
            print(train_data_loader.dataset.get_class_name(label.numpy()))
            plt.figure(figsize=(15, 4))
            for i in range(7):
                plt.subplot(1, 7, i+1)
                plt.imshow(image[:, :, i], cmap='gray')
                plt.axis('off')
            plt.show()
            if count > num_images:
                break
            count+=1

    if test_data_loader:
        # Print an image from the test set with label 0 or 1
        count = 0
        cases = []
        for image, label in test_data_loader:
            image, label = image[0], label[0]
            if label.numpy() in cases:
                continue
            cases.append(label.item())
            print(test_data_loader.dataset.get_class_name(label.numpy()))
            plt.figure(figsize=(15, 4))
            for i in range(7):
                plt.subplot(1, 7, i+1)
                plt.imshow(image[:, :, i], cmap='gray')
                plt.axis('off')
            plt.show()
            if count > num_images:
                break
            count+=1

def add_noise(image, noise_factor, p):
        '''
        Function to randomly add noise to the image.

        Parameters:
        ----------
        img : torch.Tensor
            The image to add noise to.
        noise_factor : float
            The factor to multiply the noise by.
        p : float
            The probability of adding noise to the image.

        '''
        if np.random.rand() > p:
            return image
        noise = torch.randn_like(image) * noise_factor
        noisy_image = image + noise
        return torch.clamp(noisy_image, 0., 1.)

if __name__ == "__main__":
    try:
        with open("baseline_config.yaml", "r") as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
        batch_size = cfg["batch_size"]
        data_path = cfg["data_path"]
        data_stats_path = cfg["data_stats_path"]
    except Exception as e:
        print("Error reading config file: \n", e)
        exit()

    mean = np.load(data_stats_path + "train_means.npy")
    std_dev = np.load(data_stats_path + "train_stds.npy")
    transform = custom_standardize_transform(mean, std_dev)

    # Load data
    train_dataset = CustomDataset(root_dir=data_path + "train", transform=transform, resize_dim=(256, 256))
    test_dataset = CustomDataset(root_dir=data_path + "val", transform=transform, resize_dim=(256, 256))

    train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    test_loaders(train_data_loader, None, 10)