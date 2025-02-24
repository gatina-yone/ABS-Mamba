import os
import numpy as np
import torch
import torch.nn.functional as F
import os.path
import random
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image

def save_image(tensor, path):
    """
    Save the image tensor to the specified path.
    :param tensor: PyTorch tensor with shape (C, H, W)
    :param path: Target save path
    """
    tensor = tensor.permute(1, 2, 0)  # Convert to (H, W, C)
    tensor = (tensor * 255).clamp(0, 255).byte()  # Scale to [0, 255] and convert to uint8

    if tensor.shape[2] == 1:
        tensor = tensor.squeeze(2)  # Remove single channel for grayscale images

    image = Image.fromarray(tensor.cpu().numpy())  # Convert tensor to PIL image
    image.save(path)  # Save the image
    print(f"Image saved to {path}")

def save_images(A, B, output_dir, prefix="image"):
    """
    Save images A and B to the specified directory.
    :param A: Source image tensor
    :param B: Target image tensor
    :param output_dir: Target save directory
    :param prefix: Prefix for image filenames
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Create directory if it doesn't exist

    A_path = os.path.join(output_dir, f"{prefix}_source.png")  # Path for source image A
    save_image(A, A_path)  # Save source image A

    B_path = os.path.join(output_dir, f"{prefix}_target.png")  # Path for target image B
    save_image(B, B_path)  # Save target image B

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        """
        Initialize the dataset.
        Args:
            opt: Configuration options containing data paths, image sizes, etc.
        """
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # Directory for paired images
        self.AB_paths = sorted(make_dataset(self.dir_AB))  # List of sorted .npy file paths
        assert opt.resize_or_crop == 'resize_and_crop', "Only 'resize_and_crop' is supported."

    def __len__(self):
        """
        Return the size of the dataset.
        """
        return len(self.AB_paths)

    def __getitem__(self, index):
        """
        Get data at the specified index.
        Args:
            index: Data index
        Returns:
            Dictionary containing source image 'A', target image 'B', and their paths
        """
        AB_path = self.AB_paths[index]
        AB = np.load(AB_path)  # Load .npy file

        # Convert numpy array to tensor based on dimensions
        if AB.ndim == 2:
            AB_tensor = torch.from_numpy(AB).unsqueeze(0).float()  # Grayscale: (H, W) -> (1, H, W)
        elif AB.ndim == 3:
            AB_tensor = torch.from_numpy(AB).permute(2, 0, 1).float()  # RGB: (H, W, C) -> (C, H, W)
        else:
            raise ValueError(f"Unsupported AB shape: {AB.shape}")

        _, h, w = AB_tensor.size()  # Get height and width
        target_width = 256  # Target image width
        source_width = w - target_width  # Source image width
        num_sources = source_width // 256  # Number of source images

        if num_sources < 1:
            raise ValueError(f"Invalid source width {source_width} in AB_path: {AB_path}")

        # Extract source images
        A_list = []
        for i in range(num_sources):
            left = i * 256
            A = AB_tensor[:, :, left:left + 256]  # Extract segment of width 256
            A = F.interpolate(A.unsqueeze(0), size=(self.opt.loadSize, self.opt.loadSize),
                            mode='bicubic', align_corners=True).squeeze(0)  # Resize
            A_list.append(A)

        # Combine source images along channel dimension if multiple sources exist
        if num_sources > 1:
            A = torch.cat(A_list, dim=0)
        else:
            A = A_list[0]

        right = 256 * num_sources
        B = AB_tensor[:, :, right:right + 256]  # Extract target image (rightmost 256 pixels)
        B = F.interpolate(B.unsqueeze(0), size=(self.opt.loadSize, self.opt.loadSize),
                        mode='bicubic', align_corners=True).squeeze(0)  # Resize

        # Random crop if fineSize is smaller than loadSize
        if self.opt.fineSize < self.opt.loadSize:
            w_offset = random.randint(0, self.opt.loadSize - self.opt.fineSize - 1)
            h_offset = random.randint(0, self.opt.loadSize - self.opt.fineSize - 1)
            A = A[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]
            B = B[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]

        # Random horizontal flip for data augmentation
        if not self.opt.no_flip and random.random() < 0.5:
            A = torch.flip(A, dims=[2])
            B = torch.flip(B, dims=[2])

        # Adjust input and output channel numbers based on direction
        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        # Adjust channels for A
        if input_nc == 1:
            if A.size(0) == 3:
                A = (A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114).unsqueeze(0)  # RGB to grayscale
            elif A.size(0) > 1:
                A = A[:1, ...]  # Take first channel
        elif input_nc == 2:
            if A.size(0) >= 2:
                A = A[:2, ...]  # Take first two channels
            else:
                raise ValueError(f"Input channels less than 2 in AB_path: {AB_path}")

        # Adjust channels for B
        if output_nc == 1:
            if B.size(0) == 3:
                B = (B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114).unsqueeze(0)  # RGB to grayscale
            elif B.size(0) > 1:
                B = B[:1, ...]  # Take first channel
        elif output_nc == 2:
            if B.size(0) >= 2:
                B = B[:2, ...]  # Take first two channels
            else:
                raise ValueError(f"Output channels less than 2 in AB_path: {AB_path}")

        return {
            'A': A,  # Source image tensor
            'B': B,  # Target image tensor
            'A_paths': AB_path,  # Source image path
            'B_paths': AB_path  # Target image path (same as source)
        }

    def name(self):
        """
        Return the name of the dataset.
        """
        return 'AlignedDataset'
