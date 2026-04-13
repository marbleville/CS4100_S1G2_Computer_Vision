"""
GestureDataset: Loads gesture images, applies resizing and normalization.
Assumes images are organized as: root/class_name/image.jpg
"""

import os
import numpy as np
from PIL import Image
from classifier.config import HAND_CROP_SIZE, NORMALIZATION_MEAN, NORMALIZATION_STD, ALL_GESTURE_CLASSES

class GestureDataset:
    """
    Dataset class for gesture recognition.

    Loads images from a directory structure organized by gesture class, resizes and normalizes them,
    and provides access to image/class pairs for training or evaluation.

    Directory structure should be:
        root_dir/
            class_1/
                img1.jpg
                img2.jpg
                ...
            class_2/
                img1.jpg
                ...
            ...
    """

    def __init__(self, root_dir):
        """
        Initialize the dataset by scanning for images and mapping them to class indices.

        Args:
            root_dir (str): Path to dataset root. Should contain subfolders for each gesture class.
        """
        self.root_dir = root_dir
        self.samples = []  # List of (image_path, class_idx)
        self.class_to_idx = {cls: i for i, cls in enumerate(ALL_GESTURE_CLASSES)}
        self._gather_samples()

    def _gather_samples(self):
        """
        Scan the dataset directory and collect all image file paths and their corresponding class indices.
        Only files with .jpg, .jpeg, or .png extensions are included.
        """
        for class_name in ALL_GESTURE_CLASSES:
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(class_dir, fname), self.class_to_idx[class_name]))

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Retrieve a sample by index. If the image is missing or corrupt, raises IndexError.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            img (np.ndarray): Normalized image array of shape (HAND_CROP_SIZE, HAND_CROP_SIZE, 3).
            class_idx (int): Integer index of the gesture class.
        """
        img_path, class_idx = self.samples[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize((HAND_CROP_SIZE, HAND_CROP_SIZE))
            img = np.array(img).astype(np.float32) / 255.0
            # Normalize
            img = (img - NORMALIZATION_MEAN) / NORMALIZATION_STD
            return img, class_idx
        except Exception as e:
            print(f"Warning: Could not load image {img_path}: {e}")
            raise IndexError(f"Sample at index {idx} is invalid or corrupt.")
            
    def get_class_name(self, class_idx):
        """
        Get the gesture class name corresponding to a class index.

        Args:
            class_idx (int): Integer index of the gesture class.

        Returns:
            str or None: Name of the gesture class, or None if not found.
        """
        for name, idx in self.class_to_idx.items():
            if idx == class_idx:
                return name
        return None
