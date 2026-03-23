import os
import numpy as np
import tempfile
import shutil
from PIL import Image
import pytest
from classifier.gesture_dataset import GestureDataset
from classifier.config import HAND_CROP_SIZE, STATIC_GESTURE_CLASSES

def create_dummy_dataset(root_dir, num_classes=2, num_images=3):
    os.makedirs(root_dir, exist_ok=True)
    for class_name in STATIC_GESTURE_CLASSES[:num_classes]:
        class_dir = os.path.join(root_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        for i in range(num_images):
            img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
            img.save(os.path.join(class_dir, f"img_{i}.jpg"))

def test_gesture_dataset_loading():
    tmpdir = tempfile.mkdtemp()
    try:
        create_dummy_dataset(tmpdir)
        dataset = GestureDataset(tmpdir)
        assert len(dataset) == 2 * 3  # num_classes * num_images
        img, class_idx = dataset[0]
        assert isinstance(img, np.ndarray)
        assert img.shape == (HAND_CROP_SIZE, HAND_CROP_SIZE, 3)
        assert isinstance(class_idx, int)
    finally:
        shutil.rmtree(tmpdir)

def test_gesture_dataset_missing_image():
    tmpdir = tempfile.mkdtemp()
    try:
        create_dummy_dataset(tmpdir, num_classes=1, num_images=1)
        # Remove the image to simulate missing file
        class_dir = os.path.join(tmpdir, STATIC_GESTURE_CLASSES[0])
        img_path = os.path.join(class_dir, "img_0.jpg")
        os.remove(img_path)
        dataset = GestureDataset(tmpdir)
        with pytest.raises(IndexError):
            _ = dataset[0]
    finally:
        shutil.rmtree(tmpdir)