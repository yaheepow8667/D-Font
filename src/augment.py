# Augmentation utilities for DP-Font few-shot fine-tuning
# Provides functions to perform on-the-fly augmentations for the 200 sample set.
# Uses: PIL, numpy, cv2
import random
import numpy as np
from PIL import Image, ImageFilter, ImageOps
import cv2

def to_numpy_gray(img):
    # img: PIL Image (L) -> np uint8
    return np.array(img)

def from_numpy_gray(arr):
    return Image.fromarray(arr.astype(np.uint8))

def random_rotate(img, max_angle=4.0):
    angle = random.uniform(-max_angle, max_angle)
    return img.rotate(angle, resample=Image.BILINEAR, fillcolor=255)

def random_translate_scale(img, max_shift=3, scale_range=(0.95,1.05)):
    w,h = img.size
    tx = random.uniform(-max_shift, max_shift)
    ty = random.uniform(-max_shift, max_shift)
    scale = random.uniform(scale_range[0], scale_range[1])
    M = np.array([[scale, 0, tx],[0, scale, ty]], dtype=np.float32)
    arr = to_numpy_gray(img)
    dst = cv2.warpAffine(arr, M, (w,h), borderMode=cv2.BORDER_REFLECT)
    return from_numpy_gray(dst)

def gaussian_blur(img, sigma=0.7):
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))

def random_noise(img, var=5.0):
    arr = to_numpy_gray(img).astype(np.float32)
    noise = np.random.normal(0, var, arr.shape)
    arr = np.clip(arr + noise, 0, 255)
    return from_numpy_gray(arr)

def dilate_erode(img, mode='dilate', ksize=1):
    arr = to_numpy_gray(img)
    kernel = np.ones((ksize,ksize), np.uint8)
    if mode == 'dilate':
        out = cv2.dilate(arr, kernel, iterations=1)
    else:
        out = cv2.erode(arr, kernel, iterations=1)
    return from_numpy_gray(out)

def elastic_transform(img, alpha=34, sigma=4):
    # simple elastic transform using OpenCV remap
    arr = to_numpy_gray(img)
    shape = arr.shape
    dx = (cv2.GaussianBlur((np.random.rand(*shape) * 2 - 1).astype(np.float32), (17,17), sigma) * alpha).astype(np.float32)
    dy = (cv2.GaussianBlur((np.random.rand(*shape) * 2 - 1).astype(np.float32), (17,17), sigma) * alpha).astype(np.float32)
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    warped = cv2.remap(arr, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return from_numpy_gray(warped)

def augment_image(img_pil, prob_config=None):
    """Apply a random sequence of augmentations to a PIL Image (L mode).
    prob_config: dict to control probabilities
    """
    if prob_config is None:
        prob_config = {}
    img = img_pil
    if random.random() < prob_config.get('rotate', 0.6):
        img = random_rotate(img, max_angle=prob_config.get('max_angle', 4.0))
    if random.random() < prob_config.get('translate', 0.6):
        img = random_translate_scale(img, max_shift=prob_config.get('max_shift', 3))
    if random.random() < prob_config.get('elastic', 0.3):
        img = elastic_transform(img, alpha=prob_config.get('alpha', 30), sigma=prob_config.get('sigma', 4))
    if random.random() < prob_config.get('blur', 0.2):
        img = gaussian_blur(img, sigma=prob_config.get('blur_sigma', 0.6))
    if random.random() < prob_config.get('morph', 0.2):
        img = dilate_erode(img, mode=random.choice(['dilate','erode']), ksize=random.choice([1,2]))
    if random.random() < prob_config.get('noise', 0.2):
        img = random_noise(img, var=prob_config.get('noise_var', 5.0))
    return img
