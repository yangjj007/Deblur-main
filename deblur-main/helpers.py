import cv2
import numpy as np


def open_image(image_path: str) -> np.ndarray:
    """
    Open image as NumPy Array 64-bit float with 3 channels (BGR)
    
    Parameters:
    - image_path: string containing the path to the image file

    Returns:
    - image: 3D 64bit float array, shape is (height, width, channels)
    """
    image_cv = cv2.imread(image_path, flags=cv2.IMREAD_COLOR)
    return np.array(image_cv, np.float64)

def write_image(image_path: str, image: np.ndarray, transform=None) -> None:
    """
    Write a NumPy Array as a png image. Will apply transform
    to the array before writing

    Parameters:
    - image_path: string, output image file path
    - image: numpy array containing the image data
    - transform: function that will transform
    """
    if transform:
        pass
    if type(image) != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    cv2.imwrite(image_path, image)

def kernel_from_image(image_path: str) -> np.ndarray:
    """
    Open an image as NumPy Array 64-bit float 1 channel and
    normalize values so that they sum 1

    Parameters:
    - image_path: string containing the path to the image file

    Returns:
    - kernel: 2D 64bit float array
    """
    image_cv = cv2.imread(image_path, flags=cv2.IMREAD_GRAYSCALE)
    kernel = np.array(image_cv)
    kernel = kernel / np.sum(kernel)
    return kernel
    