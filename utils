import cv2
import numpy as np

def get_block_size(img):
    # Convert image to grayscale and float32 format
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = img_gray.astype(np.float32) / 255

    # Apply 2D DCT to image
    dct = cv2.dct(img_gray)

    # Calculate the distribution of DCT coefficients
    hist, _ = np.histogram(np.abs(dct), bins=64)

    # Find the peak in the histogram
    max_bin = np.argmax(hist)

    # Calculate the block size from the peak bin index
    block_size = int(np.sqrt(img_gray.size / max_bin))

    return block_size

# Load the JPEG image
img_jpeg = cv2.imread('image.jpg')

# Get the block size
block_size = get_block_size(img_jpeg)

print('Block size:', block_size)
