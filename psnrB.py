import cv2
import numpy as np

def psnr_b(img_dist, img_orig, block_size=8):
    # Convert images to float32 and scale to [0, 1]
    img_dist = img_dist.astype(np.float32) / 255
    img_orig = img_orig.astype(np.float32) / 255

    # Calculate PSNR of original image
    mse_orig = np.mean((img_orig - img_orig.mean()) ** 2)
    psnr_orig = 10 * np.log10(1 / mse_orig)

    # Estimate blocking artifact level in distorted image
    img_dist_gray = cv2.cvtColor(img_dist, cv2.COLOR_BGR2GRAY)
    block_var = cv2.blur(img_dist_gray, (block_size, block_size))
    block_mask = (block_var > 0.01).astype(np.float32)

    # Apply filter to reduce blocking artifacts
    img_filtered = cv2.fastNlMeansDenoisingColored(img_dist, None, 10, 10, 7, 21)

    # Calculate PSNR of filtered image
    mse_filtered = np.mean((img_filtered - img_orig) ** 2)
    psnr_filtered = 10 * np.log10(1 / mse_filtered)

    # Calculate PSNR-B
    psnrb = psnr_filtered - psnr_orig

    return psnrb

# Load the blocked and reference images
img_blocked = cv2.imread('blocked.jpg')
img_ref = cv2.imread('reference.png')

# Calculate the PSNR-B score
psnrb = psnr_b(img_blocked, img_ref)

print('PSNR-B score:', psnrb)
