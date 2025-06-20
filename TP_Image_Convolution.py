# TP_Convolution_MIAA_IMSD.py

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# Function to load image, handling grayscale and RGB automatically
def image_load(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")

    if len(img.shape) == 3:
        # Check if it's a color image (3 channels)
        if img.shape[2] == 3:
            # Convert BGR to RGB for matplotlib display
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 'RGB'
        else:
            # Handle cases where image has more than 3 channels or unexpected format
            raise ValueError("Unsupported image format: expected 3 channels for RGB.")
    elif len(img.shape) == 2:
        # Grayscale image (2 dimensions)
        return img, 'Grayscale'
    else:
        raise ValueError("Unsupported image format: expected 2 or 3 dimensions.")

# Function to apply convolution
def apply_convolution(image, kernel):
    assert isinstance(image, np.ndarray), "Image must be a NumPy array"
    assert isinstance(kernel, np.ndarray), "Kernel must be a NumPy array"
    assert len(kernel.shape) == 2, "Kernel must be a 2D matrix"
    assert kernel.shape[0] == kernel.shape[1], "Kernel must be square"
    assert kernel.shape[0] % 2 == 1, "Kernel must have odd dimensions"
    assert len(image.shape) in [2, 3], "Image must be grayscale (2D) or RGB (3D)"

    if len(image.shape) == 3:  # RGB Image
        assert image.shape[2] == 3, "RGB image must have 3 channels"
        height, width, channels = image.shape
        output = np.zeros_like(image, dtype=np.float32)
        for c in range(channels):
            output[:, :, c] = convolve_channel(image[:, :, c], kernel)
    else:  # Grayscale Image
        output = convolve_channel(image, kernel)

    # TODO: Add assertions for output size based on image and kernel
    # The output size should be the same as the input image size due to padding
    assert output.shape[:2] == image.shape[:2], "Output image dimensions do not match input image dimensions."

    output = np.clip(output, 0, 255).astype(np.uint8)
    return output

# Function to convolve a single channel
def convolve_channel(image, kernel):
    assert image.shape[0] >= kernel.shape[0], "Image is too small for the kernel in height"
    assert image.shape[1] >= kernel.shape[1], "Image is too small for the kernel in width"

    img_height, img_width = image.shape
    kernel_height, kernel_width = kernel.shape
    pad_height, pad_width = kernel_height // 2, kernel_width // 2

    # TODO: Make np.pad parameters dynamic based on kernel size
    # The padding is already dynamic based on kernel_height // 2 and kernel_width // 2
    # The mode 'constant' with default 0 is suitable for convolution.
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')
    output = np.zeros_like(image, dtype=np.float32)

    for i in range(img_height):
        for j in range(img_width):
            region = padded_image[i:i + kernel_height, j:j + kernel_width]
            output[i, j] = np.sum(region * kernel)

    # TODO: Add assertions for output size based on image, kernel, and channel
    # This assertion is already handled in apply_convolution, but can be added here for redundancy
    assert output.shape == image.shape, "Convolved channel output shape does not match input channel shape."

    return output

# Function to display images (original and filtered)
def display_images(original, filtered, title):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original, cmap='gray' if len(original.shape) == 2 else None)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(title)
    plt.imshow(filtered, cmap='gray' if len(filtered.shape) == 2 else None)
    plt.axis('off')

    plt.show()

# TODO: Improve display_images to show original with different filters in one figure with captions
def display_multiple_images(original, filtered_images, titles):
    num_filters = len(filtered_images)
    fig, axes = plt.subplots(1, num_filters + 1, figsize=(5 * (num_filters + 1), 5))

    axes[0].imshow(original, cmap='gray' if len(original.shape) == 2 else None)
    axes[0].set_title("Original")
    axes[0].axis('off')

    for i in range(num_filters):
        axes[i+1].imshow(filtered_images[i], cmap='gray' if len(filtered_images[i].shape) == 2 else None)
        axes[i+1].set_title(titles[i])
        axes[i+1].axis('off')
    plt.tight_layout()
    plt.show()

# Define convolution kernels
# 1. Blur filter (average)
blur_kernel = np.array([
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9]
])

# 2. Sobel horizontal edge detection
sobel_horizontal = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

# 3. Sobel vertical edge detection
sobel_vertical = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
])

# Additional kernels from Wikipedia (example: Sharpen, Gaussian Blur)
# Sharpen kernel
sharpen_kernel = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])

# Gaussian Blur (3x3, sigma=1)
gaussian_blur_kernel = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
]) / 16

# Edge detection (Laplacian)
laplacian_kernel = np.array([
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]
])

# Emboss kernel
emboss_kernel = np.array([
    [-2, -1, 0],
    [-1, 1, 1],
    [0, 1, 2]
])

# Function to generate random kernel
def generate_random_kernel(size, seed=None):
    if seed is not None:
        np.random.seed(seed)
    kernel = np.random.rand(size, size) * 2 - 1  # Values between -1 and 1
    return kernel

# Main execution block
if __name__ == "__main__":
    image_files = {
        "Grayscale": "image_gray.jpg",
        "RGB": "image_RGB.jpg"
    }

    for img_type, filename in image_files.items():
        print(f"\nProcessing {img_type} image: {filename}")
        try:
            original_image, image_color_type = image_load(filename)
            print(f"Image loaded successfully. Type: {image_color_type}")
        except (FileNotFoundError, ValueError) as e:
            print(f"Error loading image: {e}")
            continue

        kernels = {
            "Blur": blur_kernel,
            "Sobel Horizontal": sobel_horizontal,
            "Sobel Vertical": sobel_vertical,
            "Sharpen": sharpen_kernel,
            "Gaussian Blur": gaussian_blur_kernel,
            "Laplacian": laplacian_kernel,
            "Emboss": emboss_kernel,
            "Random 3x3 (Seed 42)": generate_random_kernel(3, seed=42),
            "Random 5x5 (Seed 123)": generate_random_kernel(5, seed=123)
        }

        filtered_images = []
        filter_titles = []

        for name, kernel in kernels.items():
            print(f"Applying {name} filter...")
            filtered_img = apply_convolution(original_image, kernel)
            filtered_images.append(filtered_img)
            filter_titles.append(name)
            # Save individual filtered images
            save_name = f"filtered_{img_type}_{name.replace(' ', '_').replace('(', '').replace(')', '')}.jpg"
            cv2.imwrite(save_name, cv2.cvtColor(filtered_img, cv2.COLOR_RGB2BGR) if image_color_type == 'RGB' else filtered_img)

        # Display all filtered images in one figure
        display_multiple_images(original_image, filtered_images, filter_titles)

    print("All filters applied and images saved.")


