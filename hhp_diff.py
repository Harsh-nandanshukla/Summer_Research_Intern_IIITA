import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Directories
h_dir = "./examples_H"
hp_dir = "./examples_Hp"
output_dir = "./diff_visuals"
os.makedirs(output_dir, exist_ok=True)

# Helper to load and normalize grayscale images
def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32) / 255.0
    return torch.tensor(img)

# Visualize difference between H and Hp
def visualize_diff(H, Hp, idx):
    diff = torch.abs(H - Hp)
    amplified = diff * 10  # amplify difference
    amplified = torch.clamp(amplified, 0, 1)

    plt.figure(figsize=(12, 3))

    plt.subplot(1, 3, 1)
    plt.imshow(H, cmap='gray')
    plt.title("H (Clean High-pass)")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(Hp, cmap='gray')
    plt.title("Hp (Perturbed High-pass)")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(amplified, cmap='hot')
    plt.title("Amplified |H - Hp| × 10")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"diff_{idx:02d}.png"))
    plt.close()

# Process first 20 image pairs
h_files = sorted(os.listdir(h_dir))[:20]
hp_files = sorted(os.listdir(hp_dir))[:20]

for idx, (hf, hpf) in enumerate(zip(h_files, hp_files)):
    h_path = os.path.join(h_dir, hf)
    hp_path = os.path.join(hp_dir, hpf)

    H = load_image(h_path)
    Hp = load_image(hp_path)

    visualize_diff(H, Hp, idx)

print(" Difference visualizations saved in ./diff_visuals")
