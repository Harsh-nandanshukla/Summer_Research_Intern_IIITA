

import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from pgd import pgd_attack
from pgd_eval import load_model  # loads ViT model with weights

# === Settings ===
DATA_PATH = './datasets/'
CHECKPOINT_PATH = './CheckpointsResults/vit--CIFAR100-exp0/best.pth'
SAVE_DIR = './perturbed_images/'
NUM_SAMPLES = 20  # Number of test images to visualize


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load model ===
vit = load_model(CHECKPOINT_PATH, device)
vit.eval()

# === Load CIFAR-100 test images ===
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])
test_dataset = datasets.CIFAR100(root=os.path.join(DATA_PATH, 'cifar-100'), train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=NUM_SAMPLES, shuffle=True)

# === Get one random batch ===
images, labels = next(iter(test_loader))
images = images.to(device)
labels = labels.to(device)

# === Apply PGD ===
adv_images = pgd_attack(vit, images, labels, eps=4/255, alpha=1/255, iters=3)

# === Ensure output directory ===
os.makedirs(SAVE_DIR, exist_ok=True)

# === Save both image sets ===
def save_images(tensor, filename, title):
    tensor = tensor.detach().cpu()
    grid = vutils.make_grid(tensor, nrow=NUM_SAMPLES, normalize=True)
    plt.figure(figsize=(16, 2))
    plt.imshow(grid.permute(1, 2, 0))
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, filename))
    plt.close()

save_images(images, 'original.png', 'Original CIFAR-100 Test Images')
save_images(adv_images, 'perturbed.png', 'PGD-Perturbed Images')

print(" Images saved in folder:", SAVE_DIR)
