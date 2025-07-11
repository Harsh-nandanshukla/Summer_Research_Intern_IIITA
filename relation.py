import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import cv2
import seaborn as sns
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from pgd import pgd_attack
from pgd_eval import load_model

# === Filters ===
def low_pass(img_tensor):
    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
    img_np = cv2.GaussianBlur(img_np, (5, 5), sigmaX=1)
    return torch.tensor(img_np).permute(2, 0, 1)

def high_pass(img_tensor):
    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
    gray = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap = np.clip((lap - lap.min()) / (lap.max() - lap.min() + 1e-8), 0, 1)
    return torch.tensor(lap).unsqueeze(0)

def to_grayscale(t):
    return t.mean(dim=0, keepdim=True)

def pearson_corr(x, y):
    x_flat = x.reshape(-1).double()
    y_flat = y.reshape(-1).double()
    vx = x_flat - x_flat.mean()
    vy = y_flat - y_flat.mean()
    return (vx @ vy) / (vx.norm() * vy.norm() + 1e-8)

# === Visualization Functions ===
def save_hist(data, title, fname, color, bins=1000):
    plt.figure(figsize=(8, 4))
    plt.hist(data, bins=bins, color=color, alpha=0.7, edgecolor='black')
    plt.xlabel("Pearson Correlation")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join("correlation_visuals", fname))
    plt.close()
    print(f"Saved histogram: correlation_visuals/{fname}")

def save_kde_comparison(data_dict, fname, xlim, title):
    plt.figure(figsize=(10, 5))
    for label, (data, color) in data_dict.items():
        sns.kdeplot(data, label=label, color=color, linewidth=2)
    plt.xlabel("Pearson Correlation")
    plt.ylabel("Density")
    plt.title(title)
    plt.xlim(xlim)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join("correlation_visuals", fname))
    plt.close()
    print(f"Saved KDE comparison: correlation_visuals/{fname}")

# === Main Function ===
def main(args):
    os.makedirs("correlation_visuals", exist_ok=True)
    os.makedirs("examples_H", exist_ok=True)
    os.makedirs("examples_Hp", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    dataset = datasets.CIFAR100(
        root=os.path.join(args.data_path, 'cifar-100'),
        train=False, transform=transform, download=True
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    vit = load_model(args.vit_ckpt, device)
    vit.eval()
    for p in vit.parameters():
        p.requires_grad = False

    corr_LH, corr_LpHp, corr_LLp, corr_HHp = [], [], [], []
    mse_list = []  #  NEW: list to store MSE
    total_samples = 0
    saved_H, saved_Hp = 0, 0

    # (unchanged parts above...)

    mse_HHp = []  # 🔹 Store MSE between H and Hp for all 10k samples

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        perturbed = pgd_attack(vit, images, labels, eps=4/255, alpha=1/255, iters=3)

        for i in range(images.size(0)):
            I = images[i]
            Ip = perturbed[i]

            L = low_pass(I)
            H = high_pass(I)
            Lp = low_pass(Ip)
            Hp = high_pass(Ip)

            # Correlations
            c1 = pearson_corr(to_grayscale(L), H).item()
            c2 = pearson_corr(to_grayscale(Lp), Hp).item()
            c3 = pearson_corr(L, Lp).item()
            c4 = pearson_corr(H, Hp).item()

            corr_LH.append(c1)
            corr_LpHp.append(c2)
            corr_LLp.append(c3)
            corr_HHp.append(c4)

            # MSE for all H-Hp
            mse = torch.nn.functional.mse_loss(H, Hp).item()
            mse_HHp.append(mse)

            # Save 20 examples
            if saved_H < 20:
                save_image(H, f"examples_H/H_{saved_H:02d}.png")
                saved_H += 1
            if saved_Hp < 20:
                save_image(Hp, f"examples_Hp/Hp_{saved_Hp:02d}.png")
                saved_Hp += 1

            total_samples += 1
            if total_samples >= 10000:
                break
        if total_samples >= 10000:
            break

    # === Save histograms
    save_hist(corr_LH, "L vs H (Clean)", "corr_L_H_clean.png", "green")
    save_hist(corr_LpHp, "Lp vs Hp (Perturbed)", "corr_Lp_Hp_perturbed.png", "orange")
    save_hist(corr_LLp, "L vs Lp (Low-pass: Clean vs Perturbed)", "corr_L_Lp.png", "blue")
    save_hist(corr_HHp, "H vs Hp (High-pass: Clean vs Perturbed)", "corr_H_Hp.png", "red")

    # === Save KDE comparisons
    save_kde_comparison({
        "L vs Lp (Low-pass)": (corr_LLp, "blue"),
        "H vs Hp (High-pass)": (corr_HHp, "red"),
    }, "kde_zoom_high_corr.png", xlim=(0.9, 1.0),
       title="KDE: L vs Lp and H vs Hp (Zoomed: 0.9–1.0)")

    save_kde_comparison({
        "L vs H (Clean)": (corr_LH, "green"),
        "Lp vs Hp (Perturbed)": (corr_LpHp, "orange"),
    }, "kde_zoom_low_corr.png", xlim=(-0.4, -0.1),
       title="KDE: L vs H and Lp vs Hp (Zoomed: -0.4 to -0.1)")

    # === Plot MSE values for all 10k samples
    mean_mse = np.mean(mse_HHp)

    plt.figure(figsize=(10, 4))
    plt.plot(mse_HHp, color='purple', linewidth=1)
    plt.axhline(mean_mse, color='black', linestyle='--', label=f"Mean MSE = {mean_mse:.6f}")
    plt.title("MSE between H and Hp (10,000 samples)")
    plt.xlabel("Sample Index")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("correlation_visuals/mse_H_Hp_plot.png")
    plt.close()

    print(f"\ Mean MSE between H and Hp: {mean_mse:.6f}")
    print(" MSE plot saved: correlation_visuals/mse_H_Hp_plot.png")

# --- End of main() ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./datasets/')
    parser.add_argument('--vit_ckpt', type=str, required=True)
    args = parser.parse_args()
    main(args)
