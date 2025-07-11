import os
import torch
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from models.vit import ViT
from pgd import pgd_attack
from generator import UNetGenerator, load_model

def get_test_loader(data_path):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5070, 0.4865, 0.4409), (0.2675, 0.2565, 0.2761))
    ])
    test_dataset = datasets.CIFAR100(
        root=os.path.join(data_path, 'cifar-100'),
        train=False,
        transform=transform,
        download=True
    )
    return DataLoader(test_dataset, batch_size=64, shuffle=False)

def evaluate(generator, vit, dataloader, device):
    generator.eval()
    vit.eval()
    for p in generator.parameters(): p.requires_grad = False
    for p in vit.parameters(): p.requires_grad = False

    clean_accs = []
    perturbed_accs = []
    best_clean, best_perturbed = 0, 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        # === Clean Path ===
        P1 = generator(images)
        I1 = images - P1
        preds_clean = torch.argmax(vit(I1), dim=1)
        acc_clean = 100.0 * (preds_clean == labels).sum().item() / labels.size(0)
        clean_accs.append(acc_clean)
        best_clean = max(best_clean, acc_clean)

        # === Perturbed Path ===
        Ip = pgd_attack(vit, images, torch.zeros_like(labels), eps=4/255, alpha=1/255, iters=3)
        P2 = generator(Ip)
        I2 = Ip - P2
        preds_perturbed = torch.argmax(vit(I2), dim=1)
        acc_perturbed = 100.0 * (preds_perturbed == labels).sum().item() / labels.size(0)
        perturbed_accs.append(acc_perturbed)
        best_perturbed = max(best_perturbed, acc_perturbed)

    avg_clean = sum(clean_accs) / len(clean_accs)
    avg_perturbed = sum(perturbed_accs) / len(perturbed_accs)
    return clean_accs, perturbed_accs, avg_clean, avg_perturbed, best_clean, best_perturbed

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load generator
    generator = UNetGenerator().to(device)
    generator.load_state_dict(torch.load(args.generator_ckpt, map_location=device))
    generator.eval()

    # Load frozen ViT
    vit = load_model(args.vit_ckpt, device)
    vit.eval()

    test_loader = get_test_loader(args.data_path)

    clean_accs, perturbed_accs, avg_clean, avg_perturbed, best_clean, best_perturbed = evaluate(
        generator, vit, test_loader, device
    )

    # Save CSV
    df = pd.DataFrame({
        'batch_idx': list(range(len(clean_accs))),
        'clean_restored_acc': clean_accs,
        'perturbed_restored_acc': perturbed_accs
    })
    df.to_csv("new_test.csv", index=False)
    print("Saved accuracy CSV: new_test.csv")

    # Plot 1: Line plot of batch-wise accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(df['batch_idx'], df['clean_restored_acc'], label="I - P1 (Clean)", color='green')
    plt.plot(df['batch_idx'], df['perturbed_restored_acc'], label="Ip - P2 (Perturbed)", color='orange')
    plt.axhline(avg_clean, linestyle='--', color='green', alpha=0.6, label=f"Avg Clean Acc = {avg_clean:.2f}%")
    plt.axhline(avg_perturbed, linestyle='--', color='orange', alpha=0.6, label=f"Avg Perturbed Acc = {avg_perturbed:.2f}%")
    plt.xlabel("Batch Index"); plt.ylabel("Accuracy (%)"); plt.title("Accuracy Over Batches")
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig("new_test_accuracy_over_batches_plot.png")
    plt.close()
    print("Saved plot: new_test_accuracy_over_batches_plot.png")

    # Plot 2: Bar chart of best accuracies
    plt.figure(figsize=(6, 5))
    bars = plt.bar(['Best Clean', 'Best Perturbed'], [best_clean, best_perturbed], color=['green', 'orange'])
    for bar, acc in zip(bars, [best_clean, best_perturbed]):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{acc:.2f}%", ha='center')
    plt.ylim(0, 100)
    plt.title("Best Accuracy (Among All Batches)")
    plt.ylabel("Accuracy (%)")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig("new_test_accuracy_bar_plot.png")
    plt.close()
    print("Saved plot: new_test_accuracy_bar_plot.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./datasets/')
    parser.add_argument('--generator_ckpt', type=str, default='gen_new.pth')
    parser.add_argument('--vit_ckpt', type=str, required=True)
    args = parser.parse_args()
    main(args)
