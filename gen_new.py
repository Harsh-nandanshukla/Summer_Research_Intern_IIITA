import os
import torch
import torch.nn as nn
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from models.vit import ViT
from pgd import pgd_attack
from generator import UNetGenerator, load_model

def get_train_loader(data_path):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5070, 0.4865, 0.4409), (0.2675, 0.2565, 0.2761))
    ])
    train_dataset = datasets.CIFAR100(
        root=os.path.join(data_path, 'cifar-100'),
        train=True,
        transform=transform,
        download=True
    )
    return DataLoader(train_dataset, batch_size=64, shuffle=True)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load frozen ViT
    vit = load_model(args.vit_ckpt, device)
    vit.eval()
    for p in vit.parameters():
        p.requires_grad = False

    # Generator
    generator = UNetGenerator().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)

    train_loader = get_train_loader(args.data_path)

    batchwise_acc = []
    num_epochs = 100

    for epoch in range(1, num_epochs + 1):
        generator.train()
        correct = 0
        total = 0
        batch_acc = []

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Create adversarial image and perturbation
            Ip = pgd_attack(vit, images, torch.zeros_like(labels), eps=4/255, alpha=1/255, iters=3)
            P = Ip - images

            # Predict perturbation from Ip
            P_hat = generator(Ip)
            loss = criterion(P_hat, P)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Reconstruct clean image and evaluate
            I_prime = Ip - P_hat
            with torch.no_grad():
                logits = vit(I_prime)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                batch_acc.append(100.0 * (preds == labels).sum().item() / labels.size(0))

        epoch_acc = 100.0 * correct / total
        batchwise_acc.append(epoch_acc)
        print(f"[Epoch {epoch}/{num_epochs}] Accuracy on I': {epoch_acc:.2f}%")

    # Save generator weights
    torch.save(generator.state_dict(), "gen_new.pth")

    # Save accuracy CSV
    acc_df = pd.DataFrame({"epoch": list(range(1, num_epochs + 1)), "accuracy": batchwise_acc})
    acc_df.to_csv("new_train.csv", index=False)

    # Plot accuracy with average line
    avg_acc = sum(batchwise_acc) / len(batchwise_acc)
    plt.figure(figsize=(8, 5))
    plt.plot(acc_df['epoch'], acc_df['accuracy'], label='Accuracy per Epoch', color='blue')
    plt.axhline(y=avg_acc, color='red', linestyle='--', label=f'Avg Accuracy = {avg_acc:.2f}%')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Generator Training (Ip → P_hat → I' → ViT Accuracy)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("plot_new_train.png")
    plt.close()
    print("Saved model to gen_new.pth, CSV to new_train.csv, plot to plot_new_train.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./datasets/')
    parser.add_argument('--vit_ckpt', type=str, required=True)
    args = parser.parse_args()
    main(args)
