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

# @torch.no_grad()
def evaluate(generator, vit, dataloader, device, use_perturbation):
    generator.eval()
    vit.eval()

    per_batch_acc = []
    total_correct, total_samples = 0, 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        if use_perturbation:
            images = pgd_attack(vit, images, torch.zeros_like(labels), eps=4/255, alpha=1/255, iters=3)

        restored_images = generator(images)
        logits = vit(restored_images)
        preds = torch.argmax(logits, dim=1)

        correct = (preds == labels).sum().item()
        total = labels.size(0)

        per_batch_acc.append(100.0 * correct / total)
        total_correct += correct
        total_samples += total

    avg_acc = 100.0 * total_correct / total_samples
    return avg_acc, per_batch_acc

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    lambda_tag = os.path.basename(args.generator_ckpt).replace("generator_", "").replace(".pth", "")
    is_real = lambda_tag.startswith("real_")

    # Load Generator
    generator = UNetGenerator().to(device)
    generator.load_state_dict(torch.load(args.generator_ckpt, map_location=device))
    generator.eval()

    # Load Frozen ViT
    vit = load_model(args.vit_ckpt, device)
    vit.eval()

    test_loader = get_test_loader(args.data_path)

    print("\n→ Evaluating on Clean Test Images (through Generator)...")
    avg_clean, accs_clean = evaluate(generator, vit, test_loader, device, use_perturbation=False)
    print(f" Accuracy (Clean → Generator → ViT): {avg_clean:.2f}%")

    print("\n→ Evaluating on PGD Test Images (through Generator)...")
    avg_pgd, accs_pgd = evaluate(generator, vit, test_loader, device, use_perturbation=True)
    print(f" Accuracy (PGD → Generator → ViT): {avg_pgd:.2f}%")

    # === Save Per-Batch Accuracies ===
    per_batch_df = pd.DataFrame({
        'batch_idx': list(range(len(accs_clean))),
        'clean_acc': accs_clean,
        'pgd_acc': accs_pgd
    })
    per_batch_csv = f"pgd_test_accuracy_per_batch_{lambda_tag}.csv"
    per_batch_df.to_csv(per_batch_csv, index=False)
    print(f"Saved per-batch accuracies to: {per_batch_csv}")

    # === Save Summary ===
    summary_csv = f"test_accuracies_{lambda_tag}.csv"
    summary_df = pd.DataFrame([{
        'clean_restored': avg_clean,
        'pgd_restored': avg_pgd
    }])
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved average accuracies to: {summary_csv}")

    # === Plot Curves ===
    os.makedirs(f"plots_{lambda_tag}", exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(per_batch_df['batch_idx'], per_batch_df['clean_acc'], label='Clean→Gen→ViT', color='blue')
    plt.plot(per_batch_df['batch_idx'], per_batch_df['pgd_acc'], label='PGD→Gen→ViT', color='orange')
    plt.xlabel("Batch Index")
    plt.ylabel("Accuracy (%)")

  
    #  (supports any prefix like 'real_' or 'cleanreal_')
    parts = lambda_tag.split('_')
    l1, l2, l3 = parts[-3:]  # get last 3 elements always

    l1, l2, l3 = l1.replace('p', '.'), l2.replace('p', '.'), l3.replace('p', '.')

    plt.title(f"Per-Batch Accuracy (λ1={l1}, λ2={l2}, λ3={l3})")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"plots_{lambda_tag}/batchwise_accuracy_plot.png")
    plt.close()
    print(f"Saved batchwise accuracy plot to: plots_{lambda_tag}/batchwise_accuracy_plot.png")

    # === Bar Chart for Summary ===
    plt.figure(figsize=(6, 5))
    bars = plt.bar(['Clean→Gen→ViT', 'PGD→Gen→ViT'], [avg_clean, avg_pgd], color=['blue', 'orange'])
    for bar, val in zip(bars, [avg_clean, avg_pgd]):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{val:.2f}%", ha='center')
    plt.ylim(0, 100)
    plt.ylabel("Accuracy (%)")
    plt.title("Average Accuracy Comparison")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(f"plots_{lambda_tag}/avg_accuracy_bar_plot.png")
    plt.close()
    print(f"Saved average accuracy bar plot to: plots_{lambda_tag}/avg_accuracy_bar_plot.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./datasets/')
    parser.add_argument('--generator_ckpt', type=str, required=True)
    parser.add_argument('--vit_ckpt', type=str, required=True)
    args = parser.parse_args()
    main(args)


# import os
# import torch
# import argparse
# import pandas as pd
# import matplotlib.pyplot as plt
# from torch.utils.data import DataLoader
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
# from models.vit import ViT
# from pgd import pgd_attack
# from generator import UNetGenerator, load_model

# def get_test_loader(data_path):
#     transform = transforms.Compose([
#         transforms.Resize((32, 32)),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5070, 0.4865, 0.4409), (0.2675, 0.2565, 0.2761))
#     ])
#     test_dataset = datasets.CIFAR100(
#         root=os.path.join(data_path, 'cifar-100'),
#         train=False,
#         transform=transform,
#         download=True
#     )
#     return DataLoader(test_dataset, batch_size=64, shuffle=False)

# # @torch.no_grad()
# def evaluate(generator, vit, dataloader, device, use_perturbation):
#     generator.eval()
#     vit.eval()

#     per_batch_acc = []
#     total_correct, total_samples = 0, 0

#     for images, labels in dataloader:
#         images, labels = images.to(device), labels.to(device)

#         if use_perturbation:
#             images = pgd_attack(vit, images, torch.zeros_like(labels), eps=4/255, alpha=1/255, iters=3)

#         restored_images = generator(images)
#         logits = vit(restored_images)
#         preds = torch.argmax(logits, dim=1)

#         correct = (preds == labels).sum().item()
#         total = labels.size(0)

#         per_batch_acc.append(100.0 * correct / total)
#         total_correct += correct
#         total_samples += total

#     avg_acc = 100.0 * total_correct / total_samples
#     return avg_acc, per_batch_acc

# def main(args):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Device: {device}")

#     lambda_tag = os.path.basename(args.generator_ckpt).replace("generator_", "").replace(".pth", "")

#     generator = UNetGenerator().to(device)
#     generator.load_state_dict(torch.load(args.generator_ckpt, map_location=device))
#     generator.eval()

#     vit = load_model(args.vit_ckpt, device)
#     vit.eval()

#     test_loader = get_test_loader(args.data_path)

#     print("\n→ Evaluating on Clean Test Images (through Generator)...")
#     avg_clean, accs_clean = evaluate(generator, vit, test_loader, device, use_perturbation=False)
#     print(f" Accuracy (Clean → Generator → ViT): {avg_clean:.2f}%")

#     print("\n→ Evaluating on PGD Test Images (through Generator)...")
#     avg_pgd, accs_pgd = evaluate(generator, vit, test_loader, device, use_perturbation=True)
#     print(f" Accuracy (PGD → Generator → ViT): {avg_pgd:.2f}%")

#     # === Save Per-Batch Accuracies ===
#     per_batch_df = pd.DataFrame({
#         'batch_idx': list(range(len(accs_clean))),
#         'clean_acc': accs_clean,
#         'pgd_acc': accs_pgd
#     })
#     per_batch_csv = f"pgd_test_accuracy_per_batch_{lambda_tag}.csv"
#     per_batch_df.to_csv(per_batch_csv, index=False)
#     print(f"Saved per-batch accuracies to: {per_batch_csv}")

#     # === Save Summary ===
#     summary_csv = f"test_accuracies_{lambda_tag}.csv"
#     summary_df = pd.DataFrame([{
#         'clean_restored': avg_clean,
#         'pgd_restored': avg_pgd
#     }])
#     summary_df.to_csv(summary_csv, index=False)
#     print(f"Saved average accuracies to: {summary_csv}")

#     # === Plot Curves ===
#     os.makedirs(f"plots_{lambda_tag}", exist_ok=True)
#     plt.figure(figsize=(8, 5))
#     plt.plot(per_batch_df['batch_idx'], per_batch_df['clean_acc'], label='Clean→Gen→ViT', color='blue')
#     plt.plot(per_batch_df['batch_idx'], per_batch_df['pgd_acc'], label='PGD→Gen→ViT', color='orange')
#     plt.xlabel("Batch Index")
#     plt.ylabel("Accuracy (%)")
#     l1, l2, l3 = lambda_tag.split('_')
#     l1, l2, l3 = l1.replace('p', '.'), l2.replace('p', '.'), l3.replace('p', '.')
#     plt.title(f"Per-Batch Accuracy (λ1={l1}, λ2={l2}, λ3={l3})")
#     #plt.title(f"Per-Batch Accuracy (λ1={lambda_tag[0]}, λ2={lambda_tag[2]}, λ3={lambda_tag[4]})")
#     plt.legend()
#     plt.grid()
#     plt.tight_layout()
#     plt.savefig(f"plots_{lambda_tag}/batchwise_accuracy_plot.png")
#     plt.close()
#     print(f"Saved batchwise accuracy plot to: plots_{lambda_tag}/batchwise_accuracy_plot.png")

#     # === Bar Chart for Summary ===
#     plt.figure(figsize=(6, 5))
#     bars = plt.bar(['Clean→Gen→ViT', 'PGD→Gen→ViT'], [avg_clean, avg_pgd], color=['blue', 'orange'])
#     for bar, val in zip(bars, [avg_clean, avg_pgd]):
#         plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{val:.2f}%", ha='center')
#     plt.ylim(0, 100)
#     plt.ylabel("Accuracy (%)")
#     plt.title("Average Accuracy Comparison")
#     plt.grid(axis='y')
#     plt.tight_layout()
#     plt.savefig(f"plots_{lambda_tag}/avg_accuracy_bar_plot.png")
#     plt.close()
#     print(f"Saved average accuracy bar plot to: plots_{lambda_tag}/avg_accuracy_bar_plot.png")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data_path', type=str, default='./datasets/')
#     parser.add_argument('--generator_ckpt', type=str, required=True)
#     parser.add_argument('--vit_ckpt', type=str, required=True)
#     args = parser.parse_args()
#     main(args)
