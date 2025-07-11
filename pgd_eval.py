import argparse
import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from models.vit import ViT  # Adjust import if needed
from pgd import pgd_attack
import pandas as pd
import matplotlib.pyplot as plt


def load_model(checkpoint_path, device):
    model = ViT(
        img_size=32,
        patch_size=4,
        num_classes=100,
        dim=192,
        depth=9,
        heads=6,
        dim_head=32,
        mlp_dim_ratio=2.0,
        dropout=0.1,
        emb_dropout=0.1,
        stochastic_depth=0.1,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def evaluate(model, data_loader, device, is_train=False):
    total = 0
    correct = 0
    batch_acc_list = []

    print("Starting PGD evaluation loop...")
    for batch_idx, (images, labels) in enumerate(data_loader):
        try:
            images, labels = images.to(device), labels.to(device)

            # PGD attack (moderate settings)
            adv_images = pgd_attack(model, images, labels)

            outputs = model(adv_images)
            _, preds = torch.max(outputs, 1)

            batch_correct = (preds == labels).sum().item()
            batch_total = labels.size(0)

            correct += batch_correct
            total += batch_total

            batch_acc = 100.0 * batch_correct / batch_total
            batch_acc_list.append(batch_acc)

            print(f"🔹 Batch {batch_idx+1}: Accuracy = {batch_acc:.2f}% | Total evaluated = {total}")

        except Exception as e:
            print(f"Error in batch {batch_idx+1}: {e}")

    final_acc = 100.0 * correct / total if total > 0 else 0.0
    mean_batch_acc = sum(batch_acc_list) / len(batch_acc_list) if batch_acc_list else 0.0

    print(f'\n Final Accuracy under PGD attack: {final_acc:.2f}%')
    print(f' Average Accuracy across batches: {mean_batch_acc:.2f}%')

    # Save CSV
    tag = "train" if is_train else "test"
    csv_name = f"pgd_{tag}_accuracy_per_batch.csv"
    df = pd.DataFrame({'Batch': list(range(1, len(batch_acc_list)+1)), 'Accuracy': batch_acc_list})
    df.to_csv(csv_name, index=False)
    print(f" Saved accuracy CSV: {csv_name}")

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(df['Batch'], df['Accuracy'], marker='o', label='Batch Accuracy')
    plt.axhline(y=mean_batch_acc, color='red', linestyle='--', label=f'Mean Accuracy = {mean_batch_acc:.2f}%')
    plt.xlabel("Batch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"PGD Accuracy per Batch on {tag.capitalize()} Set\nMean = {mean_batch_acc:.2f}%")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plot_name = f"pgd_{tag}_accuracy_plot.png"
    plt.savefig(plot_name)
    print(f" Saved plot: {plot_name}")

    return mean_batch_acc


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5070, 0.4865, 0.4409),
                             (0.2675, 0.2565, 0.2761))
    ])

    if args.use_train:
        dataset = datasets.CIFAR100(
            root=os.path.join(args.data_path, 'cifar-100'),
            train=True, transform=transform, download=True
        )
        print(" Using CIFAR-100 Training Data for PGD Evaluation")
    else:
        dataset = datasets.CIFAR100(
            root=os.path.join(args.data_path, 'cifar-100'),
            train=False, transform=transform, download=True
        )
        print(" Using CIFAR-100 Test Data for PGD Evaluation")

    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    model = load_model(args.checkpoint, device)

    avg_acc = evaluate(model, loader, device, is_train=args.use_train)
    print(f"\n Evaluation Complete: Average Batch Accuracy = {avg_acc:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./datasets/',
                        help='Path to CIFAR-100 dataset')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model .pth file')
    parser.add_argument('--use_train', action='store_true',
                        help='Evaluate on training data instead of test data')
    args = parser.parse_args()

    main(args)


# import argparse
# import os
# import torch
# from torch.utils.data import DataLoader
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
# from models.vit import ViT  # Update the import path if needed
# from pgd import pgd_attack

# def load_model(checkpoint_path, device):
#     model = ViT(
#         img_size=32,
#         patch_size=4,
#         num_classes=100,
#         dim=192,
#         depth=9,
#         heads=6,
#         dim_head=32,
#         mlp_dim_ratio=2.0,  # because 384 / 192 = 2
#         dropout=0.1,
#         emb_dropout=0.1,
#         stochastic_depth=0.1,
       
#     )
#     checkpoint = torch.load(checkpoint_path, map_location=device)
#     model.load_state_dict(checkpoint["model_state_dict"])
#     model.to(device)
#     model.eval()
#     return model



# def evaluate(model, data_loader, device):
#     total = 0
#     correct = 0

#     print("Starting PGD evaluation loop...")
#     for batch_idx, (images, labels) in enumerate(data_loader):
#         try:
#             images, labels = images.to(device), labels.to(device)

#             # PGD attack
#             adv_images = pgd_attack(model, images, labels, eps=8/255, alpha=2/255, iters=10)

#             outputs = model(adv_images)
#             _, preds = torch.max(outputs, 1)

#             correct += (preds == labels).sum().item()
#             total += labels.size(0)

#             print(f"🔹 Batch {batch_idx+1}: Total evaluated = {total}")

#         except Exception as e:
#             print(f" Error in batch {batch_idx+1}: {e}")

#     if total > 0:
#         acc = 100 * correct / total
#         print(f'\n Accuracy under PGD attack: {acc:.2f}%')
#     else:
#         print(" No samples were evaluated.")



# def main(args):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     # Transform for CIFAR-100 test set
#     transform = transforms.Compose([
#         transforms.Resize((32, 32)),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5070, 0.4865, 0.4409),
#                              (0.2675, 0.2565, 0.2761))
#     ])

#     # Load CIFAR-100 test dataset
#     # test_dataset = datasets.CIFAR100(
#     #     root=os.path.join(args.data_path, 'cifar-100'),
#     #     train=False, transform=transform, download=True
#     # )

#     # test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
#     # # Load CIFAR-100 training dataset
#     train_dataset = datasets.CIFAR100(
#     root=os.path.join(args.data_path, 'cifar-100'),
#     train=True, transform=transform, download=True
#     )
#     train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)


    

#     # Load trained model
#     model = load_model(args.checkpoint, device)

#     # Evaluate model under PGD attack
#     evaluate(model, train_loader, device)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data_path', type=str, default='./datasets/',
#                         help='Path to CIFAR-100 dataset')
#     parser.add_argument('--checkpoint', type=str, required=True,
#                         help='Path to the trained model .pth file')
#     args = parser.parse_args()

#     main(args)
