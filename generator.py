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
from pgd_eval import load_model

# === U-Net Generator ===
class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.enc1 = self.down_block(in_channels, 64)
        self.enc2 = self.down_block(64, 128)
        self.enc3 = self.down_block(128, 256)
        self.enc4 = self.down_block(256, 512)
        self.bottleneck = nn.Sequential(nn.Conv2d(512, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.ReLU(True))
        self.dec4 = self.up_block(512, 512)
        self.dec3 = self.up_block(1024, 256)
        self.dec2 = self.up_block(512, 128)
        self.dec1 = self.up_block(256, 64)
        self.final = nn.Sequential(nn.ConvTranspose2d(128, out_channels, 4, 2, 1), nn.Tanh())

    def down_block(self, in_c, out_c):
        return nn.Sequential(nn.Conv2d(in_c, out_c, 4, 2, 1), nn.BatchNorm2d(out_c), nn.LeakyReLU(0.2, inplace=True))

    def up_block(self, in_c, out_c):
        return nn.Sequential(nn.ConvTranspose2d(in_c, out_c, 4, 2, 1), nn.BatchNorm2d(out_c), nn.ReLU(True))

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b = self.bottleneck(e4)
        d4 = torch.cat([self.dec4(b), e4], dim=1)
        d3 = torch.cat([self.dec3(d4), e3], dim=1)
        d2 = torch.cat([self.dec2(d3), e2], dim=1)
        d1 = torch.cat([self.dec1(d2), e1], dim=1)
        return self.final(d1)

# === Discriminator ===
class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 1, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 1)
        )

    def forward(self, x):
        return self.model(x)

# === Annotate Plot ===
def annotate_plot(ax, epochs, values):
    ax.annotate(f"{values.iloc[0]:.4f}", xy=(epochs.iloc[0], values.iloc[0]), xytext=(epochs.iloc[0], values.iloc[0] + 0.02), color="green")
    ax.annotate(f"{values.iloc[-1]:.4f}", xy=(epochs.iloc[-1], values.iloc[-1]), xytext=(epochs.iloc[-1], values.iloc[-1] + 0.02), color="red")

# === Main Training ===
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5070, 0.4865, 0.4409), (0.2675, 0.2565, 0.2761))
    ])
    dataset = datasets.CIFAR100(root=os.path.join(args.data_path, 'cifar-100'), train=True, transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    generator = UNetGenerator().to(device)
    discriminator = Discriminator().to(device)
    frozen_vit = load_model(args.checkpoint, device)
    frozen_vit.eval()
    for param in frozen_vit.parameters():
        param.requires_grad = False

    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_mse = nn.MSELoss()
    criterion_ce = nn.CrossEntropyLoss()

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.9, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

    λ1, λ2, λ3 = 1.0, 1.0, 1.0
    tag = f"cleanreal_{λ1:.1f}_{λ2:.1f}_{λ3:.1f}".replace('.', 'p')
    log = {key: [] for key in ['Epoch', 'total_loss', 'adversarial_loss', 'mse_loss', 'ce_loss', 'generator_loss', 'discriminator_loss']}

    for epoch in range(1, 101):
        total_adv = total_mse = total_ce = 0.0
        total_gen_loss = total_disc_loss = total_combined_loss = 0.0
        num_batches = 0

        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            num_batches += 1

            # === Generator inputs: clean + perturbed ===
            perturbed = pgd_attack(frozen_vit, images, torch.zeros_like(labels), eps=4/255, alpha=1/255, iters=3)
            all_inputs = torch.cat([images, perturbed], dim=0)
            all_labels = torch.cat([labels, labels], dim=0)
            all_targets = torch.cat([images, images], dim=0)

            fake_images = generator(all_inputs)

            # === Discriminator update ===
            if num_batches % 2 == 0:
                real_out = discriminator(images)  # 🔧 ONLY CLEAN IMAGES as real
                fake_out = discriminator(fake_images.detach())
                loss_D_real = criterion_gan(real_out, torch.ones_like(real_out).uniform_(0.8, 1.0).to(device))
                loss_D_fake = criterion_gan(fake_out, torch.zeros_like(fake_out).uniform_(0.0, 0.2).to(device))
                loss_disc = 0.5 * (loss_D_real + loss_D_fake)
                optimizer_D.zero_grad()
                loss_disc.backward()
                optimizer_D.step()
                total_disc_loss += loss_disc.item()

            # === Generator update ===
            pred_fake = discriminator(fake_images)
            adv_loss = criterion_gan(pred_fake, torch.ones_like(pred_fake))
            mse_loss = criterion_mse(fake_images, all_targets)
            ce_loss = criterion_ce(frozen_vit(fake_images), all_labels)

            generator_loss = λ1 * adv_loss + λ2 * mse_loss
            total_loss = generator_loss + λ3 * ce_loss

            optimizer_G.zero_grad()
            total_loss.backward()
            optimizer_G.step()

            total_adv += adv_loss.item()
            total_mse += mse_loss.item()
            total_ce += ce_loss.item()
            total_gen_loss += generator_loss.item()
            total_combined_loss += total_loss.item()

        avg = lambda x: x / num_batches
        log['Epoch'].append(epoch)
        log['adversarial_loss'].append(avg(total_adv))
        log['mse_loss'].append(avg(total_mse))
        log['ce_loss'].append(avg(total_ce))
        log['generator_loss'].append(avg(total_gen_loss))
        log['total_loss'].append(avg(total_combined_loss))
        log['discriminator_loss'].append(total_disc_loss / max(1, num_batches // 2))

        print(f"[Epoch {epoch}/100]")
        print(f"  Generator Loss    : {avg(total_gen_loss):.4f}")
        print(f"  Total Loss        : {avg(total_combined_loss):.4f}")
        print(f"    ├── Adversarial : {avg(total_adv):.4f}")
        print(f"    ├── MSE         : {avg(total_mse):.4f}")
        print(f"    └── CE          : {avg(total_ce):.4f}")
        print(f"  Discriminator Loss: {log['discriminator_loss'][-1]:.4f}")

    # === Save logs and plots ===
    df = pd.DataFrame(log)
    df.to_csv(f"loss_log_{tag}.csv", index=False)
    torch.save(generator.state_dict(), f"generator_{tag}.pth")

    os.makedirs(f"plots_{tag}", exist_ok=True)
    for key in ['total_loss', 'adversarial_loss', 'mse_loss', 'ce_loss']:
        plt.figure()
        plt.plot(df['Epoch'], df[key], label=key.replace("_", " ").capitalize())
        annotate_plot(plt.gca(), df['Epoch'], df[key])
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title(key.replace("_", " ").capitalize())
        plt.legend(); plt.grid(); plt.savefig(f"plots_{tag}/{key}.png"); plt.close()

    plt.figure()
    plt.plot(df['Epoch'], df['generator_loss'], label='Generator Loss')
    plt.plot(df['Epoch'], df['discriminator_loss'], label='Discriminator Loss')
    annotate_plot(plt.gca(), df['Epoch'], df['generator_loss'])
    annotate_plot(plt.gca(), df['Epoch'], df['discriminator_loss'])
    plt.title("Generator vs Discriminator Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid()
    plt.savefig(f"plots_{tag}/gen_vs_disc.png"); plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./datasets/')
    parser.add_argument('--checkpoint', type=str, required=True)
    args = parser.parse_args()
    main(args)


# Both clean training and perturbed images will be send to generator as input and also to discriminator as ground truth (real labeled)

# import os
# import torch
# import torch.nn as nn
# import argparse
# import pandas as pd
# import matplotlib.pyplot as plt
# from torch.utils.data import DataLoader
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
# from models.vit import ViT
# from pgd import pgd_attack
# from pgd_eval import load_model

# # === U-Net Generator ===
# class UNetGenerator(nn.Module):
#     def __init__(self, in_channels=3, out_channels=3):
#         super().__init__()
#         self.enc1 = self.down_block(in_channels, 64)
#         self.enc2 = self.down_block(64, 128)
#         self.enc3 = self.down_block(128, 256)
#         self.enc4 = self.down_block(256, 512)
#         self.bottleneck = nn.Sequential(nn.Conv2d(512, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.ReLU(True))
#         self.dec4 = self.up_block(512, 512)
#         self.dec3 = self.up_block(1024, 256)
#         self.dec2 = self.up_block(512, 128)
#         self.dec1 = self.up_block(256, 64)
#         self.final = nn.Sequential(nn.ConvTranspose2d(128, out_channels, 4, 2, 1), nn.Tanh())

#     def down_block(self, in_c, out_c):
#         return nn.Sequential(nn.Conv2d(in_c, out_c, 4, 2, 1), nn.BatchNorm2d(out_c), nn.LeakyReLU(0.2, inplace=True))

#     def up_block(self, in_c, out_c):
#         return nn.Sequential(nn.ConvTranspose2d(in_c, out_c, 4, 2, 1), nn.BatchNorm2d(out_c), nn.ReLU(True))

#     def forward(self, x):
#         e1 = self.enc1(x)
#         e2 = self.enc2(e1)
#         e3 = self.enc3(e2)
#         e4 = self.enc4(e3)
#         b = self.bottleneck(e4)
#         d4 = torch.cat([self.dec4(b), e4], dim=1)
#         d3 = torch.cat([self.dec3(d4), e3], dim=1)
#         d2 = torch.cat([self.dec2(d3), e2], dim=1)
#         d1 = torch.cat([self.dec1(d2), e1], dim=1)
#         return self.final(d1)

# # === Discriminator ===
# class Discriminator(nn.Module):
#     def __init__(self, in_channels=3):
#         super().__init__()
#         self.model = nn.Sequential(
#             nn.Conv2d(in_channels, 64, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(256, 512, 4, 1, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(512, 1, 4, 1, 1)
#         )

#     def forward(self, x):
#         return self.model(x)

# # === Annotate Plot ===
# def annotate_plot(ax, epochs, values):
#     ax.annotate(f"{values.iloc[0]:.4f}", xy=(epochs.iloc[0], values.iloc[0]), xytext=(epochs.iloc[0], values.iloc[0] + 0.02), color="green")
#     ax.annotate(f"{values.iloc[-1]:.4f}", xy=(epochs.iloc[-1], values.iloc[-1]), xytext=(epochs.iloc[-1], values.iloc[-1] + 0.02), color="red")

# # === Main Training ===
# def main(args):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     transform = transforms.Compose([
#         transforms.Resize((32, 32)),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5070, 0.4865, 0.4409), (0.2675, 0.2565, 0.2761))
#     ])
#     dataset = datasets.CIFAR100(root=os.path.join(args.data_path, 'cifar-100'), train=True, transform=transform, download=True)
#     dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

#     generator = UNetGenerator().to(device)
#     discriminator = Discriminator().to(device)
#     frozen_vit = load_model(args.checkpoint, device)
#     frozen_vit.eval()
#     for param in frozen_vit.parameters():
#         param.requires_grad = False

#     criterion_gan = nn.BCEWithLogitsLoss()
#     criterion_mse = nn.MSELoss()
#     criterion_ce = nn.CrossEntropyLoss()

#     optimizer_G = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.9, 0.999))
#     optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

#     λ1, λ2, λ3 = 1.0, 1.0, 1.0
#     tag = f"real_{λ1:.1f}_{λ2:.1f}_{λ3:.1f}".replace('.', 'p')
#     log = {key: [] for key in ['Epoch', 'total_loss', 'adversarial_loss', 'mse_loss', 'ce_loss', 'generator_loss', 'discriminator_loss']}

#     for epoch in range(1, 101):
#         total_adv = total_mse = total_ce = 0.0
#         total_gen_loss = total_disc_loss = total_combined_loss = 0.0
#         num_batches = 0

#         for images, labels in dataloader:
#             images, labels = images.to(device), labels.to(device)
#             num_batches += 1

#             # === Generator inputs: clean + perturbed ===
#             perturbed = pgd_attack(frozen_vit, images, torch.zeros_like(labels), eps=4/255, alpha=1/255, iters=3)
#             all_inputs = torch.cat([images, perturbed], dim=0)
#             all_labels = torch.cat([labels, labels], dim=0)
#             all_targets = torch.cat([images, images], dim=0)

#             fake_images = generator(all_inputs)

#             # === Discriminator update ===
#             if num_batches % 2 == 0:
#                 real_out = discriminator(all_inputs)
#                 fake_out = discriminator(fake_images.detach())
#                 loss_D_real = criterion_gan(real_out, torch.ones_like(real_out).uniform_(0.8, 1.0).to(device))
#                 loss_D_fake = criterion_gan(fake_out, torch.zeros_like(fake_out).uniform_(0.0, 0.2).to(device))
#                 loss_disc = 0.5 * (loss_D_real + loss_D_fake)
#                 optimizer_D.zero_grad()
#                 loss_disc.backward()
#                 optimizer_D.step()
#                 total_disc_loss += loss_disc.item()

#             # === Generator update ===
#             pred_fake = discriminator(fake_images)
#             adv_loss = criterion_gan(pred_fake, torch.ones_like(pred_fake))
#             mse_loss = criterion_mse(fake_images, all_targets)
#             ce_loss = criterion_ce(frozen_vit(fake_images), all_labels)

#             generator_loss = λ1 * adv_loss + λ2 * mse_loss
#             total_loss = generator_loss + λ3 * ce_loss

#             optimizer_G.zero_grad()
#             total_loss.backward()
#             optimizer_G.step()

#             total_adv += adv_loss.item()
#             total_mse += mse_loss.item()
#             total_ce += ce_loss.item()
#             total_gen_loss += generator_loss.item()
#             total_combined_loss += total_loss.item()

#         avg = lambda x: x / num_batches
#         log['Epoch'].append(epoch)
#         log['adversarial_loss'].append(avg(total_adv))
#         log['mse_loss'].append(avg(total_mse))
#         log['ce_loss'].append(avg(total_ce))
#         log['generator_loss'].append(avg(total_gen_loss))
#         log['total_loss'].append(avg(total_combined_loss))
#         log['discriminator_loss'].append(total_disc_loss / max(1, num_batches // 2))

#         print(f"[Epoch {epoch}/100]")
#         print(f"  Generator Loss    : {avg(total_gen_loss):.4f}")
#         print(f"  Total Loss        : {avg(total_combined_loss):.4f}")
#         print(f"    ├── Adversarial : {avg(total_adv):.4f}")
#         print(f"    ├── MSE         : {avg(total_mse):.4f}")
#         print(f"    └── CE          : {avg(total_ce):.4f}")
#         print(f"  Discriminator Loss: {log['discriminator_loss'][-1]:.4f}")

#     # === Save logs and plots ===
#     df = pd.DataFrame(log)
#     df.to_csv(f"loss_log_{tag}.csv", index=False)
#     torch.save(generator.state_dict(), f"generator_{tag}.pth")

#     os.makedirs(f"plots_{tag}", exist_ok=True)
#     for key in ['total_loss', 'adversarial_loss', 'mse_loss', 'ce_loss']:
#         plt.figure()
#         plt.plot(df['Epoch'], df[key], label=key.replace("_", " ").capitalize())
#         annotate_plot(plt.gca(), df['Epoch'], df[key])
#         plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title(key.replace("_", " ").capitalize())
#         plt.legend(); plt.grid(); plt.savefig(f"plots_{tag}/{key}.png"); plt.close()

#     plt.figure()
#     plt.plot(df['Epoch'], df['generator_loss'], label='Generator Loss')
#     plt.plot(df['Epoch'], df['discriminator_loss'], label='Discriminator Loss')
#     annotate_plot(plt.gca(), df['Epoch'], df['generator_loss'])
#     annotate_plot(plt.gca(), df['Epoch'], df['discriminator_loss'])
#     plt.title("Generator vs Discriminator Loss")
#     plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid()
#     plt.savefig(f"plots_{tag}/gen_vs_disc.png"); plt.close()

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data_path', type=str, default='./datasets/')
#     parser.add_argument('--checkpoint', type=str, required=True)
#     args = parser.parse_args()
#     main(args)


#BELOW CODE WAS USED WHEN ONLY PERTURBED TRAINING DATA IMAGES TO GENERATOR AS INPUT AND DISCRMINATOR WAS TAKING CLEAN TRAINING IMAGES AS GROUND TRUTH

# import os
# import torch
# import torch.nn as nn
# import argparse
# import pandas as pd
# import matplotlib.pyplot as plt
# from torch.utils.data import DataLoader
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
# from models.vit import ViT
# from pgd import pgd_attack
# from pgd_eval import load_model

# # === U-Net Generator ===
# class UNetGenerator(nn.Module):
#     def __init__(self, in_channels=3, out_channels=3):
#         super().__init__()
#         self.enc1 = self.down_block(in_channels, 64)
#         self.enc2 = self.down_block(64, 128)
#         self.enc3 = self.down_block(128, 256)
#         self.enc4 = self.down_block(256, 512)
#         self.bottleneck = nn.Sequential(nn.Conv2d(512, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.ReLU(True))
#         self.dec4 = self.up_block(512, 512)
#         self.dec3 = self.up_block(1024, 256)
#         self.dec2 = self.up_block(512, 128)
#         self.dec1 = self.up_block(256, 64)
#         self.final = nn.Sequential(nn.ConvTranspose2d(128, out_channels, 4, 2, 1), nn.Tanh())

#     def down_block(self, in_c, out_c):
#         return nn.Sequential(nn.Conv2d(in_c, out_c, 4, 2, 1), nn.BatchNorm2d(out_c), nn.LeakyReLU(0.2, inplace=True))

#     def up_block(self, in_c, out_c):
#         return nn.Sequential(nn.ConvTranspose2d(in_c, out_c, 4, 2, 1), nn.BatchNorm2d(out_c), nn.ReLU(True))

#     def forward(self, x):
#         e1 = self.enc1(x)
#         e2 = self.enc2(e1)
#         e3 = self.enc3(e2)
#         e4 = self.enc4(e3)
#         b = self.bottleneck(e4)
#         d4 = torch.cat([self.dec4(b), e4], dim=1)
#         d3 = torch.cat([self.dec3(d4), e3], dim=1)
#         d2 = torch.cat([self.dec2(d3), e2], dim=1)
#         d1 = torch.cat([self.dec1(d2), e1], dim=1)
#         return self.final(d1)

# # === Discriminator ===
# class Discriminator(nn.Module):
#     def __init__(self, in_channels=3):
#         super().__init__()
#         self.model = nn.Sequential(
#             nn.Conv2d(in_channels, 64, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(256, 512, 4, 1, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(512, 1, 4, 1, 1)
#         )

#     def forward(self, x):
#         return self.model(x)

# # === Helper for annotation ===
# def annotate_plot(ax, epochs, values):
#     ax.annotate(f"{values.iloc[0]:.4f}", xy=(epochs.iloc[0], values.iloc[0]), xytext=(epochs.iloc[0], values.iloc[0] + 0.02), color="green")
#     ax.annotate(f"{values.iloc[-1]:.4f}", xy=(epochs.iloc[-1], values.iloc[-1]), xytext=(epochs.iloc[-1], values.iloc[-1] + 0.02), color="red")


# # === Main Training Function ===
# def main(args):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     transform = transforms.Compose([
#         transforms.Resize((32, 32)),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5070, 0.4865, 0.4409), (0.2675, 0.2565, 0.2761))
#     ])
#     dataset = datasets.CIFAR100(root=os.path.join(args.data_path, 'cifar-100'), train=True, transform=transform, download=True)
#     dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

#     generator = UNetGenerator().to(device)
#     discriminator = Discriminator().to(device)
#     frozen_vit = load_model(args.checkpoint, device)
#     frozen_vit.eval()
#     for param in frozen_vit.parameters():
#         param.requires_grad = False

#     criterion_gan = nn.BCEWithLogitsLoss()
#     criterion_mse = nn.MSELoss()
#     criterion_ce = nn.CrossEntropyLoss()

#     optimizer_G = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.9, 0.999))
#     optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

#     λ1, λ2, λ3 = 1.0, 0.5, 1.0
#     tag = f"{λ1:.1f}_{λ2:.1f}_{λ3:.1f}".replace('.', 'p')

#     # tag = f"{int(λ1)}_{int(λ2)}_{int(λ3)}"

#     log = {key: [] for key in ['Epoch', 'total_loss', 'adversarial_loss', 'mse_loss', 'ce_loss', 'generator_loss', 'discriminator_loss']}

#     for epoch in range(1, 101):
#         total_adv, total_mse, total_ce = 0.0, 0.0, 0.0
#         total_gen_loss, total_disc_loss, total_combined_loss = 0.0, 0.0, 0.0
#         num_batches = 0

#         for images, labels in dataloader:
#             images, labels = images.to(device), labels.to(device)
#             num_batches += 1

#             perturbed = pgd_attack(frozen_vit, images,
#                                    torch.zeros(images.size(0), dtype=torch.long, device=device),
#                                    eps=4/255, alpha=1/255, iters=3)
#             fake_images = generator(perturbed)

#             # === Discriminator update ===
#             if num_batches % 2 == 0:
#                 real_out = discriminator(images)
#                 fake_out = discriminator(fake_images.detach())
#                 loss_D_real = criterion_gan(real_out, torch.ones_like(real_out).uniform_(0.8, 1.0).to(device))
#                 loss_D_fake = criterion_gan(fake_out, torch.zeros_like(fake_out).uniform_(0.0, 0.2).to(device))
#                 loss_disc = 0.5 * (loss_D_real + loss_D_fake)

#                 optimizer_D.zero_grad()
#                 loss_disc.backward()
#                 optimizer_D.step()

#                 total_disc_loss += loss_disc.item()

#             # === Generator update ===
#             out = discriminator(fake_images)
#             adv_loss = criterion_gan(out, torch.ones_like(out))
#             mse_loss = criterion_mse(fake_images, images)
#             ce_loss = criterion_ce(frozen_vit(fake_images), labels)

#             generator_loss = λ1 * adv_loss + λ2 * mse_loss
#             total_loss = generator_loss + λ3 * ce_loss

#             optimizer_G.zero_grad()
#             total_loss.backward()
#             optimizer_G.step()

#             total_adv += adv_loss.item()
#             total_mse += mse_loss.item()
#             total_ce += ce_loss.item()
#             total_gen_loss += generator_loss.item()
#             total_combined_loss += total_loss.item()

#         avg = lambda x: x / num_batches
#         log['Epoch'].append(epoch)
#         log['adversarial_loss'].append(avg(total_adv))
#         log['mse_loss'].append(avg(total_mse))
#         log['ce_loss'].append(avg(total_ce))
#         log['generator_loss'].append(avg(total_gen_loss))
#         log['total_loss'].append(avg(total_combined_loss))
#         log['discriminator_loss'].append(total_disc_loss / max(1, num_batches // 2))

#         print(f"[Epoch {epoch}/100]")
#         print(f"  Generator Loss    : {avg(total_gen_loss):.4f}")
#         print(f"  Total Loss        : {avg(total_combined_loss):.4f}")
#         print(f"    ├── Adversarial : {avg(total_adv):.4f}")
#         print(f"    ├── MSE         : {avg(total_mse):.4f}")
#         print(f"    └── CE          : {avg(total_ce):.4f}")
#         print(f"  Discriminator Loss: {log['discriminator_loss'][-1]:.4f}")

#     # === Save everything ===
#     df = pd.DataFrame(log)
#     df.to_csv(f"loss_log_{tag}.csv", index=False)
#     torch.save(generator.state_dict(), f"generator_{tag}.pth")

#     os.makedirs(f"plots_{tag}", exist_ok=True)
#     for key in ['total_loss', 'adversarial_loss', 'mse_loss', 'ce_loss']:
#         plt.figure()
#         plt.plot(df['Epoch'], df[key], label=key.replace("_", " ").capitalize())
#         annotate_plot(plt.gca(), df['Epoch'], df[key])
#         plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title(key.replace("_", " ").capitalize())
#         plt.legend(); plt.grid(); plt.savefig(f"plots_{tag}/{key}.png"); plt.close()

#     plt.figure()
#     plt.plot(df['Epoch'], df['generator_loss'], label='Generator Loss')
#     plt.plot(df['Epoch'], df['discriminator_loss'], label='Discriminator Loss')
#     annotate_plot(plt.gca(), df['Epoch'], df['generator_loss'])
#     annotate_plot(plt.gca(), df['Epoch'], df['discriminator_loss'])
#     plt.title("Generator vs Discriminator Loss")
#     plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid()
#     plt.savefig(f"plots_{tag}/gen_vs_disc.png"); plt.close()

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data_path', type=str, default='./datasets/')
#     parser.add_argument('--checkpoint', type=str, required=True)
#     args = parser.parse_args()
#     main(args)
