import pandas as pd
import matplotlib.pyplot as plt
import os

# === Annotate start and end points ===
def annotate_plot(ax, epochs, values):
    ax.annotate(f"{values.iloc[0]:.4f}", xy=(epochs.iloc[0], values.iloc[0]), 
                xytext=(epochs.iloc[0], values.iloc[0] + 0.02), color="green")
    ax.annotate(f"{values.iloc[-1]:.4f}", xy=(epochs.iloc[-1], values.iloc[-1]), 
                xytext=(epochs.iloc[-1], values.iloc[-1] + 0.02), color="red")

# === Load CSV ===
csv_path = "loss_log_1_1_1.csv"  # Adjust path if needed
df = pd.read_csv(csv_path)

# === Create output folder ===
output_folder = "plots_1_1_1"
os.makedirs(output_folder, exist_ok=True)

# === Plot each loss component ===
loss_keys = ['total_loss', 'adversarial_loss', 'mse_loss', 'ce_loss']
for key in loss_keys:
    plt.figure()
    plt.plot(df['Epoch'], df[key], label=key.replace("_", " ").capitalize())
    annotate_plot(plt.gca(), df['Epoch'], df[key])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{key.replace('_', ' ').capitalize()} over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, f"{key}.png"))
    plt.close()

# === Plot Generator vs Discriminator Loss ===
plt.figure()
plt.plot(df['Epoch'], df['generator_loss'], label="Generator Loss")
plt.plot(df['Epoch'], df['discriminator_loss'], label="Discriminator Loss")
annotate_plot(plt.gca(), df['Epoch'], df['generator_loss'])
annotate_plot(plt.gca(), df['Epoch'], df['discriminator_loss'])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Generator vs Discriminator Loss")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_folder, "gen_vs_disc.png"))
plt.close()

print(f" All plots saved in '{output_folder}/'")
