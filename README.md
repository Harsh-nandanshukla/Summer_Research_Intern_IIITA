# Exploring GAN-based Defense Strategy for Adversarial Images using Vision Transformer

## Objective
The goal of this project was to design a deep learning framework that is robust to adversarial attacks, with a focus on utilizing Vision Transformers (ViT) and Generative Adversarial Networks (GANs).

## Part 1: Baseline Model Training
- **Dataset Used**: CIFAR-100
- **Model**: Vision Transformer (ViT) using the [PAEViT](https://github.com/AkashVermaIN/PAEViT) implementation.
- **Training Outcome**: Achieved a classification accuracy of 74.17% on clean test images.
- **Model Checkpoint**: The trained ViT encoder was saved as `best.pth`.

## Part 2: Adversarial Attack and Initial Evaluation
- **Attack**: Projected Gradient Descent (PGD) was applied on test data.
- **Evaluation**: The adversarially perturbed images were evaluated on the trained ViT model.
- **Result**: Accuracy dropped significantly to 6.66%. A plot for this evaluation was saved as `pgd_test_accuracy_plot.png`.

## Part 3: Initial GAN Training with Frozen ViT Encoder
- **Approach**: Perturbed training images were passed through the ViT encoder.
- **CE Loss**: Batch-wise CE loss was computed. Last batch loss was 4.47; others ranged [4.0–4.47]. Later removed from training.
- **Architecture**: Generator (ViT encoder + Pix2Pix decoder), Discriminator (Pix2Pix).
- **Training Observations**: Generator loss ~2.08XX, Discriminator loss ~0.325X (indicating generator stagnation).

## Part 4: U-Net Based Generator
- **Update**: Replaced ViT encoder in generator with Pix2Pix U-Net architecture.
- **Custom Loss Function**: `total_loss = λ1 * adv_loss + λ2 * mse_loss + λ3 * ce_loss`
- **Loss Details**:
  - `adv_loss`: Adversarial loss (BCEWithLogits)
  - `mse_loss`: Between generator output and target image
  - `ce_loss`: On generator output using frozen ViT
- **Artifacts**:
  - Plots: `plots_λ1_λ2_λ3/`
  - Generator Weights: `generator_λ1_λ2_λ3.pth`
  - Accuracy: `pgd_test_accuracy_per_batch_λ1_λ2_λ3.csv`, `test_accuracies_λ1_λ2_λ3.csv`
  - Logs: `loss_log_λ1_λ2_λ3.csv`

## Part 5: Final Evaluation Setup
- Evaluated trained generator with ViT on both clean and perturbed test images.
- For `λ1=2.0, λ2=1.0, λ3=1.0`, MAE used instead of MSE.
- **Results**: Clean: 5–6%, Perturbed: 26–27%

## Part 6: Training on Clean + Perturbed Images
- **Update**: Both clean and perturbed images used for generator input and discriminator real labels.
- **Output Prefix**: `real_`
- **Results**: Clean: 30–31%, Perturbed: 26–27%

## Part 7: Real = Clean Only
- **Update**: Discriminator used only clean images as real.
- **Output Prefix**: `cleanreal_`
- **Results**: Clean: 30–31%, Perturbed: 26–27%

## Part 8: New Pipeline 
- **Script**: `gen_new.py`
- **Concept**:
  - Input: `Ip = I + P`
  - Generator predicts `P'' ≈ P`
  - Recover image: `I' = Ip - P''`
  - Accuracy evaluated on `I'` using frozen ViT
  - Loss: L2(P, P'')
- **Artifacts**:
  - Weights: `gen_new.pth`
  - Accuracy: `new_train.csv`
  - Plot: `plot_new_train.png`
- **Testing**:
  - Clean `I`, Perturbed `Ip`
  - Generator predicts `P1`, `P2`
  - Recover: `I1 = I - P1`, `I2 = Ip - P2`
  - Evaluate I1 and I2 on frozen pre-trained ViT
  - Outputs: `new_test.csv`, `new_test_accuracy_bar_plot.png`, `new_test_accutrcy_over_batches_plot.png`

## Part 9: Correlation Analysis (Planned)
- **Script**: `relation.py`
- **Concept**:
  - Apply Low Pass and High Pass filters to clean and perturbed test sets.
  - Compute correlation for L & H (clean), Lp & Hp (perturbed), and across clean/perturbed.
  - To work on the images created by  Low Pass and High Pass filters by observing their correlations and discoverinng the archtecture to acheive the aim.

## Tools & Frameworks Used
- Python, PyTorch, NumPy, Matplotlib, Pandas
- PAEViT, Pix2Pix GAN, PGD Attack

## Conclusion & Future Work
- GAN-based approach helps in restoring perturbed images to boost ViT accuracy.
- Future work includes completing the new perturbation learning pipeline .

## Team
- This project was completed independently during a summer research internship.


