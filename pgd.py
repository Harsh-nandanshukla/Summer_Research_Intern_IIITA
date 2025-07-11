import torch
import torch.nn.functional as F

def pgd_attack(model, images, labels, eps=4/255, alpha=1/255, iters=3):
    """
    Perform PGD attack on a batch of images.
    
    Args:
        model: The neural network model.
        images: Batch of input images.
        labels: Ground truth labels.
        eps: Light perturbation (L-infinity norm).
        alpha: Step size for each iteration.
        iters: Number of PGD iterations.

    Returns:
        Adversarial images.
    """
    images = images.clone().detach().to(images.device)
    ori_images = images.clone().detach()

    for i in range(iters):
        images.requires_grad = True
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        model.zero_grad()
        loss.backward()
        grad = images.grad.data

        adv_images = images + alpha * grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach()

    return images
