"""
1-Lipschitz Gradient Penalty Computation for WGAN-GP.
"""

import torch


def compute_gradient_penalty(
    critic: torch.nn.Module,
    real_eeg: torch.Tensor,
    fake_eeg: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute WGAN-GP 1-Lipschitz gradient penalty on interpolated tensors.

    Args:
        critic: Conditional Critic PyTorch module
        real_eeg: Real EEG tensor (B, Bands, Channels, Samples)
        fake_eeg: Fake EEG tensor (B, Bands, Channels, Samples)
        labels: Class labels tensor (B,)
        device: Execution device

    Returns:
        Scalar gradient penalty tensor
    """
    batch_size = real_eeg.shape[0]

    # Sample random interpolation coefficient epsilon ~ U(0, 1)
    # Shape matching (B, 1, 1, 1)
    epsilon = torch.rand((batch_size, 1, 1, 1), device=device)
    interpolated = (epsilon * real_eeg + (1 - epsilon) * fake_eeg).requires_grad_(True)

    # Critic evaluation on interpolated samples
    critic_interpolated = critic(interpolated, labels)

    # Calculate gradients of Critic scores w.r.t. interpolated samples
    gradients = torch.autograd.grad(
        outputs=critic_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(critic_interpolated),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    # Flatten gradients to compute L2 norm per sample
    gradients_flat = gradients.view(batch_size, -1)
    gradient_norm = gradients_flat.norm(2, dim=1)

    # Penalty forcing norm close to 1
    penalty = torch.mean((gradient_norm - 1.0) ** 2)
    return penalty
