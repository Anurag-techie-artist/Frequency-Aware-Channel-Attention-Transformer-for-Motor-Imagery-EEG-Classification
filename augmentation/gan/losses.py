"""
Wasserstein Loss Functions for WGAN-GP Training.
"""

import torch


def wasserstein_loss_critic(
    real_scores: torch.Tensor,
    fake_scores: torch.Tensor,
    gradient_penalty: torch.Tensor = None,
    gp_lambda: float = 10.0,
) -> torch.Tensor:
    """
    Compute Critic Wasserstein loss = E[D(fake)] - E[D(real)] + gp_lambda * GP.

    Args:
        real_scores: Scores assigned to real samples (B, 1)
        fake_scores: Scores assigned to fake samples (B, 1)
        gradient_penalty: Optional 1-Lipschitz penalty tensor
        gp_lambda: Gradient penalty weight coefficient

    Returns:
        Scalar Critic loss tensor
    """
    loss = torch.mean(fake_scores) - torch.mean(real_scores)
    if gradient_penalty is not None:
        loss += gp_lambda * gradient_penalty
    return loss


def wasserstein_loss_generator(fake_scores: torch.Tensor) -> torch.Tensor:
    """
    Compute Generator Wasserstein loss = -E[D(fake)].

    Args:
        fake_scores: Scores assigned to generated fake samples (B, 1)

    Returns:
        Scalar Generator loss tensor
    """
    return -torch.mean(fake_scores)
