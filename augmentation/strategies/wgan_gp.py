"""
Conditional WGAN-GP Data Augmentation Strategy Implementation.
"""

from typing import Dict, Any, Tuple, Optional
import torch
from torch.utils.data import DataLoader

from augmentation.strategies.base import AugmentationStrategy
from augmentation.dataset import SyntheticDataset, SimpleSyntheticDataset
from augmentation.validator import SyntheticDataValidator
from augmentation.gan.generator import ConditionalEEGGenerator
from augmentation.gan.critic import ConditionalEEGCritic
from augmentation.gan.trainer import GANTrainer
from training.device import get_device


class WGANGPStrategy(AugmentationStrategy):
    """Conditional WGAN-GP EEG augmentation strategy."""

    def __init__(self, seed: int = 42):
        super().__init__(seed=seed)
        self.generator: Optional[ConditionalEEGGenerator] = None
        self.critic: Optional[ConditionalEEGCritic] = None
        self.device = torch.device("cpu")

    def fit(self, dataloader: DataLoader, config: Dict[str, Any]):
        """Train Conditional WGAN-GP on real training dataloader."""
        gan_cfg = config.get("gan", {})
        self.device = get_device(gan_cfg.get("device", "auto"))

        # Inspect data shape from first batch
        sample_x, _ = next(iter(dataloader))
        _, bands, channels, samples = sample_x.shape

        self.generator = ConditionalEEGGenerator(
            latent_dim=int(gan_cfg.get("latent_dim", 128)),
            num_classes=int(gan_cfg.get("num_classes", 4)),
            num_bands=bands,
            num_channels=channels,
            num_samples=samples,
            hidden_dim=int(gan_cfg.get("generator_hidden_dim", 256)),
        )

        self.critic = ConditionalEEGCritic(
            num_classes=int(gan_cfg.get("num_classes", 4)),
            num_bands=bands,
            num_channels=channels,
            num_samples=samples,
            hidden_dim=int(gan_cfg.get("critic_hidden_dim", 256)),
        )

        trainer = GANTrainer(
            generator=self.generator,
            critic=self.critic,
            config=config,
            device=self.device,
        )
        trainer.fit(dataloader)

    def generate(self, num_samples: int, num_classes: int = 4) -> SyntheticDataset:
        """Generate class-balanced synthetic EEG samples."""
        if self.generator is None:
            raise RuntimeError("WGANGPStrategy generator must be trained via fit() before calling generate().")

        self.generator.eval()
        self.generator.to(self.device)

        with torch.no_grad():
            gen_rng = torch.Generator(device=self.device).manual_seed(self.seed)
            noise = torch.randn(num_samples, self.generator.latent_dim, device=self.device, generator=gen_rng)

            # Class-balanced labels
            labels = torch.arange(num_samples, device=self.device) % num_classes
            synthetic_x = self.generator(noise, labels).cpu()
            synthetic_y = labels.cpu()

        metadata = {
            "strategy": "wgan_gp",
            "seed": self.seed,
            "num_samples": num_samples,
            "num_classes": num_classes,
        }
        return SimpleSyntheticDataset(synthetic_x, synthetic_y, metadata)

    def augment(
        self,
        real_x: torch.Tensor,
        real_y: torch.Tensor,
        ratio: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Augment real EEG tensor dataset with synthetic samples."""
        if ratio <= 0.0:
            return real_x, real_y

        num_real = real_x.shape[0]
        num_synth = int(num_real * ratio)
        if num_synth <= 0:
            return real_x, real_y

        synth_ds = self.generate(num_samples=num_synth)
        synth_x = synth_ds.get_data()
        synth_y = synth_ds.get_labels()

        # Integrity Validation
        _, bands, channels, samples = real_x.shape
        SyntheticDataValidator.validate_synthetic_dataset(
            synth_x, synth_y, expected_bands=bands, expected_channels=channels, expected_samples=samples
        )

        aug_x = torch.cat([real_x, synth_x], dim=0)
        aug_y = torch.cat([real_y, synth_y], dim=0)
        return aug_x, aug_y
