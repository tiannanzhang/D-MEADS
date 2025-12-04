from abc import ABC, abstractmethod
from typing import Dict, Tuple
import torch

import constants as cst


class DiffusionAB(ABC):
    """An abstract class for loss functions."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        # Note: betas, alphas, and alphas_cumprod will be registered as buffers
        # in the child class (GaussianDiffusion) to ensure proper device handling
        self.gen_seq_size = config.HYPER_PARAMETERS[cst.LearningHyperParameter.MASKED_SEQ_SIZE]
        self.SEQ_size = config.HYPER_PARAMETERS[cst.LearningHyperParameter.SEQ_SIZE]
        self.cond_seq_size = self.SEQ_size - self.gen_seq_size

    @abstractmethod
    def loss(self, true: torch.Tensor, recon: torch.Tensor, **kwargs):
        """Computes the loss given the true and predicted values."""
        pass

    def forward_reparametrized(self, x_0: torch.Tensor, t:  torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Reparametrized forward diffusion process, takes in input x_0 and returns x_t after t steps of noise
        x_t(x_0, ϵ) = √(α̅_t)x_0 + √(1 - α̅_t)ϵ
        """
        # DEBUG: Check input x_0
        if torch.isnan(x_0).any():
            print(f"DEBUG: NaN in x_0 input! Shape: {x_0.shape}, NaN count: {torch.isnan(x_0).sum()}")
            print(f"  x_0 stats: min={x_0[~torch.isnan(x_0)].min()}, max={x_0[~torch.isnan(x_0)].max()}")

        # DEBUG: Check alphas_cumprod
        alphas_t = self.alphas_cumprod[t]
        if torch.isnan(alphas_t).any() or (alphas_t < 0).any():
            print(f"DEBUG: Invalid alphas_cumprod[t]! t={t}, alphas_t={alphas_t}")

        # CRITICAL: Remove non_blocking=True to prevent MPS corruption during training
        noise = torch.distributions.normal.Normal(0, 1).sample(x_0.shape).to(cst.DEVICE)

        # DEBUG: Check noise
        if torch.isnan(noise).any():
            print(f"DEBUG: NaN in sampled noise! Shape: {noise.shape}")

        # DEBUG: Check sqrt operations
        sqrt_alpha = torch.sqrt(alphas_t)
        sqrt_one_minus_alpha = torch.sqrt(1 - alphas_t)
        if torch.isnan(sqrt_alpha).any():
            print(f"DEBUG: NaN in sqrt(alphas_cumprod[t])! alphas_t={alphas_t}")
        if torch.isnan(sqrt_one_minus_alpha).any():
            print(f"DEBUG: NaN in sqrt(1-alphas_cumprod[t])! 1-alphas_t={1-alphas_t}")

        first_term = torch.einsum('bld, b -> bld', x_0, sqrt_alpha)
        second_term = torch.einsum('bld, b -> bld', noise, sqrt_one_minus_alpha)

        # DEBUG: Check terms
        if torch.isnan(first_term).any():
            print(f"DEBUG: NaN in first_term!")
        if torch.isnan(second_term).any():
            print(f"DEBUG: NaN in second_term!")

        x_t = first_term + second_term

        # DEBUG: Check output
        if torch.isnan(x_t).any():
            print(f"DEBUG: NaN in x_t output!")

        return x_t, noise

    def forward_process(self, x_0: torch.Tensor, t: torch.Tensor):
        # Standard forward process, takes in input x_0 and returns x_t after t steps of noise
        cov_matrix = torch.eye(x_0.shape)
        mean = torch.mul(x_0, torch.sqrt(self.alphas_cumprod[t]))
        std = torch.mul(cov_matrix, torch.sqrt(1 - self.alphas_cumprod[t]))
        # CRITICAL: Remove non_blocking=True to prevent MPS corruption
        x_T = torch.distributions.Normal(mean, std).rsample().to(cst.DEVICE)
        return x_T, {'mean': mean, 'std': std}