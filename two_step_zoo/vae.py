import numpy as np
import torch

from .density_estimator import DensityEstimator
from .generalized_autoencoder import GeneralizedAutoEncoder
from .utils import batch_or_dataloader
from .distributions import diagonal_gaussian_log_prob, diagonal_gaussian_entropy, diagonal_gaussian_sample


class GaussianVAE(GeneralizedAutoEncoder, DensityEstimator):
    model_type = "vae"
    
    def __init__(
        self, 
        latent_dim,
        encoder,
        decoder,
        k=1,
        beta=1.0,
        **kwargs
    ):
        super().__init__(
            latent_dim,
            encoder,
            decoder,
            **kwargs
        )
        self.k = k
        self.beta = beta

    def sample(self, n_samples, y, true_sample=True):
        z = torch.randn((n_samples, self.latent_dim)).to(self.device)
        mu, log_sigma = self.decode_to_transformed(z, y)
        sample = diagonal_gaussian_sample(mu, torch.exp(log_sigma)) if true_sample else mu
        return self._inverse_data_transform(sample)

    @batch_or_dataloader()
    def log_prob(self, x, y):
        # NOTE: With k=1, this gives the ELBO.
        batch_size = x.shape[0]

        # NOTE: Perform data transform _before_ repeat_interleave because we do not want
        #       to dequantize the same x point in several different ways.
        x = self._data_transform(x)
        x = x.repeat_interleave(self.k, dim=0)

        mu_z, log_sigma_z = self.encode_transformed(x, y)
        z = diagonal_gaussian_sample(mu_z, torch.exp(log_sigma_z))
        mu_x, log_sigma_x = self.decode_to_transformed(z, y)

        log_p_z = diagonal_gaussian_log_prob(z, torch.zeros_like(z), torch.zeros_like(z))
        log_p_x_given_z = diagonal_gaussian_log_prob(
            x.flatten(start_dim=1),
            mu_x.flatten(start_dim=1),
            log_sigma_x.flatten(start_dim=1)
        )
        if self.k == 1:
            h_z_given_x = diagonal_gaussian_entropy(log_sigma_z)
            return log_p_x_given_z + self.beta * (log_p_z + h_z_given_x)
        else:
            log_q_z_given_x = diagonal_gaussian_log_prob(z, mu_z, log_sigma_z)
            elbo = log_p_x_given_z + self.beta * (log_p_z - log_q_z_given_x)
            return torch.logsumexp(elbo.reshape(batch_size, self.k, 1), dim=1) - np.log(self.k)
