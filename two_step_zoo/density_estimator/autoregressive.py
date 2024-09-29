import torch

from . import DensityEstimator
from ..distributions import get_gaussian_mixture
from ..utils import batch_or_dataloader


class AutoRegressiveModel(DensityEstimator):

    model_type = 'arm'

    def __init__(self, ar_network, **kwargs):
        super().__init__(**kwargs)
        self.ar_network = ar_network


class GaussianMixtureLSTMModel(AutoRegressiveModel):

    def __init__(self, ar_network, image_height, input_length, **kwargs):
        super().__init__(ar_network, **kwargs)
        self.image_height = image_height
        if image_height is None:
            assert len(self.data_shape) == 1
        self.input_length = input_length

    @batch_or_dataloader()
    def log_prob(self, x, y):
        x = self._data_transform(x)
        if self.image_height is None:
            x = torch.unsqueeze(x, 1)
        weights, mus, sigmas = self.ar_network.forward(x, y)
        gmm = get_gaussian_mixture(weights, mus, sigmas)
        out = gmm.log_prob(torch.permute(x.flatten(start_dim=2), (0, 2, 1)))
        return out.sum(dim=1, keepdim=True)

    def sample(self, n_samples, y):
        new_coordinate = torch.unsqueeze(y, 2)
        samples = None
        h_c = None
        for _ in range(self.input_length):
            weights, mus, sigmas, h_c = self.ar_network.forward(x=new_coordinate,
                                                                return_h_c=True,
                                                                h_c=h_c,
                                                                not_sampling=False)
            new_coordinate = get_gaussian_mixture(weights, mus, sigmas).sample()
            if samples is not None:
                samples = torch.cat((samples, new_coordinate), dim=1)
            else:
                samples = new_coordinate
        if self.image_height is None:
            samples = torch.squeeze(samples, 2)
        else:
            samples = torch.permute(samples, (0, 2, 1))
            samples = torch.reshape(samples, (samples.shape[0], samples.shape[1], self.image_height, -1))
        return self._inverse_data_transform(samples)
