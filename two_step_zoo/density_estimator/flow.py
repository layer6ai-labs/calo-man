from nflows.distributions import Distribution, StandardNormal
from nflows.flows.base import Flow

from . import DensityEstimator
from ..utils import batch_or_dataloader


class NormalizingFlow(DensityEstimator):

    model_type = "nf"

    def __init__(self, dim, transform, base_distribution: Distribution=None, **kwargs):
        super().__init__(**kwargs)
        self.transform = transform

        if base_distribution is None:
            self.base_distribution = StandardNormal([dim])
        else:
            self.base_distribution = base_distribution

        self._nflow = Flow(
            transform=self.transform,
            distribution=self.base_distribution
        )

    def sample(self, n_samples, y):
        if y is not None:
            # nflows will produce a number of samples n_samples * context.shape[0]
            # Since our context already has shape = n_samples, we can set n_samples=1.
            samples = self._nflow.sample(1, context=y)[:, 0, ...]
        else:
            samples = self._nflow.sample(n_samples)
        return self._inverse_data_transform(samples)

    @batch_or_dataloader()
    def log_prob(self, x, y):
        x = self._data_transform(x)
        if y is not None:
            # context expects y to have the same number of dimensions as x
            dummy_dim = [1]*(x.dim()-y.dim())
            shape = list(y.size())
            shape.extend(dummy_dim)
            log_prob = self._nflow.log_prob(x, context=y.reshape(shape).double())
        else:
            log_prob = self._nflow.log_prob(x)

        if len(log_prob.shape) == 1:
            log_prob = log_prob.unsqueeze(1)

        return log_prob
