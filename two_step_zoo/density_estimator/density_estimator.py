from ..two_step import TwoStepComponent


class DensityEstimator(TwoStepComponent):

    def sample(self, n_samples, y):
        raise NotImplementedError("sample not implemented")

    def log_prob(self, x, y, **kwargs):
        raise NotImplementedError("log_prob not implemented")

    def loss(self, x, y, **kwargs):
        return -self.log_prob(x, y, **kwargs).mean()
