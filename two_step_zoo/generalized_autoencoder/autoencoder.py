from . import GeneralizedAutoEncoder


class AutoEncoder(GeneralizedAutoEncoder):
    model_type = "ae"

    def loss(self, x, y):
        return self.rec_error(x, y).mean()
