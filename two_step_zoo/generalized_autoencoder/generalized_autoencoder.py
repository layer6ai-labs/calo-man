import warnings

import torch
from functorch import jvp

from ..two_step import TwoStepComponent
from ..utils import batch_or_dataloader


class GeneralizedAutoEncoder(TwoStepComponent):
    """
    GeneralizedAutoEncoder Parent Class
    """
    model_type = None

    def __init__(
            self,

            latent_dim,

            encoder,
            decoder,

            **kwargs
    ):
        super().__init__(**kwargs)

        self.latent_dim = latent_dim

        self.encoder = encoder
        self.decoder = decoder

    @batch_or_dataloader()
    def encode(self, x, y):
        # NOTE: Assume encode *only* wants a single output
        x = self._data_transform(x)
        return self._encode_transformed_without_tuple(x, y)

    @batch_or_dataloader()
    def encode_transformed(self, x, y):
        return self.encoder(x, y)

    def _encode_transformed_without_tuple(self, x, y):
        z = self.encode_transformed(x, y)
        return z[0] if type(z) == tuple else z

    @batch_or_dataloader()
    def decode(self, z, y):
        # NOTE: Assume decode *only* wants a single output
        x = self._decode_to_transformed_without_tuple(z, y)
        return self._inverse_data_transform(x)

    @batch_or_dataloader()
    def decode_to_transformed(self, z, y):
        return self.decoder(z, y)

    def _decode_to_transformed_without_tuple(self, z, y):
        x = self.decode_to_transformed(z, y)
        return x[0] if type(x) == tuple else x

    @batch_or_dataloader()
    def rec_error(self, x, y, return_z=False):
        x = self._data_transform(x)
        z = self._encode_transformed_without_tuple(x, y)
        rec_x = self._decode_to_transformed_without_tuple(z, y)

        if return_z:
            return torch.sum(torch.square(x - rec_x).flatten(start_dim=1), dim=1, keepdim=True), z
        else:
            return torch.sum(torch.square(x - rec_x).flatten(start_dim=1), dim=1, keepdim=True)

    def _decoder_jacobian(self, z, y):
        """
        Compute flattened Jacobian of decoder at inputs `z` by calling jvp against every one-hot vector of length
        latent_dim. Note that jvp only parallelizes over the batch coordinate, so that in order to parallelize over
        one-hot vectors as well (when PARALLELIZE_OVER_ONE_HOTS=True), the first dimension is overloaded to index both
        batches and one-hots. Parallelizing over one-hots requires more memory, so switching PARALLELIZE_OVER_ONE_HOTS
        to False will result in only parallelizing over the batch and iterating over one-hot vectors, reducing memory
        requirements.
        """
        PARALLELIZE_OVER_ONE_HOTS = True
        if not PARALLELIZE_OVER_ONE_HOTS:
            jac = []
            for i in range(self.latent_dim):
                v = torch.zeros_like(z)
                v[:, i] = 1
                _, jac_vec_prod = jvp(lambda input_: self.decode(input_, y), (z,), (v,))
                jac.append(jac_vec_prod.flatten(start_dim=1))
            out = torch.stack(jac, dim=2)
        else:
            vs = []
            for i in range(self.latent_dim):
                v = torch.zeros_like(z)
                v[:, i] = 1
                vs.append(v)
            vs = torch.stack(vs, dim=0)
            zs = torch.repeat_interleave(z.unsqueeze(0), self.latent_dim, 0)

            vs = torch.flatten(vs, start_dim=0, end_dim=1)
            zs = torch.flatten(zs, start_dim=0, end_dim=1)

            _, jac_vec_prod = jvp(lambda input_: self.decode(input_, y), (zs,), (vs,))
            out = torch.reshape(jac_vec_prod.flatten(start_dim=1), (self.latent_dim, z.shape[0], -1)).permute((1, 2, 0))

        return out

    @batch_or_dataloader()
    def log_det_jtj(self, x, y, eps=1e-9):
        z = self.encode(x, y)
        jac = self._decoder_jacobian(z, y)
        jtj = torch.bmm(jac.transpose(1, 2), jac)

        try:
            cholesky_factor = torch.linalg.cholesky(jtj)
        except torch._C._LinAlgError:
            warnings.warn(
                f"Singular JTJ for {self.model_type}. Adding eps={eps} to its diagonal entries for "
                "log-det computation.")
            jtj += torch.eye(jac.shape[-1]) * eps
            cholesky_factor = torch.linalg.cholesky(jtj)

        cholesky_diagonal = torch.diagonal(cholesky_factor, dim1=1, dim2=2)
        log_det_jtj = 2 * torch.sum(torch.log(cholesky_diagonal), dim=1, keepdim=True)

        return log_det_jtj
