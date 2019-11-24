import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dbt

class MultivariateNormalDiag(nn.Module):
    def __init__(self, in_width, base_depth, latent_size, scale=None):
        super(MultivariateNormalDiag, self).__init__()
        self._latent_size = latent_size
        self._base_depth = base_depth
        self._scale = scale
        self._dense1 = nn.Linear(in_width, base_depth)
        self._dense2 = nn.Linear(base_depth, base_depth)
        if scale is None:
            self._out_width = 2 * self._latent_size
        else:
            self._out_width = self._latent_size
        self._out_layer = nn.Linear(base_depth, self._out_width)

    def forward(self, data):
        out = self._dense1(data)
        out = F.relu(out)
        out = self._dense2(out)
        out = F.relu(out)
        out = self._out_layer(out)
        if self._scale is None:
            loc, logscale = torch.chunk(out, 2, dim=-1)
            scale = F.softplus(logscale)
        else:
            loc = out
            scale = self._scale
#        return dbt.Independent(
        return dbt.multivariate_normal.MultivariateNormal(
                    loc=loc,
                    covariance_matrix=torch.diag_embed(scale))
#                    self._latent_size)
