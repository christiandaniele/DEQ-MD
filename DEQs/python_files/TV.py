import torch
from deepinv.models import TVDenoiser
from deepinv.optim import Prior

class TVPrior(Prior):
    r"""
    Total variation (TV) prior :math:`\reg{x} = \| D x \|_{1,2}`.

    :param float def_crit: default convergence criterion for the inner solver of the TV denoiser; default value: 1e-8.
    :param int n_it_max: maximal number of iterations for the inner solver of the TV denoiser; default value: 1000.
    """

    def __init__(self, def_crit=1e-8, n_it_max=1000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.explicit_prior = True
        self.TVModel = TVDenoiser(crit=def_crit, n_it_max=n_it_max)

    def g(self, x, *args, **kwargs):
        r"""
        Computes the regularizer

        .. math::
            g(x) = \|Dx\|_{1,2}

        where D is the finite differences linear operator,
        and the 2-norm is taken on the dimension of the differences.

        :param torch.Tensor x: Variable :math:`x` at which the prior is computed.
        :return: (torch.Tensor) prior :math:`g(x)`.
        """

        y = torch.sqrt(torch.sum(self.nabla(x) ** 2 , dim=-1)+1e-6)
        return torch.sum(y.reshape(x.shape[0], -1), dim=-1)

    def nabla(self, x):
        r"""
        Applies the finite differences operator associated with tensors of the same shape as x.
        """
        return self.TVModel.nabla(x)

    def nabla_adjoint(self, x):
        r"""
        Applies the adjoint of the finite difference operator.
        """
        return self.TVModel.nabla_adjoint(x)