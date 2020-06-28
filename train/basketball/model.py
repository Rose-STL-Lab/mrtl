import torch


class DataParallelPassthrough(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class Full(torch.nn.Module):
    def __init__(self, a_dims, b_dims, c_dims, counts):
        super().__init__()
        self.a_dims = a_dims
        self.b_dims = b_dims
        self.c_dims = c_dims
        self.W = torch.nn.Parameter(torch.randn((a_dims, *b_dims, *c_dims)),
                                    requires_grad=True)
        # self.b = torch.nn.Parameter(torch.ones(a_dims) * np.log(counts[1] / (counts[0])), requires_grad=True)
        self.b = torch.nn.Parameter(torch.zeros(a_dims), requires_grad=True)

    def forward(self, a, bh_pos, def_pos):
        # Only some defenders in defender box
        out = torch.einsum(
            'bcd,bcd->b', self.W[a.long(), bh_pos[:, 0].long(),
                                 bh_pos[:, 1].long(), :, :], def_pos.float())
        return out.add_(self.b[a.long()])


class Low(torch.nn.Module):
    def __init__(self, a_dims, b_dims, c_dims, K, counts):
        super().__init__()
        self.a_dims = a_dims
        self.b_dims = b_dims
        self.c_dims = c_dims
        self.K = K

        self.A = torch.nn.Parameter(torch.randn((a_dims, K)),
                                    requires_grad=True)
        self.B = torch.nn.Parameter(torch.randn(*b_dims, K),
                                    requires_grad=True)
        self.C = torch.nn.Parameter(torch.randn(*c_dims, K),
                                    requires_grad=True)

        # self.b = torch.nn.Parameter(torch.ones(a_dims) * np.log(counts[1] / (counts[0])), requires_grad=True)
        self.b = torch.nn.Parameter(torch.zeros(a_dims), requires_grad=True)

    def forward(self, a, bh_pos, def_pos):
        # Tensor multiply def_pos and self.C to get sum for all defenders, and then sum over latent factors
        out = (self.A[a.long(), :] *
               self.B[bh_pos[:, 0].long(), bh_pos[:, 1].long(), :] *
               torch.einsum('bcd,cde->be', def_pos.float(), self.C)).sum(1)
        return out.add_(self.b[a.long()])

    def constrain(self):
        # Clamp weights
        self.A.clamp_min_(min=0)
        self.B.clamp_min_(min=0)
        # self.C.clamp_max_(max=0)
