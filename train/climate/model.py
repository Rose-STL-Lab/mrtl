# Source: https://github.com/ktcarr/salinity-corn-yields/tree/master/mrtl

import numpy as np
import torch
import torch.nn.functional as F


def kr(matrices):
    """Khatri-Rao product of a list of matrices"""
    n_col = matrices[0].shape[1]
    for i, e in enumerate(matrices[1:]):
        if not i:
            res = matrices[0]
        a = res.reshape(-1, 1, n_col)
        b = e.reshape(1, -1, n_col)

        res = (a * b).reshape(-1, n_col)

    return res


def kruskal_to_tensor(factors):
    """Turns the Khatri-product of matrices into a full tensor"""
    shape = [factor.shape[0] for factor in factors]

    full_tensor = torch.mm(factors[0], kr(factors[1:]).T)
    return full_tensor.reshape(*shape)


class my_regression(torch.nn.Module):
    def __init__(self, lead, input_shape, output_shape):
        super().__init__()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n = np.prod([lead, 2, *input_shape])
        k = 1. / (n**(1 / 2))
        #         k = 1./n**2
        self.w = torch.nn.Parameter(torch.FloatTensor(
            lead, 2, np.prod(input_shape)).uniform_(-k, k).to(device),
                                    requires_grad=True)
        self.b = torch.nn.Parameter(torch.FloatTensor(output_shape).uniform_(
            0, 1).to(device),
                                    requires_grad=True)

    def forward(self, x):
        out = torch.einsum('abcd,bcd->a', x, self.w)
        return out + self.b


class mini_net(torch.nn.Module):
    def __init__(self, lead, input_shape, output_shape, hidden_neurons=3):
        super().__init__()

        n = np.prod([lead, *input_shape])
        k = 1. / np.sqrt(n)
        self.w1 = torch.nn.Parameter(torch.FloatTensor(
            lead, 2, np.prod(input_shape), hidden_neurons).uniform_(-k, k),
                                     requires_grad=True)
        self.b1 = torch.nn.Parameter(
            torch.FloatTensor(hidden_neurons).uniform_(-k, k),
            requires_grad=True)

        self.w2 = torch.nn.Parameter(
            torch.FloatTensor(hidden_neurons).uniform_(-k, k),
            requires_grad=True)
        self.b2 = torch.nn.Parameter(torch.FloatTensor(output_shape).uniform_(
            -k, k),
                                     requires_grad=True)

    def forward(self, x):
        out = F.relu(torch.einsum('abcd,bcde->ae', x, self.w1)) + self.b1
        out = torch.einsum('ae,e->a', out, self.w2) + self.b2
        return out


class my_regression_low(torch.nn.Module):
    def __init__(self, lead, input_shape, output_shape, K):
        # input_shape represents spatial dimensions (i.e. lat/lon)
        # lead represents number of months in advance used for prediction
        # K is rank of decomposition
        super().__init__()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n = np.prod([lead, *input_shape])
        k_A = 1. / np.sqrt((lead))
        k_B = 1. / np.sqrt(2)
        k_C = 1. / np.sqrt(np.prod(input_shape))

        n = np.prod([lead, 2, *input_shape])
        k_b = 1. / (n**(1 / 2))

        # Different initialization
        # Latent factors
        self.A = torch.nn.Parameter(torch.FloatTensor(lead, K).normal_(
            0, k_A).to(device),
                                    requires_grad=True)
        self.B = torch.nn.Parameter(torch.FloatTensor(2, K).normal_(
            0, k_B).to(device),
                                    requires_grad=True)
        self.C = torch.nn.Parameter(torch.FloatTensor(
            np.prod(input_shape), K).normal_(0, k_C).to(device),
                                    requires_grad=True)

        # Bias
        self.b = torch.nn.Parameter(torch.FloatTensor(output_shape).normal_(
            0, k_b).to(device),
                                    requires_grad=True)

    def forward(self, X):
        core = (X * kruskal_to_tensor([self.A, self.B, self.C])).reshape(
            X.shape[0], -1).sum(1)
        out = core.add_(self.b)
        return out
