import logging
import time

import torch

from config import config

logger = logging.getLogger(config.parent_logger_name).getChild(__name__)


# Modified from tensorly (https://github.com/tensorly/tensorly)
def init_factors(X, rank, nonnegative, init='svd'):
    if init == 'random':
        factors = [torch.randn((dim_size, rank)) for dim_size in X.size()]
        if nonnegative:
            factors[0] = abs(factors[0])
            factors[1] = abs(factors[1])
            factors[2] = -abs(factors[2])
        return factors

    elif init == 'svd':
        # Initialize using left singular vectors
        # Fill with random values if any mode.size() < rank
        factors = []
        for mode in range(X.dim()):
            # Use eigendecomposition method on Gram matrix (XX^T) for SVD
            # mode-unfold of X
            G = unfold(X, mode)
            G = torch.mm(G, G.t())
            eigvals, eigvecs = torch.symeig(G, eigenvectors=True)
            # eigenvectors are reverse sorted (ascending)
            factor = torch.flip(eigvecs[:, -int(min(X.size(mode), rank)):],
                                [1])
            # Fill remaining columns with random vectors
            if rank > X.size(mode):
                factor = torch.cat(
                    (factor, torch.randn(X.size(mode), rank - X.size(mode)).to(
                        X.device)),
                    dim=1)
            if nonnegative:
                if mode in [2]:
                    factor = -abs(factor)
                else:
                    factor = abs(factor)
            factors.append(factor)
        return factors
    else:
        raise ValueError('Wrong value for init: {}'.format(init))


def unfold(X, mode):
    U = X.permute([mode] + list(range(mode)) + list(range(mode + 1, X.dim())))
    return U.reshape([X.size(mode), -1])


# Modified from tensorly (https://github.com/tensorly/tensorly)
def cp_decompose(X,
                 rank,
                 tol=1e-8,
                 max_iter=100,
                 orthogonalize=False,
                 nonnegative=False,
                 verbose=False):
    device = X.device
    factors = init_factors(X, rank, nonnegative, init='svd')

    X_norm = X.norm(p=2)

    fits = []
    grams = torch.zeros((len(factors), rank, rank)).to(device)
    weights = torch.ones(rank).to(device)

    start_time = time.time()

    for t in range(max_iter):
        # Create Gram matrices for each factor
        for i, f in enumerate(factors):
            if orthogonalize and t < 5:
                factors[i] = torch.qr(f).Q if min(f.size()) >= rank else f
            grams[i] = torch.mm(f.t(), f)
        for mode in range(X.dim()):
            V = torch.ones((rank, rank)).to(device)
            khatri_rao = torch.ones((1, rank)).to(device)
            for i in range(len(factors) - 1, -1, -1):
                if i != mode:
                    V = V * grams[i]
                    khatri_rao = torch.reshape(
                        torch.einsum('ir,jr->ijr', (factors[i], khatri_rao)),
                        [-1, rank])
            mttkrp = torch.mm(unfold(X, mode), khatri_rao)
            if nonnegative:
                factor = factors[mode] * (
                    mttkrp.clamp_min_(1e-30) /
                    (factors[mode] @ V).clamp_min_(1e-30))
            else:
                factor = torch.solve(mttkrp.t(), V).solution.t()
            factors[mode] = factor

            if t == 0:
                weights = torch.norm(factors[mode], dim=0)
            else:
                weights = torch.max(
                    torch.max(torch.abs(factors[mode]), dim=0)[0],
                    torch.ones(rank).to(device))

            factors[mode] = factors[mode] / weights
            grams[mode] = torch.mm(factors[mode].t(), factors[mode])

        fit = 1 - (torch.norm(reconstruct_from_cp(weights, factors) - X,
                              p=2)) / X_norm
        fits.append(fit)
        if verbose:
            if t == 0:
                logger.info(
                    "[{0:.2f}s] | Iter: {1} | Fit: {2}, Fit change: {3}".
                    format(time.time() - start_time, t + 1, fits[-1],
                           abs(fits[-1])))
            else:
                logger.info(
                    "[{0:.2f}s] | Iter: {1} | Fit: {2}, Fit change: {3}".
                    format(time.time() - start_time, t + 1, fits[-1],
                           abs(fits[-2] - fits[-1])))
        if t >= 1 and abs(fits[-2] - fits[-1]) < tol:
            logger.info(
                "FINISH | [{0:.2f}s] | Iter: {1} | Fit: {2}, Fit change: {3} < Tol:{4}"
                .format(time.time() - start_time, t + 1, fits[-1],
                        abs(fits[-2] - fits[-1]), tol))
            break
        elif t == max_iter - 1:
            logger.info(
                "FINISH | [{0:.2f}s] | Iter: {1} = Max_iter ({2}) | Fit: {3}, Fit change:{4}"
                .format(time.time() - start_time, t + 1, max_iter, fits[-1],
                        abs(fits[-2] - fits[-1])))

    # Reorder columns according decreasing weights
    idx = torch.argsort(weights, descending=True)
    weights_sorted = torch.index_select(weights, dim=0, index=idx)
    for i in range(len(factors)):
        factor = torch.index_select(factors[i], dim=1, index=idx)
        factors[i] = factor

    return weights_sorted, factors


def reconstruct_from_cp(weights, factors):
    # A * (C khatri_rao B)^T
    rank = factors[0].size(1)
    orig_shape = [factor.size(0) for factor in factors]

    khatri_rao = torch.ones((1, rank)).to(factors[0].device)
    for i in range(len(factors) - 1, 0, -1):
        khatri_rao = torch.reshape(
            torch.einsum('ir,jr->ijr', (factors[i], khatri_rao)), [-1, rank])
    return torch.mm(factors[0] * weights, khatri_rao.t()).view(*orig_shape)
