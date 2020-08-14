import logging
import os
from datetime import datetime

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.utils.data
import xarray as xr
from cp_als import unfold


def set_logger(logger, log_path=None):
    # create logger
    logger.setLevel(logging.DEBUG)

    # create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # create formatter and add to handler
    formatter = logging.Formatter(
        '[%(asctime)s] %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(ch)

    # File handler
    if log_path is not None:
        fh = logging.FileHandler(log_path, 'w+')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)


def size_to_str(lst):
    lst_str = "x".join([str(i).zfill(2) for i in lst])
    return lst_str


def calc_F1(fp, fn, tp):
    if tp == 0 or (tp + fp) == 0 or (tp + fn) == 0:
        precision = 0.0
        recall = 0.0
        F1 = 0.0
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        F1 = 2 * (precision * recall) / (precision + recall)
    return F1, precision, recall


def accum_grad(gradients, model):
    for i, (name, p) in enumerate(model.named_parameters()):
        if name != 'module.b' and name != 'b':
            gradients[i].add_(p.grad.data)


def grad_stats(avg_grads):
    grads = torch.cat([g.contiguous().view(-1) for g in avg_grads])

    grad_norm = (torch.norm(grads, p=2)**2).item()
    grad_entropy = (-(grads.clamp_min(1e-30) *
                      torch.log(grads.clamp_min(1e-30))).sum()).item()
    grad_var = torch.var(grads).item()

    return grad_norm, grad_entropy, grad_var


def l1_regularizer(model, device):
    reg = torch.tensor(0.).to(device)
    numel = 0
    for name, p in model.named_parameters():
        if name != 'module.b':
            reg.add_(torch.norm(p.view(-1), p=1))
            numel += p.numel()
    return reg / numel


def l2_regularizer(model, device):
    reg = torch.tensor(0.).to(device)
    numel = 0
    for name, p in model.named_parameters():
        if name != 'module.b':
            reg.add_(torch.norm(p.view(-1), p=2)**2)
            numel += p.numel()
    return reg / numel


def create_kernel(dims, sigma, device):
    coords = torch.cartesian_prod(torch.arange(0, dims[0], dtype=torch.float),
                                  torch.arange(0, dims[1], dtype=torch.float))
    dist = torch.cdist(coords, coords, p=2).to(device)

    # To normalize distances across different resolutions
    dist = dist / torch.max(dist)

    # K is matrix of degree of similarity between coordinates
    K = torch.exp(-dist**2 / sigma)
    return K


# Implement cdist from https://github.com/pytorch/pytorch/issues/15253
def pdist(X):
    X_norm = X.pow(2).sum(dim=-1, keepdim=True)
    res = torch.addmm(X_norm.transpose(-2, -1),
                      X,
                      X.transpose(-2, -1),
                      alpha=-2).add_(X_norm)
    res = res.clamp_min_(1e-30)
    return res


def bball_spatial_regularizer(model, K_B, K_C, device):
    reg = torch.tensor(0.).to(device)
    if type(model.module).__name__.startswith('Full'):
        W_size = model.W.size()

        # Court dimension
        W_unfold = unfold(model.W.view(W_size[0], W_size[1] * W_size[2],
                                       W_size[3], W_size[4]),
                          mode=1).contiguous()
        reg.add_((K_B * pdist(W_unfold)).sum() /
                 (torch.numel(model.W) * np.prod(model.b_dims)))

        # Defender position
        W_unfold = unfold(model.W.view(W_size[0], W_size[1], W_size[2],
                                       W_size[3] * W_size[4]),
                          mode=3).contiguous()
        reg.add_((K_C * pdist(W_unfold)).sum() /
                 (torch.numel(model.W) * np.prod(model.c_dims)))
    else:
        # Court position
        reg.add_((K_B * pdist(model.B.view(-1, model.K))).sum() /
                 (torch.numel(model.B) * np.prod(model.b_dims)))

        # Defender position
        reg.add_((K_C * pdist(model.C.view(-1, model.K))).sum() /
                 (torch.numel(model.C) * np.prod(model.c_dims)))

    return reg


def class_counts(dataset):
    _, counts = np.unique(dataset.y, return_counts=True)
    return counts


def calc_weights(dataset):
    counts = class_counts(dataset)
    return np.where(dataset.y == 1, counts[0] / counts.sum(),
                    counts[1] / counts.sum())


def expand_pos(T, shape, dim):
    T_size = list(T.size())
    T_size.insert(dim + 1, shape[1])
    T_size[dim] = shape[0]
    return T.view(*T_size)


def contract_pos(T, dim):
    T_size = list(T.size())
    val = T_size.pop(dim + 1)
    T_size[dim] = val * T_size[dim]
    return T.view(*T_size)


def finegrain(T, new_shape, start_dim, mode='nearest'):
    old_shape = T.shape

    assert T.ndim in [3, 5], "T.ndim must be 3 or 5"
    assert start_dim in [0, 1, 3], "start_dim must be 0, 1, or 3"

    # Calculate scale
    scale = float(new_shape[0]) / old_shape[start_dim]
    assert scale == (
        float(new_shape[1]) /
        old_shape[start_dim + 1]), "Scale is not the same across axes."

    new = None
    if T.ndim == 5:
        old = T.clone().detach().permute(
            0, 4 - start_dim, 5 - start_dim, start_dim, start_dim + 1).view(
                old_shape[0],
                old_shape[4 - start_dim] * old_shape[5 - start_dim],
                old_shape[start_dim], old_shape[start_dim + 1])
        interp = torch.nn.functional.interpolate(old,
                                                 scale_factor=scale,
                                                 mode=mode)
        new = interp.view(old_shape[0], old_shape[4 - start_dim],
                          old_shape[5 - start_dim],
                          *new_shape).permute(0, 4 - start_dim, 5 - start_dim,
                                              start_dim, start_dim + 1)
    elif T.ndim == 3:
        old = T.clone().detach().permute(2, 0, 1).unsqueeze(0)
        interp = torch.nn.functional.interpolate(old,
                                                 scale_factor=scale,
                                                 mode=mode)
        new = interp.squeeze().permute(1, 2, 0)

    return new


# Source: https://github.com/ktcarr/salinity-corn-yields/tree/master/mrtl
def plot_setup(plot_range=[-125.25, -66, 22.5, 50],
               figsize=(7, 5),
               central_lon=0):
    # Function sets up plotting environment for continental US
    # Returns fig, ax

    # Set up figure for plotting
    sns.set()
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=central_lon))
    states_provinces = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='50m',
        facecolor='none')
    ax.add_feature(states_provinces, edgecolor='black')
    ax.coastlines()
    ax.set_extent(plot_range, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.BORDERS)
    ax.title.set_fontsize(30)
    return fig, ax


def multis_to_datetime(multis):
    # Function to convert multi-index of year/month to Pandas datetime index
    multi_to_datetime = lambda multi: datetime(multi[0], multi[1], 1)
    return (pd.Index([multi_to_datetime(multi) for multi in multis]))


def minmax_scaler(x, old_min, old_max, new_min, new_max):
    # Scale elements in a 1-dimensional array to [0,1]
    x_scaled = (x - old_min) / (old_max - old_min)
    x_scaled = x_scaled * (new_max - new_min) + new_min
    return x_scaled


def remove_season(data, standardize=True, mean=None, std=None):
    # Function to remove seasonality from data
    # Returns de-seasonalized data with same shape as input
    if mean is None:
        mean = data.mean(dim='year')
        std = data.std(dim='year')
    if standardize:
        data = (data - data.mean(dim='year')) / data.std(dim='year')
    else:
        data = data - data.mean(dim='year')

    return data, mean, std


def normalize(data,
              old_min=None,
              old_max=None,
              new_min=0,
              new_max=1,
              dim='time'):
    #     Function to remove seasonality from data
    #     Returns de-seasonalized data with same shape as input

    if 'time' in data.dims:  # get year and month as separate dimension
        data = unstack_month_and_year(data)

    if dim == 'time':
        data = data.stack(time=['year', 'month'])

    if old_min is None:
        old_min = data.min(dim=dim)
        old_max = data.max(dim=dim)

    data.values = np.float32(
        minmax_scaler(data,
                      old_min=old_min,
                      new_min=new_min,
                      old_max=old_max,
                      new_max=new_max))

    return data.unstack(), old_min, old_max


def weight_by_area(data_fp, data):
    #     Function to weight dataarray by the area of each gridcell
    #     Returns dataarray with same dimensions

    dim = [len(data.lat), len(data.lon)]
    fp = os.path.join(data_fp, 'gridarea_{0}x{1}.nc'.format(*dim))
    grid_area = xr.open_dataarray(fp)
    grid_prop = grid_area / np.max(grid_area)
    grid_prop = grid_prop.assign_coords({
        'lon': data.lon,
        'lat': data.lat
    })  # 'snap' coords to match data

    return data * grid_prop


def preprocess(data_fp,
               data,
               do_remove_season=True,
               mean=None,
               std=None,
               do_normalize=True,
               old_min=None,
               old_max=None):
    # Function to pre-process data, with options to remove seasonality, detrend
    # and normalize
    # Returns pre-processed data with time, lat, and lon dimensions

    if 'time' in data.dims:  # get year and month as separate dimension
        year = data.time.dt.year
        month = data.time.dt.month
        times = pd.MultiIndex.from_arrays([year, month],
                                          names=('year', 'month'))
        data = unstack_month_and_year(data)

    # REMOVE SEASONAL CYCLE
    if do_remove_season:
        data, mean, std = remove_season(data,
                                        standardize=True,
                                        mean=mean,
                                        std=std)

    # NORMALIZE
    if do_normalize:
        if remove_season:
            data, old_min, old_max = normalize(data,
                                               dim='time',
                                               old_min=old_min,
                                               old_max=old_max)
        else:
            data, old_min, old_max = normalize(data,
                                               dim='year',
                                               old_min=old_min,
                                               old_max=old_max)

    # WEIGHT BY GRIDCELL AREA
    if 'lat' in data.dims:
        data = weight_by_area(data_fp, data)

    data = data.stack(time=['year', 'month'
                            ])  # Make time a coordinate (and a datetime index)
    data = data.sel(time=times)
    data = data.assign_coords({
        'time': multis_to_datetime(data.time.values)
    }).transpose('time', ...)

    return (data, mean, std, old_min, old_max)


def unstack_month_and_year(data):
    # Function 'unstacks' month and year in a dataframe with 'time' dimension
    # The 'time' dimension is separated into a month and a year dimension
    # This increases the number of dimensions by 1

    year = data.time.dt.year
    month = data.time.dt.month
    new_idx = pd.MultiIndex.from_arrays([year, month], names=('year', 'month'))

    return (data.assign_coords({'time': new_idx}).unstack('time'))


def diff_detrend(x):
    #     Function does 'difference' detrending
    #     x is the vector to detrend
    #     returns a vector of length len(x)-1
    return (x[1:] - x[:-1])


def diff_detrend_xr(data):
    #     Detrend xarray dataarray along particular axis
    if not ('time' in data.dims):
        data = data.stack(time=['year', 'month'])

    time_dim = data.dims.index('time')  # Get dimension corresponding to time
    #time_dim = data.da.dims.index('time')  # Get dimension corresponding to time

    #     Update coordinates by reducing time dimension by 1
    new_coords = {
        coord: data.coords[coord]
        for coord in data.coords if coord != 'time'
    }
    new_coords['time'] = data.time[1:]

    #     Detrend
    vals = np.apply_along_axis(diff_detrend, axis=time_dim, arr=data)
    data_new = xr.DataArray(vals, coords=new_coords, dims=data.dims)
    return (data_new)


def mse(x, y):
    # Custom function to compute MSE for sanity check
    #     return(torch.sum((x-y)**2) / len(x))
    x = x.float()
    y = y.float()
    return (torch.mean((x - y)**2))


def mae(x1, x2):
    # Mean absolute error
    return (torch.sum(torch.abs(x1 - x2)))


def climate_spatial_regularizer(model, K, device):
    reg = torch.tensor(0.).to(device)

    if 'low' not in type(model).__name__:
        # Make spatial dimension the 0th dimension
        w_unfold = unfold(model.w.detach(), mode=2).contiguous()
        reg.add_((K * pdist(w_unfold)).sum() / (torch.numel(model.w)))
    else:
        reg.add_((K * pdist(model.C.detach())).sum())

    return reg


def compareStats(y_train, y_val, preds_val):
    # Function computes model MSE/MAE and compares to several naïve approaches
    normal_preds = torch.zeros(y_val.shape)
    for i in np.arange(len(y_val)):
        normal_preds[i] = torch.normal(y_train.mean(), y_train.std())

    dumb_pred = torch.cat((y_val[0].unsqueeze(0), y_val[0:-1]))
    constant_pred = y_train.mean() * torch.ones(len(y_val))

    print('MSE')
    print('Model   : {:4f}'.format(mse(y_val, preds_val)))
    print('Constant: {:4f}'.format(mse(y_val, constant_pred)))
    print('Previous: {:4f}'.format(mse(y_val, dumb_pred)))
    print('Normal  : {:4f}'.format(mse(y_val, normal_preds)))

    print('MAE')
    print('Model   : {:4f}'.format(mae(y_val, preds_val)))
    print('Constant: {:4f}'.format(mae(y_val, constant_pred)))
    print('Previous: {:4f}'.format(mae(y_val, dumb_pred)))
    print('Normal  : {:4f}'.format(mae(y_val, normal_preds)))
