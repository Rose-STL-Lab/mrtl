import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.utils.data
import utils
import xarray as xr
from cp_als import cp_decompose
from data.climate.dataset import getData
from train.climate.model import my_regression, my_regression_low
from train.climate.multi import Multi
import cmocean

sns.set()

if __name__ == '__main__':

    save_results = True

    # Arguments Parse
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', dest='method')
    parser.add_argument('--data_dir', dest='data_dir')
    parser.add_argument('--save_dir', dest='save_dir')
    parser.add_argument('--experiment_name', dest='experiment_name')
    args = parser.parse_args()

    # Get paths and create directories if necessary
    data_fp = args.data_dir
    save_fp = args.save_dir
    save_fp = os.path.join(save_fp, args.experiment_name)
    if not os.path.isdir(save_fp):
        os.makedirs(save_fp)

    lead = 6
    # list of dims to use for MRTL (reverse order so that smallest is first)
    dims = [[4, 9], [8, 18], [12, 27], [24, 54], [40, 90], [60, 135],
            [80, 180]]

    #### Full-rank
    show_preds = False
    plot_weights = False
    period = 1

    stop_cond = 'val_loss'
    max_epochs = 2
    do_normalize_X = False
    do_normalize_y = True

    # DIMENSIONS TO TRAIN ON
    
    if args.method == 'mrtl':
        start_dim_idx = 0   # index of resolution to start training on 
        end_dim_idx = 1     # index of transition resolution
        new_end_dim_idx = 6 # index of final resolution
    elif args.method == 'fixed':
        start_dim_idx = 6
        end_dim_idx = 6
        new_end_dim_idx = 6
    
    # INSTANTIATE MODEL
    multi = Multi(batch_size=10, stop_cond=stop_cond)

    # HYPERPARAMETERS
    start_lr = .001
    step = 1
    gamma = .95
    multi.reg, multi.reg_coef = None, .003
    multi.spatial_reg = True
    multi.spatial_reg_coef = .25
    multi.counter_thresh = 4  # number of increases in loss before stopping

    multi.model = my_regression(lead=lead,
                                input_shape=dims[start_dim_idx],
                                output_shape=1)
    
    # SET LEARNING RATES ACROSS RESOLUTOINS
    ratios = [dims[idx + 1][0] / dims[idx][0] for idx in range(len(dims) - 1)]
    lrs = np.zeros(len(dims))
    for i, lr in enumerate(lrs):
        if not i:
            lrs[i] = start_lr
        else:
            lrs[i] = lrs[i - 1] / (ratios[i - 1]**2)

    random_seed = 1000

    # Track loss
    full_train_loss = []
    full_val_loss = []
    full_finegrain_epochs = []

    # for idx, dim in enumerate(dims[start_dim_idx:]):
    for idx in np.arange(start_dim_idx, end_dim_idx + 1):
        multi.optimizer = torch.optim.Adam(multi.model.parameters(),
                                           lrs[idx])  # set optimizer
        multi.scheduler = torch.optim.lr_scheduler.StepLR(multi.optimizer,
                                                          step_size=step,
                                                          gamma=gamma)
        dim = dims[idx]

        if multi.spatial_reg:
            multi.K = utils.create_kernel(dim, sigma=.05, device=multi.device)

        print('\n\nLearning for resolution {0}x{1} (FULL RANK)'.format(
            dim[0], dim[1]))

        # CREATE DATASET
        train_set, val_set, test_set = getData(dim,
                                               data_fp=data_fp,
                                               lead_time=lead,
                                               do_normalize_X=do_normalize_X,
                                               do_normalize_y=do_normalize_y,
                                               random_seed=random_seed,
                                               ppt_file='ppt_midwest.nc')
        #     break
        multi.init_loaders(train_set, val_set)  # initialize loaders

        # TRAIN
        multi.train(train_set,
                    val_set,
                    epochs=max_epochs,
                    period=period,
                    plot_weights=plot_weights)

        # FINEGRAIN
        if idx != end_dim_idx:
            full_finegrain_epochs.append(len(full_train_loss))

            # get ratio for number of gridpoints in next dims and current set
            ratio = (dims[idx + 1][0] / dims[idx][0])**2
            # Finegrain weights and update optimizer
            new_w = torch.nn.functional.interpolate(
                multi.model.w.clone().detach().cpu().reshape(lead, 2, *dim),
                size=dims[idx + 1],
                align_corners=False,
                mode='bilinear') / ratio
            new_w = new_w.reshape(lead, 2, -1)
            multi.model.w = torch.nn.Parameter(new_w.to(multi.device),
                                               requires_grad=True)

    # #### Low-rank
    max_epochs = 30

    if 'low' not in type(multi.model).__name__:
        w = multi.model.w.detach().clone()
        b_full = multi.model.b.detach().clone()

    # Factorize the weight tensor
    K = 2  # rank of the decomposition

    weights, factors = cp_decompose(w,
                                    K,
                                    max_iter=200,
                                    nonnegative=False,
                                    verbose=True)
    factors = [f * torch.pow(weights, 1 / len(factors)) for f in factors]
    # factors = [f * weights for f in factors]

    # initialize low-rank model
    multi.spatial_reg = False
    multi.spatial_reg_coef = .01
    multi.reg = None
    multi.model = my_regression_low(lead=lead,
                                    input_shape=dims[end_dim_idx],
                                    output_shape=1,
                                    K=K)
    multi.model.A = torch.nn.Parameter(factors[0].detach().clone(),
                                       requires_grad=True)
    multi.model.B = torch.nn.Parameter(factors[1].detach().clone(),
                                       requires_grad=True)
    multi.model.C = torch.nn.Parameter(factors[2].detach().clone(),
                                       requires_grad=True)
    multi.model.b = torch.nn.Parameter(b_full, requires_grad=True)

    low_rank_dims = dims[end_dim_idx:new_end_dim_idx + 1]
    lrs = np.zeros(new_end_dim_idx - end_dim_idx + 1)
    ratios = [
        low_rank_dims[idx + 1][0] / low_rank_dims[idx][0]
        for idx in range(len(low_rank_dims) - 1)
    ]

    for i, lr in enumerate(lrs):
        if not i:
            lrs[i] = 10e-5
        else:
            lrs[i] = lrs[i - 1] / (ratios[i - 1]**2)

    # Track loss
    low_train_loss = []
    low_val_loss = []
    low_finegrain_epochs = []

    for idx, dim in enumerate(low_rank_dims):
        multi.optimizer = torch.optim.Adam(multi.model.parameters(), lrs[idx])
        multi.scheduler = torch.optim.lr_scheduler.StepLR(multi.optimizer,
                                                          step_size=step,
                                                          gamma=gamma)

        if multi.spatial_reg:
            multi.K = utils.create_kernel(dim, sigma=.05, device=multi.device)

        print('\n\nLearning for resolution {0}x{1} (LOW RANK)'.format(
            dim[0], dim[1]))

        # CREATE DATASET
        # need to make sure train/val/test are the same for each loop
        train_set, val_set, test_set = getData(dim,
                                               data_fp=data_fp,
                                               lead_time=lead,
                                               do_normalize_X=do_normalize_X,
                                               do_normalize_y=do_normalize_y,
                                               random_seed=random_seed,
                                               ppt_file='ppt_midwest.nc')
        multi.init_loaders(train_set, val_set)  # initialize loaders

        # TRAIN
        multi.train(train_set,
                    val_set,
                    epochs=max_epochs,
                    period=period,
                    plot_weights=plot_weights)

        # FINEGRAIN
        # Just need to fine grain the spatial factor
        if idx != (len(low_rank_dims) - 1):
            #         low_finegrain_epochs.append(len(train_loss))

            # get ratio for number of gridpoints in next dim and current dim
            ratio = (low_rank_dims[idx + 1][0] / low_rank_dims[idx][0])**2
            new_C = multi.model.C.detach().clone().cpu().T
            new_C = new_C.reshape(K, *dim).unsqueeze(0)
            new_C = torch.nn.functional.interpolate(
                new_C,
                size=low_rank_dims[idx + 1],
                align_corners=False,
                mode='bilinear') / ratio
            new_C = new_C.squeeze(0).reshape(K, -1).T
            multi.model.C = torch.nn.Parameter(new_C.to(multi.device),
                                               requires_grad=True)

    # Plot (and save) Latent Factors
    temp = xr.open_dataarray(
        os.path.join(data_fp, 'sss_{}x{}.nc'.format(*dims[new_end_dim_idx])))
    latent_factors = multi.model.C.detach().cpu().numpy()
    if save_results:
        f = open(os.path.join(save_fp, 'mrtl_latent-factors.pkl'), 'wb')
        pickle.dump(latent_factors, f)
        f.close()

    max_abs = np.max(np.abs(latent_factors))  # for scaling
    for idx in np.arange(K):
        # TODO fix figures (GEOS errors)
        # contourf plot
        fig, ax = utils.plot_setup(plot_range=[-150, -40, 10, 56])
        cp = ax.contourf(temp.lon,
                         temp.lat,
                         latent_factors[:, idx].reshape(*dim),
                         np.linspace(-max_abs, max_abs, 15),
                         cmap='cmo.balance')
        fig.savefig(os.path.join(save_fp, f'latent-factor{idx}_contourf.png'),
                    dpi=200)
        # plt.show()

        # imshow plot
        fig, ax = plt.subplots()
        cp = ax.imshow(np.flipud(
            latent_factors[:, idx].reshape(*dims[new_end_dim_idx])),
                       vmin=-max_abs,
                       vmax=max_abs,
                       cmap='cmo.balance')
        cb = fig.colorbar(cp, orientation='vertical', fraction=.021)
        fig.savefig(os.path.join(save_fp, f'latent-factor{idx}_imshow.png'),
                    dpi=200)
#         plt.show()
