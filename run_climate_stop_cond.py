import argparse
import copy
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.utils.data
import utils
from cp_als import cp_decompose
from data.climate.dataset import getData
from train.climate.model import my_regression, my_regression_low
from train.climate.multi import Multi
import xarray as xr
import cmocean

sns.set()
if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', dest='data_dir')
    parser.add_argument('--save_dir', dest='save_dir')
    parser.add_argument('--experiment_name', dest='experiment_name')
    parser.add_argument('--n_trials', dest='n_trials', type=int)
    parser.add_argument('--K', dest='K', type=int)
    args = parser.parse_args()
    data_fp = args.data_dir
    save_fp = args.save_dir
    save_fp = os.path.join(save_fp, args.experiment_name)
    n_trials = args.n_trials
    K = args.K  # rank of CP decomposition
    if not os.path.isdir(save_fp):
        os.makedirs(save_fp)
    lead = 1
    # list of dims to use for MRTL (reverse order so that smallest is first)
    dims = [[4, 9], [8, 18], [12, 27], [24, 54], [40, 90], [60, 135],
            [80, 180]]
    period = 10
    plot_weights = False
    plot_results = False

    # Compute learning rates
    start_lr = .01  # Hyperparams for full_rank
    step = 1
    gamma = .95
    ratios = [dims[idx + 1][0] / dims[idx][0] for idx in range(len(dims) - 1)]
    lrs = np.zeros(len(dims))
    for i, lr in enumerate(lrs):
        if not i:
            lrs[i] = start_lr
        else:
            lrs[i] = lrs[i - 1] / (ratios[i - 1]**2)
    lrs = dict(list(zip([dim[0] for dim in dims], lrs)))
    all_data = dict()  # dictionary to hold results
    res = []

    for stop_cond in ['val_loss', 'grad_entropy', 'grad_var', 'grad_norm']:
        print(f'\nTesting {stop_cond} stopping condition')
        
        for trial in np.arange(n_trials) + 1:
            print(f'Trial {trial}/{n_trials}')
            time_res_lead=[1,4]
            # INSTANTIATE MODEL
            multi = Multi(batch_size=128, stop_cond=stop_cond)
            max_epochs_full = 5
            max_epochs_low = 30
            start_lr = .01
            step = 1
            gamma = .95
            # Full-rank regularization
            multi.reg, multi.reg_coef = 'l1', .0001
            multi.spatial_reg = True
            multi.spatial_reg_coef = .01
            multi.counter_thresh = 3  # number of increases in loss before stopping
            do_normalize_X = False # normalize inputs to [0, 1]?
            do_normalize_y = True  # normalize outputs to [0, 1]?
            lead = 1
            start_dim_idx = 0  # index of resolution to start training on
            end_dim_idx = 2  # index of transition resolution
            new_end_dim_idx = 6  # index of final resolution
            multi.model = my_regression(lead=lead,
                                            input_shape=dims[start_dim_idx],
                                            output_shape=1)
            random_seed = 1000
            # Track loss
            full_train_loss = []
            full_val_loss = []
            full_finegrain_epochs = []
            # FULL-RANK
            full_rank_dims = dims[:end_dim_idx + 1]
            for r in range(len(time_res_lead)):
                for idx, dim in enumerate(full_rank_dims):
                    multi.optimizer = torch.optim.Adam(multi.model.parameters(), lrs[dim[0]])  # set optimizer
                    multi.scheduler = torch.optim.lr_scheduler.StepLR(
                        multi.optimizer, step_size=step, gamma=gamma)
                    if multi.spatial_reg:
                        multi.K = utils.create_kernel(dim,
                                                      sigma=.05,
                                                      device=multi.device)
                    if idx > 0:
                        plot_weights = False  # choose whether to plot the weights
                    else:
                        plot_weights = False
                    # CREATE DATASET
                    train_set, val_set, test_set = getData(
                        dim,
                        data_fp=data_fp,
                        lead_time=time_res_lead[r],
                        do_normalize_X=do_normalize_X,
                        do_normalize_y=do_normalize_y,
                        random_seed=random_seed,
                        ppt_file='ppt_midwest.nc')
                    multi.init_loaders(train_set, val_set)  # initialize loaders
                    # TRAIN
                    multi.train(train_set,
                                val_set,
                                epochs=max_epochs_full,
                                period=period,
                                plot_weights=plot_weights)
                    # FINEGRAIN
                    if idx != end_dim_idx:
                        full_finegrain_epochs.append(len(full_train_loss))
                        # get ratio for number of gridpoints in next dims and current set
                        ratio = (dims[idx + 1][0] / dims[idx][0])**2
                        # Finegrain weights and update optimizer
                        new_w = torch.nn.functional.interpolate(
                            multi.model.w.clone().detach().cpu().reshape(
                                time_res_lead[r], 2, *dim),
                            size=dims[idx + 1],
                            mode='bilinear') / ratio
                        new_w = new_w.reshape(time_res_lead[r], 2, -1)
                        multi.model.w = torch.nn.Parameter(new_w.to(multi.device),
                                                           requires_grad=True)
                    if (idx == end_dim_idx):
                        if (time_res_lead[r]!=4):
                            full_finegrain_epochs.append(len(full_train_loss))
                            ratio = ((np.prod([dims[0][0],dims[0][1],time_res_lead[r+1]])) / (np.prod([dims[end_dim_idx][0],dims[end_dim_idx][1],time_res_lead[r]])))
                            new_w = torch.nn.functional.interpolate(
                                multi.model.w.reshape(
                                    1, 2,time_res_lead[r], *dim),
                                size=[time_res_lead[r+1], *dims[0]],
                                align_corners=False,
                                mode='trilinear') /ratio
                            new_w = new_w.reshape(time_res_lead[r+1], 2, -1)
                            multi.model.w = torch.nn.Parameter(new_w.to(multi.device),
                                                               requires_grad=True)
            # LOW RANK
            w = multi.model.w.detach().clone()
            b_full = multi.model.b.detach().clone()
            # Factorize the weight tensor
            weights, factors = cp_decompose(w,
                                            K,
                                            max_iter=200,
                                            nonnegative=False,
                                            verbose=True)
            factors = [
                f * torch.pow(weights, 1 / len(factors)) for f in factors
            ]
            # initialize low-rank model
            multi.spatial_reg = False
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
            # Track loss
            low_train_loss = []
            low_val_loss = []
            low_finegrain_epochs = []
            time_res_lead=[4,12]
            for j in range(len(time_res_lead)):
                for idx, dim in enumerate(low_rank_dims):
                    multi.optimizer = torch.optim.Adam(multi.model.parameters(),lrs[dim[0]])
                    multi.scheduler = torch.optim.lr_scheduler.StepLR(
                        multi.optimizer, step_size=step, gamma=gamma)
                    if multi.spatial_reg:
                        multi.K = utils.create_kernel(dim,
                                                      sigma=.05,
                                                      device=multi.device)
                    # CREATE DATASET
                    # need to make sure train/val/test are the same for each loop
                    train_set, val_set, test_set = getData(
                        dim,
                        data_fp=data_fp,
                        lead_time=time_res_lead[j],
                        do_normalize_X=do_normalize_X,
                        do_normalize_y=do_normalize_y,
                        random_seed=random_seed,
                        ppt_file='ppt_midwest.nc')
                    multi.init_loaders(train_set, val_set)  # initialize loaders
                    # TRAIN
                    multi.train(train_set,
                                val_set,
                                epochs=max_epochs_low,
                                period=period,
                                plot_weights=plot_weights)
                    # FINEGRAIN
                    # Just need to fine grain the spatial factor
                    if idx != (len(low_rank_dims) - 1):
                        # get ratio for number of gridpoints in next dim and
                        # current dim
                        ratio = (low_rank_dims[idx + 1][0] /
                                 low_rank_dims[idx][0])**2
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
                    if (idx == (len(low_rank_dims) - 1) and (time_res_lead[j]!=time_res_lead[1])):
                        latent_factors = multi.model.A.detach().cpu().numpy()
                        lf_four=[]
                        # get a list of the different latent factors in "A"
                        for idx in np.arange(K):
                            lf_four.append(latent_factors[:,idx])
                        ratio_A = ((time_res_lead[j+1]) / (time_res_lead[j]))
                        ratio_C = (low_rank_dims[0][0] /low_rank_dims[-1][0])**2
                        new_A = multi.model.A.detach().clone().cpu().T
                        # since were just changing the lead value, the size is a 1D 
                        # list with the new lead value
                        new_A = new_A.reshape(1,K,time_res_lead[j])
                        new_A = torch.nn.functional.interpolate(new_A,size=[time_res_lead[j+1]],
                            align_corners=False,mode='linear') / ratio_A
                        new_A = new_A.squeeze().T
                        multi.model.A = torch.nn.Parameter(new_A.to(multi.device),
                                                           requires_grad=True)
                        new_C = multi.model.C.detach().clone().cpu().T
                        new_C = new_C.reshape(K, *dim).unsqueeze(0)
                        new_C = torch.nn.functional.interpolate(
                                new_C,
                                size=low_rank_dims[0],
                                align_corners=False,
                                mode='bilinear') / ratio_C
                        new_C = new_C.squeeze(0).reshape(K, -1).T
                        multi.model.C = torch.nn.Parameter(new_C.to(multi.device),
                                                               requires_grad=True)

            
            data = {
                'time': multi.train_times,
                'train_loss': multi.train_loss,
                'val_loss': multi.val_loss,
                'finegrain_times': multi.finegrain_times
            }

            all_data[trial] = copy.deepcopy(data)
            res.append({
                'stop_cond': stop_cond,
                'trial': trial,
                'full_time': multi.train_times['full'][-1],
                'low_time': multi.train_times['low'][-1],
                'full_val_loss': multi.val_loss['full'][-1],
                'low_val_loss': multi.val_loss['low'][-1]
            })

        # Save results for given stopping condition
        f = open(os.path.join(save_fp, f'results_{stop_cond}.pkl'), 'wb')
        pickle.dump(all_data, f)
        f.close()

    # Save results for all stopping conditions
    f = open(os.path.join(save_fp, f'results_all_stop_conds.pkl'), 'wb')
    pickle.dump(pd.DataFrame(res).set_index('trial'), f)
    f.close()

    # ### Load data for analysis
    f = open(os.path.join(save_fp, 'results_all_stop_conds.pkl'), 'rb')
    summary_stats = pickle.load(f)
    f.close()

    # #### view statistics
    print('Mean:')
    print(summary_stats.groupby('stop_cond').mean())
    print('\nStd:')
    print(summary_stats.groupby('stop_cond').std())
    # Plot the latent_factor A
    latent_factors = multi.model.A.detach().cpu().numpy()
    for idx in np.arange(K):
        sns.distplot(latent_factors[:,idx]);
        sns_plot=sns.distplot(lf_four[idx]).legend(title='Resolutions',labels=['Months','Seasons']);
        sns_plot.figure.savefig(os.path.join(save_fp, f'latent-factor_a_{idx}.png'))
        plt.clf()

    latent_factors = multi.model.C.detach().cpu().numpy()
    temp = xr.open_dataarray(
        os.path.join(data_fp, 'sss_{}x{}.nc'.format(*dims[new_end_dim_idx][:2])))
    latent_factors = multi.model.C.detach().cpu().numpy()
    save_results = True
    if save_results:
        f = open(os.path.join(save_fp, 'mrtl_latent-factors.pkl'), 'wb')
        pickle.dump(latent_factors, f)
        f.close()
    # PLOT RESULTS
    for idx in np.arange(K):
        plt.savefig(os.path.join(save_fp, f'latent-factor{idx}_contourf.png'))
        # contourf plot
        max_abs = np.max(np.abs(latent_factors[:,idx]))  # for scaling
        fig, ax = utils.plot_setup(plot_range=[-150, -40, 10, 56])
        cp = ax.contourf(temp.lon,
                         temp.lat,
                         latent_factors[:, idx].reshape(*dim),
                         np.linspace(-max_abs, max_abs, 15),
                         cmap='cmo.balance')
        fig.savefig(os.path.join(save_fp, f'latent-factor{idx}_contourf.png'),
                    dpi=200,
                    bbox_inches='tight')
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
                    dpi=200,
                    bbox_inches='tight')
    # Plot loss curves for each stopping condition
    for stop_cond in ['val_loss', 'grad_entropy', 'grad_var', 'grad_norm']:
        f = open(os.path.join(save_fp, f'results_{stop_cond}.pkl'), 'rb')
        results = pickle.load(f)
        f.close()
        fig, ax = plt.subplots()
        for trial in np.arange(len(results.keys())) + 1:
            ax.plot(results[trial]['time']['full'],
                    results[trial]['val_loss']['full'],
                    label=trial)
            #for t in results[trial]['finegrain_times']['full'][:-1]:
                #ax.axvline(t)
        ax.legend()
        ax.set_title(f'Full rank loss ({stop_cond})')
        ax.set_ylabel('Loss (MSE)')
        ax.set_xlabel('Time (s)')
        fig.savefig(os.path.join(save_fp, f'full-rank-loss_{stop_cond}.png'),
                    dpi=200, bbox_inches='tight')
        fig, ax = plt.subplots()
        for trial in np.arange(len(results.keys())) + 1:
            ax.plot(results[trial]['time']['low'],
                    results[trial]['val_loss']['low'],
                    label=trial)
            #for t in results[trial]['finegrain_times']['low'][:-1]:
                #ax.axvline(t)
        ax.legend()
        ax.set_title(f'Low rank loss ({stop_cond})')
        ax.set_ylabel('Loss (MSE)')
        ax.set_xlabel('Time (s)')
        fig.savefig(os.path.join(save_fp, f'low-rank-loss_{stop_cond}.png'),
                    dpi=200, bbox_inches='tight')