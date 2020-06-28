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

# from plotter import plot_loss, plot_w, plotActualPreds

sns.set()

if __name__ == '__main__':

    # Set directories
    # data_fp = './data/'
    # save_fp = './results/stopping-cond/'
    save_results = True

    # Arguments Parse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', dest='data_dir')
    parser.add_argument('--save_dir', dest='save_dir')
    parser.add_argument('--experiment_name', dest='experiment_name')
    parser.add_argument('--n_trials', dest='n_trials', type=int)
    args = parser.parse_args()
    data_fp = args.data_dir
    save_fp = args.save_dir

    save_fp = os.path.join(save_fp, args.experiment_name)
    n_trials = args.n_trials

    if not os.path.isdir(save_fp):
        os.mkdir(save_fp)

    lead = 6
    # list of dims to use for MRTL (reverse order so that smallest is first)
    dims = [[4, 9], [8, 18], [12, 27], [24, 54], [40, 90], [60, 135],
            [80, 180]]
    # dims = [[4, 9], [8, 18], [40, 90], [80, 180]]
    #     dims = [[8,18]]

    # FULL RANK
    start_dim_idx = 0
    end_dim_idx = 0
    full_max_epochs = 7
    low_max_epochs = 100
    do_normalize_X = False
    do_normalize_y = True

    period = 10
    plot_weights = False
    plot_results = False

    # Compute learning rates
    start_lr = .001  # Hyperparams for full_rank
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

            # INSTANTIATE MODEL
            multi = Multi(batch_size=10, stop_cond=stop_cond)
            multi.reg, multi.reg_coef = None, .003
            multi.spatial_reg = True
            multi.spatial_reg_coef = .25
            multi.counter_thresh = 4
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

            for idx, dim in enumerate(full_rank_dims):
                multi.optimizer = torch.optim.Adam(
                    multi.model.parameters(), lrs[dim[0]])  # set optimizer
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

                # print('\n\nLearning for resolution {0}x{1} (FULL RANK)'.format(
                # dim[0], dim[1]))

                # CREATE DATASET
                # need to make sure train/val/test are the same for each loop
                train_set, val_set, test_set = getData(
                    dim,
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
                            epochs=full_max_epochs,
                            period=period,
                            plot_weights=plot_weights)

                # # Plot predictions
                # if plot_results:
                #     y_train, preds_train = multi.getPreds(train_set)
                #     y_val, preds_val = multi.getPreds(val_set)

                #     fig, ax = plotActualPreds(y_train,
                #                               preds_train,
                #                               title='Training')
                #     plt.show()
                #     fig, ax = plotActualPreds(y_val,
                #                               preds_val,
                #                               title='Validation')
                #     plt.show()

                # FINEGRAIN
                if idx != end_dim_idx:
                    full_finegrain_epochs.append(len(full_train_loss))

                    # get ratio for number of gridpoints in next dims and
                    # current set
                    ratio = (dims[idx + 1][0] / dims[idx][0])**2
                    # Finegrain weights and update optimizer
                    new_w = multi.model.w.detach().cpu().numpy().reshape(
                        lead, 2, *dim)
                    new_w = torch.nn.functional.interpolate(
                        multi.model.w.clone().detach().cpu().reshape(
                            lead, 2, *dim),
                        size=dims[idx + 1],
                        mode='bilinear') / ratio
                    new_w = new_w.reshape(lead, 2, -1)
                    multi.model.w = torch.nn.Parameter(new_w.to(multi.device),
                                                       requires_grad=True)

            # LOW RANK
    #         print('Begin low rank training')
            max_epochs = 30
            period = low_max_epochs + 1

            new_end_dim_idx = 0

            if 'low' not in type(multi.model).__name__:
                w = multi.model.w.detach().clone()
                b_full = multi.model.b.detach().clone()

            # Factorize the weight tensor
            K = 3  # rank of the decomposition

            weights, factors = cp_decompose(w,
                                            K,
                                            max_iter=200,
                                            nonnegative=False,
                                            verbose=True)
            factors = [
                f * torch.pow(weights, 1 / len(factors)) for f in factors
            ]
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

            # Track loss
            low_train_loss = []
            low_val_loss = []
            low_finegrain_epochs = []

            for idx, dim in enumerate(low_rank_dims):
                multi.optimizer = torch.optim.Adam(multi.model.parameters(),
                                                   lrs[dim[0]])
                multi.scheduler = torch.optim.lr_scheduler.StepLR(
                    multi.optimizer, step_size=step, gamma=gamma)

                if multi.spatial_reg:
                    multi.K = utils.create_kernel(dim,
                                                  sigma=.05,
                                                  device=multi.device)

                # print('\n\nLearning for resolution {0}x{1} (LOW RANK)'.format(
                #     dim[0], dim[1]))

                # CREATE DATASET
                # need to make sure train/val/test are the same for each loop
                train_set, val_set, test_set = getData(
                    dim,
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
                            epochs=low_max_epochs,
                            period=period,
                            plot_weights=plot_weights)

                # # Plot predictions
                # if plot_results:
                #     y_train, preds_train = multi.getPreds(train_set)
                #     y_val, preds_val = multi.getPreds(val_set)

                #     fig, ax = plotActualPreds(y_train,
                #                               preds_train,
                #                               title='Training')
                #     plt.show()
                #     fig, ax = plotActualPreds(y_val,
                #                               preds_val,
                #                               title='Validation')
                #     plt.show()

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
        if save_results:
            f = open(os.path.join(save_fp, f'results_{stop_cond}.pkl'), 'wb')
            pickle.dump(all_data, f)
            f.close()

    # Save results for all stopping conditions
    if save_results:
        f = open(os.path.join(save_fp, f'results_all_stop_conds.pkl'), 'wb')
        pickle.dump(pd.DataFrame(res).set_index('trial'), f)
        f.close()

    # ### Load data for analysis
    f = open(os.path.join(save_fp, 'results_all_stop_conds.pkl'), 'rb')
    summary_stats = pickle.load(f)
    f.close()

    f = open(os.path.join(save_fp, f'results_{stop_cond}.pkl'), 'rb')
    val_loss_results = pickle.load(f)
    f.close()

    #     # Pre-computed results
    #     f = open('../results/all_stop_conds_df_thresh_4.pkl','rb')
    #     summary_stats = pickle.load(f)
    #     f.close()

    #     f = open('../results/{}_temp3.pkl'.format('val_loss'),'rb')
    #     val_loss_results = pickle.load(f)
    #     f.close()

    # #### view statistics
    print('Mean:')
    print(summary_stats.groupby('stop_cond').mean())
    print('\nStd:')
    print(summary_stats.groupby('stop_cond').std())

    fig, ax = plt.subplots()
    for trial in np.arange(len(val_loss_results.keys())) + 1:
        ax.plot(val_loss_results[trial]['time']['full'],
                val_loss_results[trial]['val_loss']['full'],
                label=trial)
        for t in val_loss_results[trial]['finegrain_times']['full'][:-1]:
            ax.axvline(t)
    ax.legend()
    ax.set_title('Full rank loss')
    ax.set_ylabel('Loss (MSE)')
    ax.set_xlabel('Time (s)')
    # TODO savefig instead plt.show()
    plt.show()

    fig, ax = plt.subplots()
    for trial in np.arange(len(val_loss_results.keys())) + 1:
        ax.plot(val_loss_results[trial]['time']['low'],
                val_loss_results[trial]['val_loss']['low'],
                label=trial)
    ax.legend()
    ax.set_title('Low rank loss')
    ax.set_ylabel('Loss (MSE)')
    ax.set_xlabel('Time (s)')
    # TODO savefig instead plt.show()
    plt.show()
