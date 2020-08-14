import argparse
import os
import pickle

import cmocean
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.utils.data
import xarray as xr

import utils
from cp_als import cp_decompose
from data.climate.dataset import getData
from train.climate.model import my_regression, my_regression_low
from train.climate.multi import Multi

sns.set()


if __name__ == '__main__':

    save_results = True

    # Arguments Parse
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', dest='method')
    parser.add_argument('--K', dest='K', type=int)
    parser.add_argument('--data_dir', dest='data_dir')
    parser.add_argument('--save_dir', dest='save_dir')
    parser.add_argument('--experiment_name', dest='experiment_name')
    args = parser.parse_args()
    K = args.K  # rank of CP decomposition
    
    # Get paths and create directories if necessary
    data_fp = args.data_dir
    save_fp = args.save_dir
    save_fp = os.path.join(save_fp, args.experiment_name)
    if not os.path.isdir(save_fp):
        os.makedirs(save_fp)

    # list of dims to use for MRTL (reverse order so that smallest is first)
    dims = [[4, 9], [8, 18], [12, 27], [24, 54], [40, 90], [60, 135],
            [80, 180]]


    period = 1 # how often to display loss (every 'period' epochs)
    plot_weights = False
    random_seed = 1000 
    
    # INSTANTIATE MODEL
    stop_cond = 'val_loss'

    multi = Multi(batch_size=10, stop_cond=stop_cond)


    # SET HYPERPARAMETERS (based on method)
    if args.method == 'mrtl':
        print('mrtl outside for loop')
      
        max_epochs_full = 5
        max_epochs_low = 30
        start_lr = .001
        step = 1
        gamma = .95
        multi.reg, multi.reg_coef = 'l1', .003
        multi.spatial_reg = True
        multi.spatial_reg_coef = 10
        multi.counter_thresh = 3  # number of increases in loss before stopping
        do_normalize_X = False # normalize inputs to [0, 1]?
        do_normalize_y = True  # normalize outputs to [0, 1]?
        lead =4
        start_dim_idx = 0  # index of resolution to start training on
        end_dim_idx = 2  # index of transition resolution
        new_end_dim_idx = 6  # index of final resolution
        multi.model = my_regression(lead=lead,
                                    input_shape=dims[start_dim_idx],
                                    output_shape=1)
    elif args.method == 'fixed':

        print('fixed outside for loop')
      
        max_epochs_full = 5
        max_epochs_low = 50
        start_lr = .001
        step = 1
        gamma = .95
        multi.reg, multi.reg_coef = 'l1', .003
        multi.spatial_reg = True
        multi.spatial_reg_coef = 10
        multi.counter_thresh = 3  # number of increases in loss before stopping
        do_normalize_X = False # normalize inputs to [0, 1]?
        do_normalize_y = True  # normalize outputs to [0, 1]?
        lead = 4
        start_dim_idx = 6
        end_dim_idx = 6
        new_end_dim_idx = 6
        multi.model = my_regression(lead=lead,
                                    input_shape=dims[start_dim_idx],
                                    output_shape=1)
    elif args.method == 'random':
        print('random outside for loop')
            
        max_epochs_full = 5
        max_epochs_low = 70
        start_lr = .0001
        step = 1
        gamma = .95
        multi.reg, multi.reg_coef = 'l1', .003
        multi.spatial_reg = True
        multi.spatial_reg_coef = 10
        multi.counter_thresh = 3  # number of increases in loss before stopping
        do_normalize_X = False # normalize inputs to [0, 1]?
        do_normalize_y = True  # normalize outputs to [0, 1]?
        lead = 4
        start_dim_idx = 6
        end_dim_idx = 6
        new_end_dim_idx = 6
        multi.model = my_regression_low(lead=lead,
                                        input_shape=dims[start_dim_idx],
                                        output_shape=1,
                                        K=K)

    # SET LEARNING RATES ACROSS RESOLUTOINS
    ratios = [dims[idx + 1][0] / dims[idx][0] for idx in range(len(dims) - 1)]
    lrs = np.zeros(len(dims))

    if args.method != 'random':
        print('not random outside for loop')
              
        for i, lr in enumerate(lrs):
            if not i:
                lrs[i] = start_lr
            else:
                lrs[i] = lrs[i - 1] / (ratios[i - 1]**2)

        # Track loss
        full_train_loss = []
        full_val_loss = []
        full_finegrain_epochs = []
        time_res=["year","season","month"]
        ##each value in this list represents the number of months you have to add to the previous date to get to the next date
        #after the datset is aggregated
        time_res_lead=[1,4,12]
        count=0
        for r in range(len(time_res_lead)):
            for i, lr in enumerate(lrs):
                if not i:
                    lrs[i] = start_lr
                else:
                    lrs[i] = lrs[i - 1] / (ratios[i - 1]**2)

            # Track loss
            full_train_loss = []
            full_val_loss = []
            full_finegrain_epochs = []
            
            
      

            if args.method != 'random':
              
                for idx in np.arange(start_dim_idx, end_dim_idx + 1):
                    
                    
                    multi.optimizer = torch.optim.Adam(multi.model.parameters(),
                                                       lrs[idx])  # set optimizer
                    multi.scheduler = torch.optim.lr_scheduler.StepLR(multi.optimizer,
                                                                      step_size=step,
                                                                      gamma=gamma)
                    dim = dims[idx]

                    if multi.spatial_reg:
                        multi.K = utils.create_kernel(dim,
                                                      sigma=.05,
                                                      device=multi.device)

                    print('\n\nLearning for resolution {0}x{1} (FULL RANK)'.format(
                        dim[0], dim[1]))

                    # CREATE DATASET
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
                                epochs=max_epochs_full,
                                period=period,
                                plot_weights=plot_weights)
                                        


                    # FINEGRAIN
                    if idx != end_dim_idx:


                        full_finegrain_epochs.append(len(full_train_loss))
                        
                        # get ratio for number of gridpoints in
                        # next dims and current set
                        #redefine,dims and add another dimension, and do it the same way
                        #ratio: we take the dot product in regression()multiply all the coefficients times all of thhe actual values and sum them
                        #so if the scale factor is 2, and you have twice as many coefficients , then the output is going to be twice as large
                        #because you're multiplying two times as many things, so you have to divide by ration
                        #if we increase the number of weights by a factor of 2, we shhould divide all thhe weights by 2, so we don't double the outputs
                        #before finegraining, we might predic the precipitation is 200, then if we finegrain, we hahve twice as many weights, and we would predict 400
                        ratio = (dims[idx + 1][0] / dims[idx][0])**2
                        
                        # Finegrain weights and update optimizer
                        #multi.model.w is getting w from the model. w is the weight coefficient tensot
                        #clone().detach().cpu() is creating a copy of the weight so we don't modify it in any way
                        #it's detaching it from the computation graphh, which is keeping track of all the gradients.
                        #then it's moving ti from the GPU to the CPU
                        #to do interpolation,we want to have lat and lon as separate dimensions. so we're reshaping it so
                        #that lat and lon have separate dimensions
                        #"size" is specifying the new size that we want to interpolate to 
                        #in this case,dimensions was the new set of spatial dimensions. 
                        #so if were going from 4x9 to 8x18
                        #we're doing three dimesnions of interpolation b/c I'm doing lat,lon and time,
                        #instead of the size being 8x18, it will be 4x8x18, if you're interpolating froma  year to seasons
                        #scale_factor tells you the multiplying factor
                        #size tells you the size that you want to interpolate to 
                


                        new_w = torch.nn.functional.interpolate(
                            multi.model.w.clone().detach().cpu().reshape(
                                lead, 2, *dim),
                            size=dims[idx + 1],
                            align_corners=False,
                            mode='bilinear') / ratio
                        
                        #get "w" from the model. multi is the object that contains the model.
                        #w is the weight coefficient tensor 
                       
                        #in this case 
                        #since lat and lon are not separate dimensions(they are 1 dimension), they are flattended
                        #the -1 is telling Pytorch to reshape the new weight tensor with first dimension of lead. 
                        #-1 multiplies the lat and lon together
                        new_w = new_w.reshape(lead, 2, -1)
                        
                        multi.model.w = torch.nn.Parameter(new_w.to(multi.device),
                                                           requires_grad=True)

                        
                        #train on the full rank model,which has w as the coefficient. the low rank model(instead of having a full 6x2x36 dimension)
                        #the low rank model has three vectors corresponding to each of the dimensions. This is the CP decomposition
                        #train the full rank model, which has this full weight tensor, then we decompose it(do tensor decomposition) and we factorize 
                        #it into these vectors. The purpose of that is taht thhe vecgtors hhave less parameters/data to worry about
                        #after we factorize it to a low rank model, we train that separately
                        #the purpose of the full rank model is to initialize the low rank model.
                        # we can do everything on the low rank model, but we have to randomly initilize the weights
                        #The problem with doing that for low rank models is when you initialize them randomly, it's hard to get good results 
                        #You need a good initilization. Initialize the low rank model with the full rank model. 
                        #Train the full rank model, and we have some coefficients for the weight that are resonaably good on the coarse data. Then start the low rank training now,
                        #and factorize the full rank model , to get the intilial weights , then train the low rank model
                    if (idx == end_dim_idx):
                        if (time_res_lead[r]!=12):
                            full_finegrain_epochs.append(len(full_train_loss))



                            ratio = ((np.prod([dims[0][0],dims[0][1],time_res_lead[r+1]])) / (np.prod([dims[end_dim_idx][0],dims[end_dim_idx][1],time_res_lead[r]])))
                            
                            
                            

                            new_w = torch.nn.functional.interpolate(
                                multi.model.w.reshape(
                                    lead, 2, *dim),
                                size=dims[0],
                                align_corners=False,
                                mode='bilinear') / ratio
                            new_w = new_w.reshape(lead, 2, -1)
                            multi.model.w = torch.nn.Parameter(new_w.to(multi.device),
                                                               requires_grad=True)
                            for idx in np.arange(start_dim_idx, end_dim_idx + 1):

                               

                                multi.optimizer = torch.optim.Adam(multi.model.parameters(),
                                                           lrs[idx])  # set optimizer
                                multi.scheduler = torch.optim.lr_scheduler.StepLR(multi.optimizer,
                                                                                  step_size=step,
                                                                                  gamma=gamma)
                                dim = dims[idx]

                                if multi.spatial_reg:
                                    multi.K = utils.create_kernel(dim,
                                                                  sigma=.05,
                                                                  device=multi.device)

                                print('\n\nLearning for resolution {0}x{1} (FULL RANK)'.format(
                                    dim[0], dim[1]))

                                # CREATE DATASET

                                train_set, val_set, test_set = getData(
                                    dims[idx],
                                    data_fp=data_fp,
                                    lead_time=4,
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
                                print("worked")
                                if idx != end_dim_idx:


                                    full_finegrain_epochs.append(len(full_train_loss))
                                    

                                    ratio = (dims[idx + 1][0] / dims[idx][0])**2
                                  


                                    new_w = torch.nn.functional.interpolate(
                                        multi.model.w.clone().detach().cpu().reshape(
                                            4, 2, *dim),
                                        size=dims[idx + 1],
                                        align_corners=False,
                                        mode='bilinear') / ratio
                                    
                                    #get "w" from the model. multi is the object that contains the model.
                                    #w is the weight coefficient tensor 
                                   
                                    #in this case 
                                    #since lat and lon are not separate dimensions(they are 1 dimension), they are flattended
                                    #the -1 is telling Pytorch to reshape the new weight tensor with first dimension of lead. 
                                    #-1 multiplies the lat and lon together
                                    new_w = new_w.reshape(4, 2, -1)
                                    
                                    multi.model.w = torch.nn.Parameter(new_w.to(multi.device),
                                                                       requires_grad=True)

                                            




                    

                


                

                # CP DECOMPOSITION
                w = multi.model.w.detach().clone()
                b_full = multi.model.b.detach().clone()

                # CP Decomposition
                weights, factors = cp_decompose(w,
                                                K,
                                                max_iter=200,
                                                nonnegative=False,
                                                verbose=True)
                factors = [f * torch.pow(weights, 1 / len(factors)) for f in factors]
                # factors = [f * weights for f in factors]

                # initialize low-rank model
                multi.model = my_regression_low(lead=lead,
                                                input_shape=dims[end_dim_idx],
                                                output_shape=1,
                                                K=K)
                #these are the vectors of the low rank model. If you have two matrices and you take their outerproduct: 5x1 *1x5. 
                #So the the ouput is 5x5. When you multiply them to gethher thhe size is AXC. 
                multi.model.A = torch.nn.Parameter(factors[0].detach().clone(),
                                                   requires_grad=True)
                multi.model.B = torch.nn.Parameter(factors[1].detach().clone(),
                                                   requires_grad=True)
                multi.model.C = torch.nn.Parameter(factors[2].detach().clone(),
                                                   requires_grad=True)
                multi.model.b = torch.nn.Parameter(b_full, requires_grad=True)

            # Begin low-rank training
            multi.spatial_reg = False
            multi.spatial_reg_coef = .01
            multi.reg = None

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
    
        for j in range(len(time_res_lead)):

            #if r

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
                            epochs=max_epochs_low,
                            period=period,
                            plot_weights=plot_weights)

                # FINEGRAIN
                # Just need to fine grain the spatial factor
                #this piece of code changes the shape of C to same dimensions as the next resolution
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
                #once the spatial resolution is [12,27],which is the last full rank spatial resoltuion, the resolution changes to[4,9], whhich is the first full rank
                #resoltuion
                #once the inner loop ends, the temporal resolution changes to the next resolution, which is year(1),to seasons(4), 
                    
                if (idx == (len(low_rank_dims) - 1) and (time_res_lead[j]!=12)):
                    #         low_finegrain_epochs.append(len(train_loss))
                    
                    # get ratio for number of gridpoints in next dim and current dim
                    #ratio = (low_rank_dims[-1][0] / low_rank_dims[0][0])**2
                    #ratio = np.prod([new_spatial_dim[0], new_spatial_dim[1], new_time_res]) / np.prod([old_spatial_dim[0], old_spatial_dim[1], old_time_res])
                    #if (time_res_lead[r]!=12) and (idx != (len(low_rank_dims) - 1)):
                    
                    ratio = ((np.prod([low_rank_dims[0][0],low_rank_dims[0][1],time_res_lead[j+1]])) / (np.prod([low_rank_dims[-1][0],low_rank_dims[-1][1],time_res_lead[j]])))
                        

                    new_C = multi.model.C.detach().clone().cpu().T
                    new_C = new_C.reshape(K, *dim).unsqueeze(0)
                    new_C = torch.nn.functional.interpolate(
                            new_C,
                            size=low_rank_dims[0],
                            align_corners=False,
                            mode='bilinear') / ratio
                        
                    new_C = new_C.squeeze(0).reshape(K, -1).T

                        
                    
                    multi.model.C = torch.nn.Parameter(new_C.to(multi.device),
                                                           requires_grad=True)
              
            #if idx == (len(low_rank_dims) - 1):
            


    
    print('\n\nRESULTS:')
    if args.method != 'random':
        print(f"Training time for full rank (s): {multi.train_times['full'][-1]:.4f}")
    else:
        print(f"Training time for full rank (s): N/A (low rank is initialized randomly)")
    print(f"Training time for low rank (s):  {multi.train_times['low'][-1]:.4f}")
    
    if args.method != 'random':
        print(f"Val loss after full rank (MSE):  {np.min(multi.val_loss['full'][-1]):.4f}")
    else:
        print(f"Val loss after full rank (MSE):  N/A (low rank is initialized randomly)")
    print(f"Val loss after low rank (MSE):   {np.min(multi.val_loss['low'][-1]):.4f}")
    print('\n')

    # Plot (and save) Latent Factors
    
    temp = xr.open_dataarray(
        os.path.join(data_fp, 'sss_{}x{}.nc'.format(*dims[new_end_dim_idx][:2])))
    latent_factors = multi.model.C.detach().cpu().numpy()
    if save_results:
        f = open(os.path.join(save_fp, 'mrtl_latent-factors.pkl'), 'wb')
        pickle.dump(latent_factors, f)
        f.close()

    # PLOT RESULTS
    for idx in np.arange(K):
        
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