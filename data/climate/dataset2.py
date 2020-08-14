# Source: https://github.com/ktcarr/salinity-corn-yields/tree/master/mrtl
import numpy as np
import pandas as pd
import torch
import xarray as xr
from dateutil.relativedelta import relativedelta
from utils import diff_detrend_xr, preprocess
import os
import math


class ClimateDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_fp,
                 X,
                 y,
                 lead,
                 add_noise=False,
                 noise_std=.1,
                 X_min=None,
                 X_max=None,
                 X_mean=None,
                 X_std=None,
                 y_min=None,
                 y_max=None,
                 y_mean=None,
                 y_std=None,
                 do_remove_season_X=True,
                 do_normalize_X=True,
                 do_remove_season_y=True,
                 do_normalize_y=True):
        """
        Initialize climate dataset
        X_fn is filename of features (e.g., salinity data)
            – a 3-dimensional array, with time/lat/lon dimensions
        y_fn is filename of target (e.g., midwest precipitation)
        start_year and end_year are the first/last years to include in dataset
        lead is the number of months of data (minus 1) to include for each
        sample as input
        If we are predicting September precip, and lead=3, then input months
        are June, July, August, September
        i.e., there are three months of data preceding the prediction month
        """

        self.lead = lead
        self.input_shape = X[0].shape

        # PRE-PROCESS
        self.X, self.X_mean, self.X_std, self.X_min, self.X_max = preprocess(
            data_fp,
            X,
            old_min=X_min,
            old_max=X_max,
            mean=X_mean,
            std=X_std,
            do_remove_season=do_remove_season_X,
            do_normalize=do_normalize_X)
        if add_noise:  # add noise to training data if desired
            self.X = self.X + np.random.normal(
                size=self.X.shape, loc=0, scale=noise_std)

        self.y, self.y_mean, self.y_std, self.y_min, self.y_max = preprocess(
            data_fp,
            y,
            old_min=y_min,
            old_max=y_max,
            mean=y_mean,
            std=y_std,
            do_remove_season=do_remove_season_y,
            do_normalize=do_normalize_y)

        self.X = self.X.sortby('time')
        self.y = self.y.sortby('time')

        self.time_values = pd.to_datetime(self.X.time.values) 
        print("self.time_values")
        print(self.time_values)
        print("len(self.time_values)")
        print(len(self.time_values))
        #print("len(self.time_values)")
        #self.valid_time_idx = list(range(0,self.time_values))
        #for idx, time in enumerate(self.time_values):
          #f time + relativedelta(months=+self.lead) in self.time_values:
            #print("time + relativedelta(months=+self.lead)")
            #print(time + relativedelta(months=+self.lead))


        
        self.valid_time_idx = [idx for idx, time in enumerate(self.time_values) if time + relativedelta(months=+self.lead) in self.time_values]
        print("self.valid_time_idx in __init_")
        print(self.valid_time_idx)
        print("len(self.valid_time_idx) in __init_")
        print(len(self.valid_time_idx))
        self.X = self.X.fillna(0)  # Fill NaN values with 0
        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.float32)

    def __len__(self):
        #print("(len(self.time_values))")
        #print((len(self.time_values)))
        return (len(self.valid_time_idx))
        #return (len(self.time_values))

    def __getitem__(self, idx):
        # Sample consists of:
        #     - month label, corresponding to output month
        #     - input data, corresponding to lead+1 months of oceanic data
        #     - output, corresponding to precipitation in the midwest
        #print("idx")
        #print(idx)
        print("first idx before change")
        print(idx)
        print("self.valid_time_idx in __getitem__")
        print(self.valid_time_idx)
        
        idx = self.valid_time_idx[idx]
        print("idx")
        print(idx)
        print("idx + self.lead")
   
        print(idx + self.lead)
        print("len(self.X[idx:idx + self.lead])")
        print(len(self.X[idx:idx + self.lead]))

        X, y = self.X[idx:idx + self.lead], self.y[idx + self.lead]
        #print("hi")
        return X.reshape(self.lead, 2, -1), y

def get_resolution(da,months):
    da.name='da'
    
    count=0
    #list that will contain the amount of precipitation for a specific lon and lattidue for a specific date
    data_vars=[]
    dates=[]
    #iterate through all of the dates
    first=True
    for i in range(len(da.time.values)):
        count+=1
        if count==months:
            if i==(months-1):
                #get the average precipitation from all of the lon and lat for a season
                new_da=da.sel(time=slice(da.time.values[i-(months-1)],da.time.values[i])).sum(dim='time')/months
                dates.append(da.time.values[i])
                new_ds=new_da.to_dataset()
                date_lst=[da.time.values[i]]
                new_ds['date']=date_lst
            if i>(months-1):
                if first:
                    new_da2=da.sel(time=slice(da.time.values[i-(months-1)],da.time.values[i])).sum(dim='time')/months
                    new_ds2=new_da2.to_dataset()
                    date_lst2=[da.time.values[i]]
                    dates.append(da.time.values[i])
                    new_ds2['time']=date_lst2
                    new_ds=new_ds2
                    first=False
                if first==False:
                #append all of the datasets together
                    new_da2=da.sel(time=slice(da.time.values[i-(months-1)],da.time.values[i])).sum(dim='time')/months
                    new_ds2=new_da2.to_dataset()
                    date_lst2=[da.time.values[i]]
                    dates.append(da.time.values[i])
                    new_ds2['time']=date_lst2

                    new_ds=xr.concat([new_ds,new_ds2],dim='time')
                
            count=0   
    return new_ds

def getData(dim,
            data_fp,
            lead_time,
            start_year=1930,
            split_prop=[.6, .2, .2],
            do_normalize_X=True,
            do_normalize_y=True,
            do_remove_season_X=True,
            do_remove_season_y=True,
            detrend_X=True,
            detrend_y=False,
            detrend_fn=diff_detrend_xr,
            random_seed=0,
            ppt_file='ppt_midwest.nc'):
    '''
    Function to get the train/val/test data for a specific dimension.
    Args: dim – lat/lon dimension (2-item list)
    TODO: finish function description
    '''
    # Returns train, test, and validation datasets


    sss_fn = os.path.join(data_fp, f'sss_{dim[0]}x{dim[1]}.nc')
    sst_fn = os.path.join(data_fp, f'sst_{dim[0]}x{dim[1]}.nc')
    precip_fn = os.path.join(data_fp, f'ppt_{dim[0]}x{dim[1]}.nc')
    X1 = xr.open_dataarray(sss_fn).sel(
        time=slice(f'{start_year}-01-01', '2018-01-31'))
    X1=get_resolution(X1,12)
    #print("len(X1.time)")
    #print(len(X1.time))



    X2 = xr.open_dataarray(sst_fn).sel(time=slice(f'{start_year}-01-01', '2018-01-31'))
    X2=get_resolution(X2,12)
    #print("len(X2.time)")
    #print(len(X2.time))
    X3 = xr.open_dataarray(precip_fn).sel(
        time=slice(f'{start_year}-01-01', '2018-01-31'))
    X3=get_resolution(X3,12)
    # make sure time units are the same for different variables
    X3['time'] = X2.time

    X = xr.concat([X1.da, X2.da],dim='var')  # concatenate variables into single dataarray

    y_fn = os.path.join(data_fp, ppt_file)

    y = xr.open_dataarray(y_fn).sel(
        time=slice('{}-01-01'.format(start_year), '2018-01-31'))
    #print("y first len")
    #print(len(y))
    y=get_resolution(y,12).da
    #print("after 12 month resolution y ")
    #print(len(y.time))
    train_index=math.floor(len(X.time)*split_prop[0])
    valid_index=math.floor(len(X.time)*split_prop[1])
    test_index=math.floor(len(X.time)*split_prop[2])

    if detrend_X:
        X = detrend_fn(X)
    if detrend_y:
        y = detrend_fn(y)
    elif (detrend_fn is diff_detrend_xr) and detrend_X:
        y = y.isel(time=slice(1, None))

    N = len(X.time)  # Number of samples

    subseq_len = 37  # Months of data per subsequence
    subseq_start_idx = np.arange(0, N, subseq_len)
    np.random.seed(random_seed)
    subseq_start_idx = np.random.choice(subseq_start_idx,
                                        size=len(subseq_start_idx),
                                        replace=False)

    # Compute number of samples for train/val/test
    split_dims = len(subseq_start_idx) * np.array(split_prop)
    #print("split_dims")
    
    split_dims = [int(dim) for dim in split_dims]

    split_dims[0] = len(subseq_start_idx) - np.sum(split_dims[1:]).item()
    split_dims = np.cumsum(split_dims)
    #print(split_dims)

    # Get train, validation, and test indices (no overlap)
    train_idx=list(range(0, train_index))
    #train_idx = np.concatenate([np.arange(idx, min(N, idx + subseq_len))for idx in subseq_start_idx[:split_dims[0]]])
    #print("len(y.time)")
    #print(len(y.time))
    #print("y.time")
    #print(y.time)
    #print("len(X.time)")
    #print(len(X.time))
    #print("X.time")
    #print(X.time)
    #print(y.isel(time=train_idx))
    #print("len(train_idx)")
    #print(len(train_idx))
    #print("train_idx")
    #print(train_idx)
    val_idx= list(range(train_index,train_index+valid_index))
    test_idx=list(range(train_index+valid_index,train_index+valid_index+test_index))
    #val_idx = np.concatenate([np.arange(idx, min(N, idx + subseq_len))for idx in subseq_start_idx[split_dims[0]:split_dims[1]]])
    #test_idx = np.concatenate([np.arange(idx, min(N, idx + subseq_len))for idx in subseq_start_idx[split_dims[1]:]])
    #get the indices for the year
    #count=0
    #for a in range(len(X.time)):
        #count+=1
        #if r=="year" and count==12:
            #print("X.time[a]")
            #print(X.time[a])


    # SPLIT (and pre-process)
    train_set = ClimateDataset(data_fp,
                               X.isel(time=train_idx),
                               y.isel(time=train_idx),
                               do_normalize_X=do_normalize_X,
                               do_normalize_y=do_normalize_y,
                               do_remove_season_X=do_remove_season_X,
                               do_remove_season_y=do_remove_season_y,
                               add_noise=True,
                               noise_std=.1,
                               lead=12)
    val_set = ClimateDataset(data_fp,
                             X.isel(time=val_idx),
                             y.isel(time=val_idx),
                             X_mean=train_set.X_mean,
                             X_std=train_set.X_std,
                             X_min=train_set.X_min,
                             X_max=train_set.X_max,
                             y_mean=train_set.y_mean,
                             y_std=train_set.y_std,
                             y_min=train_set.y_min,
                             y_max=train_set.y_max,
                             do_normalize_X=do_normalize_X,
                             do_normalize_y=do_normalize_y,
                             do_remove_season_X=do_remove_season_X,
                             do_remove_season_y=do_remove_season_y,
                             lead=12)
    test_set = ClimateDataset(data_fp,
                              X.isel(time=test_idx),
                              y.isel(time=test_idx),
                              X_mean=train_set.X_mean,
                              X_std=train_set.X_std,
                              X_min=train_set.X_min,
                              X_max=train_set.X_max,
                              y_mean=train_set.y_mean,
                              y_std=train_set.y_std,
                              y_min=train_set.y_min,
                              y_max=train_set.y_max,
                              do_normalize_X=do_normalize_X,
                              do_normalize_y=do_normalize_y,
                              do_remove_season_X=do_remove_season_X,
                              do_remove_season_y=do_remove_season_y,
                              lead=12)

    return train_set, val_set, test_set