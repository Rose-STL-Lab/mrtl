import argparse
import os
from datetime import datetime

import numpy as np
import torch
import xarray as xr
from cdo import *
from matplotlib.patches import Polygon
from matplotlib.path import Path
from torch.nn.functional import interpolate

cdo = Cdo()


def convertLons(lons):
    # Change longitude from [0,360] to [-180,180]
    # Takes in array/list of longitudes and returns array/list of same length
    return ([x if x <= 180 else x - 360 for x in lons])


def getPointsForRange(minmax, n, delta=None):
    '''
    Function to get value of points for given range and number of points
    - minmax is two element list, in format [min, max] and n is integer
      representing number of points
    - Similar to np.linspace, but does not include endpoints
    '''
    if delta is None:
        delta = (minmax[1] - minmax[0]) / (n + 1)
    centers = [minmax[0] + delta / 2 + delta * i for i in np.arange(0, n)]
    return (centers)


def downscale_centers(x, n, old_delta=None):
    '''Function to downsample points to number of points 'n'.
    - x is vector of points
    - n is new number of points'''
    if old_delta is None:
        # Compute average distance between centers in old data
        old_delta = np.mean([x[i + 1] - x[i] for i, _ in enumerate(x[:-1])
                             ])  # get resolution of vector

    # compute distance between centers in new data
    # new_delta = old_delta * len(x)/n
    new_delta = (x[-1] + old_delta / 2 - x[0] - old_delta / 2) / (n + 1)
    centers = getPointsForRange(
        minmax=[x[0] - old_delta / 2, x[-1] + old_delta / 2],
        n=n,
        delta=new_delta)
    return (centers)


def padVector(x, target_range=None):
    '''
    Function extrapolates values of x to match range given by target_range.
    Extrapolates beginning and end
    - x is a vector
    - target_range is a two-element list specifying the min and max of the
      target range
    Returns a vector
    '''
    if target_range is None:  # extrapolate to nearest integer outside of range
        target_range = np.floor(x[0]), np.ceil(x[-1])

    delta_in = np.mean([x[i + 1] - x[i] for i, _ in enumerate(x[:-1])
                        ])  # get resolution of vector

    # Extrapolate the tail
    start = x[-1]
    excess_after = target_range[-1] - start
    n = np.round(excess_after / delta_in)
    tail = np.array([start + delta_in * i for i in np.arange(1, n + 1)])

    # Extrapolate the "nose" (beginning)
    end = x[0]
    excess_before = end - target_range[0]
    n = np.round(excess_before / delta_in)
    nose = np.array([end - delta_in * i for i in np.arange(n, 0, -1)])

    return (np.concatenate([nose, x, tail]), [len(nose), len(tail)])


def pad_xarray(x, target_lon=None, target_lat=None):
    '''
    Funtion pads an array with NaN values to match the given longitude
    and latitude
    '''
    target_lon, lon_idx = padVector(x.lon, target_lon)
    target_lat, lat_idx = padVector(x.lat, target_lat)

    new_grid = xr.DataArray(
        np.nan *
        np.zeros([x.shape[0], len(target_lat),
                  len(target_lon)]),
        coords={
            'time': x.time,
            'lat': target_lat,
            'lon': target_lon
        },
        dims=['time', 'lat', 'lon'])
    new_grid[:, lat_idx[0]:-lat_idx[1], lon_idx[0]:-lon_idx[1]] = x
    return (new_grid)


# Create mask from the path
def makeMask(path, lat, lon):
    '''
    Function returns a mask with with 1s representing the area inside of path
    '''
    lon = convertLons(lon)  # make sure longitudes are in [-180,180]
    lon_lat_grid = np.meshgrid(lon, lat)
    t = zip(lon_lat_grid[0].flatten(),
            lon_lat_grid[1].flatten())  # get pairs of lat/lon
    t = np.array(list(t))  # convert to array
    mask = path.contains_points(t).reshape(len(lat), len(lon))  # create mask
    return (mask)


def addOutline(ax, region_path, edgecolor='red'):
    #     Function to add outline of midwest region to map
    outline = Polygon(region_path.vertices,
                      closed=True,
                      fill=False,
                      linewidth=3,
                      edgecolor=edgecolor)
    ax.add_patch(outline)


# #### Get precip in region
def get_region_precip(data, path):
    # Function to get the total precipitation inside a given region
    # Takes in precipitation array and path representing outline of the region
    lat_range = [np.min(path.vertices[:, 1]), np.max(path.vertices[:, 1])]
    lon_range = [np.min(path.vertices[:, 0]), np.max(path.vertices[:, 0])]

    temp = data.sel(lon=slice(*lon_range), lat=slice(*lat_range))

    mask = xr.DataArray(makeMask(path, temp.lat, temp.lon),
                        coords={
                            'lat': temp.lat,
                            'lon': temp.lon
                        },
                        dims=['lat', 'lon'])
    res = (mask * temp).sum(dim=['lat', 'lon'])
    return (res)


# ### Get multiple resolutions of data
def get_multi_res(model,
                  var,
                  lonlat_range=[181, 360, -20, 59],
                  dims=[[80, 180], [60, 135], [40, 90], [24, 54], [12, 27],
                        [8, 18], [4, 9]]):
    '''
    Function to get multiple resolutions of data
    Args: - model and var are strings representing the data source and
            variable, respectively
          - dims is a list of dimensions representing dimensions to rescale to
          - lonlat_range represents the extent of the data in lat/lon
            coordinates
    '''
    input_fp = os.path.join(out_fp, f'{var}_lonlat.nc')
    data = xr.open_dataarray(input_fp)  # Open xarray data
    tensor_data = torch.from_numpy(data.values).unsqueeze(
        1)  # Get numpy values and add 4th dimension

    for dim in dims:
        lon = getPointsForRange(minmax=lonlat_range[:2], n=dim[1])
        lat = getPointsForRange(minmax=lonlat_range[2:], n=dim[0])
        new_data = interpolate(tensor_data,
                               size=dim,
                               mode='bilinear',
                               align_corners=False).squeeze(1)  # downsample
        new_data = xr.DataArray(new_data.numpy(),
                                coords={
                                    'time': data.time,
                                    'lon': lon,
                                    'lat': lat
                                },
                                dims=['time', 'lat', 'lon'])
        new_data.lon.attrs = data.lon.attrs
        new_data.lat.attrs = data.lat.attrs
        new_data.to_netcdf(os.path.join(out_fp, f'{var}_{dim[0]}x{dim[1]}.nc'))

        # Get area of each gridcell and write to file
        gridarea_fn = f'gridarea_{dim[0]}x{dim[1]}.nc'
        if not (gridarea_fn in os.listdir(out_fp)):
            infile = os.path.join(out_fp, f'{var}_{dim[0]}x{dim[1]}.nc')
            outfile = os.path.join(out_fp, gridarea_fn)
            cdo.gridarea(input=infile, output=outfile)


if __name__ == '__main__':

    # Arguments Parse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', dest='out_fp')

    args = parser.parse_args()
    out_fp = args.out_fp

    if not os.path.isdir(out_fp):
        os.makedirs(out_fp)

    # # Precip
    ppt = xr.open_dataset(os.path.join(out_fp, 'all_prism.nc')).rename({'__xarray_dataarray_variable__':'ppt'}).ppt
    ppt = ppt.sel(time=slice(None, '2018-11-30'))

    # Interpolate precipitation to 1º resolution, then embed in larger grid
    temp = interpolate(torch.tensor(ppt.values).unsqueeze(1),
                       scale_factor=(1 / 24, 1 / 24),
                       mode='bilinear',
                       align_corners=False).squeeze(1)

    # Get coordinates of downscaled array
    new_lat = downscale_centers(ppt.lat.values, n=temp.shape[-2])
    new_lon = downscale_centers(ppt.lon.values, n=temp.shape[-1])
    temp = xr.DataArray(temp.numpy(),
                        coords={
                            'time': ppt.time,
                            'lat': new_lat,
                            'lon': new_lon
                        },
                        dims=['time', 'lat', 'lon'])

    # Pad the array with zeros (to match size of salinity data)
    ppt = pad_xarray(temp, target_lon=[-179, -1], target_lat=[59, -20])
    ppt = ppt.isel(lon=slice(None, -1), lat=slice(None, -1))
    ppt = np.flip(ppt, axis=1)

    # Save to file
    ppt.to_netcdf(os.path.join(out_fp, 'ppt_lonlat.nc'))

    # ### Get precip in corn belt
    # Create region to use for precipitation shed
    south_path = Path(
        np.array([[-89.5, 36.5], [-94.5, 36.5], [-94.5, 37], [-102.5, 37],
                  [-102.5, 27], [-100, 22], [-97, 20], [-97.5, 24], [-97, 28],
                  [-95, 29], [-94, 30], [-90, 29]]))
    midwest_path = Path(
        np.array([[-104, 41], [-104, 49], [-95, 49], [-90, 48], [-84, 46],
                  [-80.5, 42], [-80.5, 40.5], [-82.5, 38.5], [-83.5, 36.5],
                  [-94.5, 36.5], [-94.5, 37], [-102, 37], [-102, 41]]))
    na_monsoon_path = Path(
        np.array([[-115, 35], [-105, 35], [-105, 30], [-104, 27.5],
                  [-103.5, 25], [-100, 21], [-96, 15], [-98, 15], [-101, 16.5],
                  [-106, 19], [-107, 22], [-110, 23], [-111, 24], [-113, 26],
                  [-115, 29], [-117, 32]]))
    gulf_path = Path(
        np.array([[-90, 17], [-96, 17], [-100, 20], [-100, 35], [-83, 35],
                  [-81, 25]]))
    pacific_path = Path(
        np.array([[-77, -10], [-77, 8], [-81, 8], [-83, 9], [-85, 11],
                  [-85, 13], [-95, 17], [-123, 60], [-170, 60], [-170, -10]]))
    atlantic_path = Path(
        np.array([[-90, 17], [-83, 11], [-81, 9], [-79, 9], [-60, 5], [-15, 5],
                  [-15, 50], [-83, 50], [-83, 35], [-81, 25]]))
    pacific_path2 = Path(
        np.array([[-85, 10], [-85, 13], [-95, 17], [-123, 48], [-135, 48],
                  [-135, 28], [-128, 13], [-115, 10]]))
    west_path = Path(
        np.array([[-115, 35], [-103, 35], [-103, 37], [-102, 37], [-102, 41],
                  [-104, 41], [-104, 49], [-127, 49], [-127, 35], [-117, 32]]))
    mask_midwest = xr.DataArray(makeMask(midwest_path, ppt.lat, ppt.lon),
                                coords={
                                    'lat': ppt.lat,
                                    'lon': ppt.lon
                                },
                                dims=['lat', 'lon'])

    # Midwest
    midwest_precip = get_region_precip(ppt, midwest_path)
    midwest_precip.to_netcdf(os.path.join(out_fp,
                                          'ppt_midwest.nc'))  # export to file

    # West
    west_precip = get_region_precip(ppt, west_path)
    west_precip.to_netcdf(os.path.join(out_fp,
                                       'ppt_west.nc'))  # export to file

    # South
    south_precip = get_region_precip(ppt, south_path)
    south_precip.to_netcdf(os.path.join(out_fp,
                                        'ppt_south.nc'))  # export to file

    # Southwest
    southwest_precip = get_region_precip(ppt, na_monsoon_path)
    southwest_precip.to_netcdf(os.path.join(
        out_fp, 'ppt_southwest.nc'))  # export to file

    # # SSS/SST
    # Use CDO to remap the data onto coarser grid
    # #### Put on lat/lon grid and trim
    all_en4 = xr.open_dataset(os.path.join(out_fp, 'all_en4.nc'))
    all_en4.salinity.squeeze('depth', drop=True).to_netcdf(os.path.join(out_fp, 'en4_so.nc'))
    all_en4.temperature.squeeze('depth', drop=True).to_netcdf(os.path.join(out_fp, 'en4_temp.nc'))
    for var in [['en4_so.nc', 'sss_lonlat.nc'], ['en4_temp.nc', 'sst_lonlat.nc']]:
        cdo.sellonlatbox(181,
                         360,
                         -20,
                         59,
                         input=os.path.join(out_fp, var[0]),
                         output=os.path.join(out_fp, var[1]))

    # Get multiple resolutions of data
    for var in ['sss', 'sst', 'ppt']:
        print(f'processing {var}')
        get_multi_res('reanalysis', var)
