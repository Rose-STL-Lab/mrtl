import argparse
import glob
import os
from datetime import datetime
from zipfile import ZipFile

import xarray as xr
from cdo import *

cdo = Cdo()
cdo.debug = False


# TEMP_PATH = os.path.join(DATA_PATH, 'temp.nc') # temporary file path
def get_en4_single_year(f, path):
    '''Function downloads single year of EN4 data, extracts surface level,
    and saves as NetCDF file
    args: local_path is the directory to save the data to
          f is the http address of the file
    returns: none (saves file to a directory)'''

    temp_path = os.path.join(path, 'temp.nc')
    year = f[-8:-4]  # get the year for the data
    print(f'Processing EN4 year {year}')

    file_start = f'{path}/EN.4.2.1.f.analysis.g10.{year}*'  # start of filename
    out_path = os.path.join(path, f'en4_{year}.nc')  # file to save to

    # Unzip and get top level of the data
    with ZipFile(f, 'r') as zip_ref:
        zip_ref.extractall(path)  # unzip the downloaded file
    cdo.mergetime(input=file_start, output=temp_path)  # merge NetCDF files
    cdo.sellevel(5.0215898,
                 input=f'-selname,salinity,temperature {temp_path}',
                 output=out_path)  # get top level

    # Cleanup (remove downloaded and intermediate files)
    #     os.remove(f_local)
    os.remove(temp_path)
    for f in glob.glob(file_start + '*'):
        os.remove(f)
    return


def cleanup(path):
    '''Remove extraneous files after processing unzipped file'''
    types = ('*.stx', '*.csv', '*.info.txt', '*.prj', '*.xml', '*.hdr', '*.bil'
             )  # file types to remove
    patterns = [path + t for t in types]
    for pattern in patterns:  # check all patterns
        for f in glob.glob(pattern):  # remove all files matching patters
            os.remove(f)
    return


def get_prism_from_zip(f, path):
    '''Function extracts data from .zip files and saves to .nc format
    (with monthly files)'''
    year = f[len(path) + 23:len(path) + 27]  # for ppt

    with ZipFile(f, 'r') as zip_ref:
        zip_ref.extractall(path)  # unzip the downloaded file

    if 'all' in f:
        for m in [
                '01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
                '11', '12'
        ]:
            get_prism_single_month(year, m, path)
    else:
        m = f[len(path) + 27:len(path) + 29]  # for ppt
        get_prism_single_month(year, m, path)

    cleanup(path)  # remove unnecessary files
    return


def get_prism_single_month(year, month, path):
    '''function gets NetCDF file corresponding to single month of prism data
    Args: year is an integer in [1895, 2018]
          month is an integer in [1,12]'''
    suffix = 2 if (int(year) <= 1980) else 3
    in_path = os.path.join(
        path, f'PRISM_ppt_stable_4kmM{suffix}_{year}{month}_bil.bil')
    d = xr.open_rasterio(in_path)
    d = d.rename({
        'x': 'lon',
        'y': 'lat',
        'band': 'time'
    }).assign_coords(time=[datetime(int(year), int(month), 1)])
    d.to_netcdf(os.path.join(path, f'prism_{year}{month}.nc'))
    return


if __name__ == '__main__':

    # Arguments Parse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', dest='data_dir')
    args = parser.parse_args()
    path = args.data_dir

    #     #################### EN4 DATA ####################
    print('Processing EN4 (ocean) data')
    # Unzip each file, and extract surface layer
    en4_files = sorted(glob.glob(os.path.join(path, 'EN.4*')))
    for file in en4_files:
        get_en4_single_year(file, path)

    # Merge all years together to form single NetCDF file
    cdo.mergetime(input=os.path.join(path, 'en4*.nc'),
                  output=os.path.join(path, 'all_en4.nc'))

    # Remove temporary files
    for f in glob.glob(os.path.join(path, 'en4*.nc')):
        os.remove(f)

    # PRISM DATA
    print('\nProcessing PRISM (precip) data')
    cleanup(path)
    prism_files = sorted(glob.glob(os.path.join(path, 'PRISM*')))
    for f in prism_files:

        year = int(f[52:56])  # for ppt
        if year <= 1980:
            print(f'Processing PRISM year {year}')
            f_local = os.path.join(
                path, f'PRISM_ppt_stable_4kmM2_{year}_all_bil.zip')
        else:
            month = f[56:58]  # for ppt
            if month == '01':
                print(f'Processing PRISM year {year}')
            f_local = os.path.join(
                path, f'PRISM_ppt_stable_4kmM3_{year}{month}_bil.zip')

        get_prism_from_zip(f_local, path)
        # merge files
        cdo.mergetime(input=os.path.join(path, 'prism*.nc'),
                      output=os.path.join(path, 'all_prism.nc'))

        # Cleaning up (remove old files)
        for file in glob.glob(os.path.join(path, 'prism*.nc')):
            os.remove(file)

        # rename consolidated prism file
        os.rename(os.path.join(path, 'all_prism.nc'),
                  os.path.join(path, 'prism.nc'))  # rename full data

    os.rename(os.path.join(path, 'prism.nc'),
              os.path.join(path, 'all_prism.nc'))  # rename full data
