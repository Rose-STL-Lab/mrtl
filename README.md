# Multiresolution Tensor Learning

This repository contains code for the multiresolution tensor learning model (MRTL).

Paper: Jung Yeon (John) Park, Kenneth (Theo) Carr, Stephan Zheng, Yisong Yue, Rose Yu [Multiresolution Tensor Learning for Efficient and Interpretable Spatial Analysis](https://arxiv.org/abs/2002.05578), ICML 2020

# Requirements

Make sure `miniconda` or `Anaconda` is installed (https://docs.anaconda.com/anaconda/install/). Create and activate the environment using the provided environment file.

```bash
conda env create --name $NAME -f environment.yml
conda activate $NAME
```

# Description

Description of subfolders:

1. data/: process raw data and create pytorch dataset
2. config/: global configuration parameters
3. train/: contains models
4. visualization/: plotting tools

# Basketball

## Dataset and Preprocessing
STATS SportsVU player tracking data for the NBA 2012-2013 season was used [Yue et al. 2014](https://ieeexplore.ieee.org/document/7023384). As this data is proprietary, this repo only contains preprocessing code.

See `raw.py` and `read_raw.py`.

### Read raw data
```bash
python data/basketball/read_raw.py \
    --input-dir $RAW_DATA_DIR \
    --output-dir $OUTPUT_DIR
```
The `read_raw.py` produces text files containing all the used/discarded possessions, an intermediate pickle file containing all columns `cleaned_raw.pkl`, and the final preprocessed data `full_data.pkl`.

## Training
Run `prepare_bball.py` and `run_bball.py` with arguments to run a single experiment. `prepare_bball.py` creates the training dataset and sets miscellaneous parameters for a single trial (logger, seed, etc.)
```bash
python prepare_bball.py \
    --root-dir $RUN_DIR \
    -data-dir $DATA_DIR

python run_bball.py \
    --root-dir $RUN_DIR \
    --data-dir $DATA_DIR \ 
    --type $TYPE \ 
    --stop-cond $STOP_CONDITION \ 
    --batch_size $BATCH_SIZE \ 
    --sigma $SIGMA \ 
    --K $K \ 
    --step_size 1 \ 
    --gamma 0.95 \ 
    --full_lr $FULL_LR \ 
    --full_reg $FULL_REG \ 
    --low_lr $LOW_LR \ 
    --low_reg $LOW_REG
```

Helper scripts are provided in `src/` to do 10 trials of fixed vs multi resolution and stop_condition experiments.

# Climate

## Dataset and Preprocessing
There are two datasets, one consisting of precipitation over the U.S. (PRISM) and one consisting of global sea surface salinity and sea surface temperature (EN4).  
- PRISM data was accessed at https://prism.oregonstate.edu/. Monthly precipitation (ppt) data from 1895-2019 was used. Data from 1895-1980 can be downloaded from the "Historical Past" page, and data from 1981-2018 can be downloaded from the "Recent Years" page.  
- EN4 data was accessed at https://www.metoffice.gov.uk/hadobs/en4/download-en4-2-1.html. Objective analyses from 1900-2018 were used.  

To run our code, first download all raw data into a single directory (downloaded files are in .zip format). Then, unzip the files and aggregate the data using the following command.
```bash
python data/climate/extract_data.py \
    --data_dir $DATA_DIR
```

Next, preprocess the oceanic and precipitation data into separate data files for all resolutions, using the following command. The files are saved in the netCDF4 format.

```bash
python data/climate/get_multires.py \
    --data_dir $DATA_DIR
```

## Training
Run `run_climate.py` with arguments to run a single experiment. The method argument should be one of {`mrtl`, `fixed`, `random`}. `run_climate_stop_cond.py` compares the various stopping conditions. Results are saved in `$SAVE_DIR`.

```bash
python run_climate.py \
    --data_dir $DATA_DIR \
    --save_dir $SAVE_DIR \
    --experiment_name $RUN_NAME \
    --method mrtl
    --K $K
```

```bash
python run_climate_stop_cond.py \
    --data_dir $DATA_DIR \
    --save_dir $SAVE_DIR \
    --experiment_name $RUN_NAME \
    --n_trials $TRIALS
    --K $K
```
