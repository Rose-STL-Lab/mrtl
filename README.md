# Multiresolution Tensor Learning

This repository contains code for the multiresolution tensor learning model (MRTL).

Paper: [Multiresolution Tensor Learning for Efficient and Interpretable Spatial Analysis](https://arxiv.org/abs/2002.05578)

# Requirements

For visualizing climate data, [PROJ](https://proj.org/) and [GEOS](https://trac.osgeo.org/geos/) need to be installed for Cartopy.

For other python packages, install using `requirements.txt`. For Cartopy, numpy needs to be installed previously.

```bash
pip install -r requirements.txt
```

# Description

Description of subfolders:

1. data/: process raw data and create pytorch dataset
2. config/: global configuration parameters
3. train/: contains models
4. visualization/: plotting tools

# Dataset and Preprocessing

## Basketball
STATS SportsVU player tracking data for the NBA 2012-2013 season was used [Yue et al. 2014](https://ieeexplore.ieee.org/document/7023384). As this data is proprietary, this repo only contains preprocessing code.

See `raw.py` and `read_raw.py`.

### Read raw data
```bash
python data/basketball/read_raw.py \
    --input-dir $RAW_DATA_DIR \
    --output-dir $OUTPUT_DIR
```
The `read_raw.py` produces text files containing all the used/discarded possessions, an intermediate pickle file containing all columns `cleaned_raw.pkl`, and the final preprocessed data `full_data.pkl`.

## Climate

Run `get_multires.py` to preprocess the oceanic and precipitation data into separate data files for all resolutions. The files are saved in the netCDF4 format. Trailing slashes are required.

```bash
python data/climate/get_multires.py \
    --ocean_data_fp dataset/climate/reanalysis/ \
    --precip_fp dataset/climate/prism/ \
    --data_dir dataset/climate/
```


# Training
## Basketball
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
## Climate
Run `run_climate.py` with arguments to run a single experiment. The method argument should be one of {mrtl, fixed}. `run_climate_stop_cond.py` compares the various stopping conditions.

```bash
python run_climate.py \
    --data_dir dataset/climate/ \
    --save_dir $SAVE_DIR \
    --experiment_name $RUN_NAME \
    --method mrtl
```

```bash
python run_climate_stop_cond.py \
    --data_dir dataset/climate/ \
    --save_dir $SAVE_DIR \
    --experiment_name $RUN_NAME \
    --n_trials $TRIALS
```
