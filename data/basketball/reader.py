import logging
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from config import config

logger = logging.getLogger(config.parent_logger_name).getChild(__name__)


def process_raw(fn, data_dir, train_percent, val_percent, fn_train, fn_val,
                fn_test, seed):
    df = pd.read_pickle(fn)
    logger.info('READ | Loaded {0} | Shape = {1}'.format(fn, df.shape))

    # Split into train and scratch sets
    train, test = train_test_split(df,
                                   train_size=train_percent,
                                   random_state=seed,
                                   stratify=df[['a', 'shoot_label']])
    train.reset_index()
    test.reset_index()

    val, test = train_test_split(test,
                                 train_size=(val_percent /
                                             (1 - train_percent)),
                                 random_state=seed,
                                 stratify=test[['a', 'shoot_label']])
    val.reset_index()
    test.reset_index()

    # Save files
    # Train
    logger.info("Train dataset: Shape = {0}".format(train.shape))
    train.to_pickle(os.path.join(data_dir, fn_train))
    logger.info("Saved {0} in {1}".format(fn_train, data_dir))

    # Validation
    logger.info("Validation dataset: Shape = {0}".format(val.shape))
    val.to_pickle(os.path.join(data_dir, fn_val))
    logger.info("Saved {0} in {1}".format(fn_val, data_dir))

    # Test
    logger.info("Test dataset: Shape = {0}".format(test.shape))
    test.to_pickle(os.path.join(data_dir, fn_test))
    logger.info("Saved {0} in {1}".format(fn_test, data_dir))
