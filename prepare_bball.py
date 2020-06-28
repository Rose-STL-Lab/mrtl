import argparse
import logging
import os
import random

from config import config
from data.basketball.reader import process_raw
import utils

# Arguments Parse
parser = argparse.ArgumentParser()
parser.add_argument('--root-dir', dest='root_dir')
parser.add_argument('--data-dir', dest='data_dir')
args = parser.parse_args()

# Set logger
main_logger = logging.getLogger(config.parent_logger_name)
os.makedirs(args.root_dir, exist_ok=True)
utils.set_logger(main_logger, os.path.join(args.root_dir, "prepare.log"))

# Write seed
seed = random.SystemRandom().randint(1, 2**32 - 1)
main_logger.info(f"Seed: {seed}")

# Create train, val, test files
os.makedirs(args.data_dir, exist_ok=True)
data_filename = os.path.join(args.data_dir, 'full_data.pkl')
process_raw(data_filename, args.data_dir, config.train_percent,
            config.val_percent, config.fn_train, config.fn_val, config.fn_test,
            seed)
