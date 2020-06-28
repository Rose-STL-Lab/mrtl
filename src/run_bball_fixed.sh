#!/usr/bin/env bash

if [ "$#" -ne 2 ]; then
    echo "Missing arguments: root_dir, data_dir"
    exit 1
fi
root_dir=$1
data_dir=$2

batch_size=1024
sigma=0.05
K=20
step_size=1
gamma=0.95
stop_cond="val_loss_increase"

# Multi
multi_full_lr=5e-3
multi_full_reg=5e-2
multi_low_lr=3e-2
multi_low_reg=3e-1

# Fixed
fixed_full_lr=2e-2
fixed_full_reg=5e-2
fixed_low_lr=2e-2
fixed_low_reg=3e-1

# Rand
rand_low_lr=3e-2
rand_low_reg=3e-1

num_trials=10

for ((i = 1; i <= num_trials; i++)); do
  echo "Trial ${i}: Prepare"
  python3 "prepare_bball.py" --root-dir "${root_dir}/${i}" --data-dir "${data_dir}"
  echo "Trial ${i}: Random Initialization"
    python3 "run_bball.py" --root-dir "${root_dir}/${i}/rand" --data-dir "${data_dir}" --type "rand" --stop-cond "${stop_cond}" \
  --batch-size "${batch_size}" --sigma "${sigma}" --K "${K}" --step-size "${step_size}" --gamma "${gamma}" \
  --low-lr "${rand_low_lr}" --low-reg "${rand_low_reg}"
  echo "Trial ${i}: Multi"
  python3 "run_bball.py" --root-dir "${root_dir}/${i}/multi" --data-dir "${data_dir}" --type "multi" --stop-cond "${stop_cond}" \
  --batch-size "${batch_size}" --sigma "${sigma}" --K "${K}" --step-size "${step_size}" --gamma "${gamma}" \
  --full-lr "${multi_full_lr}" --full-reg "${multi_full_reg}" --low-lr "${multi_low_lr}" --low-reg "${multi_low_reg}"
  echo "Trial ${i}: Fixed"
    python3 "run_bball.py" --root-dir "${root_dir}/${i}/fixed" --data-dir "${data_dir}" --type "fixed" --stop-cond "${stop_cond}" \
  --batch-size "${batch_size}" --sigma "${sigma}" --K "${K}" --step-size "${step_size}" --gamma "${gamma}" \
  --full-lr "${fixed_full_lr}" --full-reg "${fixed_full_reg}" --low-lr "${fixed_low_lr}" --low-reg "${fixed_low_reg}"
done
