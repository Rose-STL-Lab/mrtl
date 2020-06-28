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
multi_full_lr=5e-3
multi_full_reg=5e-2
multi_low_lr=2e-2
multi_low_reg=2e-1

num_trials=10

for ((i = 1; i <= num_trials; i++)); do
  echo "Trial ${i}: Prepare"
  python3 "prepare_bball.py" --root-dir "${root_dir}/${i}" --data-dir "${data_dir}"
  echo "Trial ${i}: Val Loss Increase"
  python3 "run_bball.py" --root-dir "${root_dir}/${i}/val_loss_increase" --data-dir "${data_dir}" --type "multi" --stop-cond "val_loss_increase" \
  --batch-size "${batch_size}" --sigma "${sigma}" --K "${K}" --step-size "${step_size}" --gamma "${gamma}" \
  --full-lr "${multi_full_lr}" --full-reg "${multi_full_reg}" --low-lr "${multi_low_lr}" --low-reg "${multi_low_reg}"
  echo "Trial ${i}: Gradient Entropy"
    python3 "run_bball.py" --root-dir "${root_dir}/${i}/gradient_norm" --data-dir "${data_dir}" --type "multi" --stop-cond "gradient_entropy" \
  --batch-size "${batch_size}" --sigma "${sigma}" --K "${K}" --step-size "${step_size}" --gamma "${gamma}" \
  --full-lr "${multi_full_lr}" --full-reg "${multi_full_reg}" --low-lr "${multi_low_lr}" --low-reg "${multi_low_reg}"
  echo "Trial ${i}: Gradient Norm"
    python3 "run_bball.py" --root-dir "${root_dir}/${i}/gradient_norm" --data-dir "${data_dir}" --type "multi" --stop-cond "gradient_norm" \
  --batch-size "${batch_size}" --sigma "${sigma}" --K "${K}" --step-size "${step_size}" --gamma "${gamma}" \
  --full-lr "${multi_full_lr}" --full-reg "${multi_full_reg}" --low-lr "${multi_low_lr}" --low-reg "${multi_low_reg}"
  echo "Trial ${i}: Gradient Variance"
    python3 "run_bball.py" --root-dir "${root_dir}/${i}/gradient_variance" --data-dir "${data_dir}" --type "multi" --stop-cond "gradient_variance" \
  --batch-size "${batch_size}" --sigma "${sigma}" --K "${K}" --step-size "${step_size}" --gamma "${gamma}" \
  --full-lr "${multi_full_lr}" --full-reg "${multi_full_reg}" --low-lr "${multi_low_lr}" --low-reg "${multi_low_reg}"
done
