import argparse
import copy
import json
import logging
import os

import torch

import utils
from config import config
from cp_als import cp_decompose
from data.basketball.dataset import BballRawDataset
from train.basketball.multi import BasketballMulti
from visualization import plot

# Arguments Parse
parser = argparse.ArgumentParser()
parser.add_argument('--root-dir', dest='root_dir')
parser.add_argument('--data-dir', dest='data_dir')
parser.add_argument('--type', dest='type', choices=['multi', 'fixed', 'rand'])
parser.add_argument('--stop-cond',
                    dest='stop_cond',
                    choices=[
                        'val_loss_increase', 'gradient_entropy',
                        'gradient_norm', 'gradient_variance'
                    ])
parser.add_argument('--batch-size', dest='batch_size', type=int)
parser.add_argument('--sigma', dest='sigma', type=float)
parser.add_argument('--K', dest='K', type=int)
parser.add_argument('--step-size', dest='step_size', type=int)
parser.add_argument('--gamma', dest='gamma', type=float)
parser.add_argument('--full-lr', dest='full_lr', type=float)
parser.add_argument('--full-reg', dest='full_reg', type=float)
parser.add_argument('--low-lr', dest='low_lr', type=float)
parser.add_argument('--low-reg', dest='low_reg', type=float)
args = parser.parse_args()

# Parameters
params = dict()
for arg in vars(args):
    if arg not in ['project_dir', 'root_dir', 'data_dir', 'type']:
        params[arg] = getattr(args, arg)

# Save parameters
os.makedirs(args.root_dir, exist_ok=True)
with open(os.path.join(args.root_dir, f"params_{args.type}.json"),
          "w",
          encoding='utf-8') as f:
    json.dump(params, f, ensure_ascii=False, indent=4)

# Create separate params dict
hyper = copy.deepcopy(params)
for k in list(hyper):
    if k.startswith('full') or k.startswith('low'):
        hyper.pop(k)
_ = [hyper.pop(k, None) for k in ['K', 'nonnegative_weights']]

# Set directories
fig_dir = os.path.join(args.root_dir, "fig")
os.makedirs(fig_dir, exist_ok=True)
save_dir = os.path.join(args.root_dir, "saved")
os.makedirs(save_dir, exist_ok=True)

# Set logger
main_logger = logging.getLogger(config.parent_logger_name)
utils.set_logger(main_logger, os.path.join(args.root_dir, f"{args.type}.log"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Results
results = {
    'best_epochs': [],
    'best_lr': [],
    'train_times': [],
    'train_loss': [],
    'grad_norms': [],
    'grad_entropies': [],
    'grad_vars': [],
    'val_times': [],
    'val_loss': [],
    'val_conf_matrix': [],
    'val_acc': [],
    'val_precision': [],
    'val_recall': [],
    'val_F1': [],
    'test_conf_matrix': [],
    'test_acc': [],
    'test_precision': [],
    'test_recall': [],
    'test_F1': [],
    'test_out': [],
    'test_labels': []
}
if args.type == 'multi':
    results['dims'] = [[[4, 5], [6, 6]], [[8, 10], [6, 6]], [[8, 10], [12,
                                                                       12]],
                       [[8, 10], [12, 12]], [[20, 25], [12, 12]],
                       [[40, 50], [12, 12]]]  # Low rank
    results['low_start_idx'] = 3
elif args.type == 'fixed':
    results['dims'] = [[[40, 50], [12, 12]], [[40, 50], [12, 12]]]
    results['low_start_idx'] = 1
elif args.type == 'rand':
    results['dims'] = [[[40, 50], [12, 12]]]
    results['low_start_idx'] = 0

# Create datasets
train_set = BballRawDataset(os.path.join(args.data_dir, config.fn_train))
val_set = BballRawDataset(os.path.join(args.data_dir, config.fn_val))
test_set = BballRawDataset(os.path.join(args.data_dir, config.fn_test))

if args.type == 'multi' or args.type == 'fixed':
    # Full-rank first resolution
    b = results['dims'][0][0]
    c = results['dims'][0][1]
    b_str = utils.size_to_str(b)
    c_str = utils.size_to_str(c)
    train_set.calculate_pos(b, c)
    val_set.calculate_pos(b, c)

    # Train
    multi = BasketballMulti(device)
    multi.init_full_model(train_set)
    hyper['lr'] = params['full_lr']
    hyper['reg_coeff'] = params['full_reg']
    hyper['stop_threshold'] = params.get('full_stop_threshold')
    multi.init_params(**hyper)
    multi.init_loaders(train_set, val_set)
    multi.train_and_evaluate(save_dir)

    # Test
    # Create dataset
    test_set.calculate_pos(b, c)
    multi.model.load_state_dict(multi.best_model_dict)
    test_conf_matrix, test_acc, test_precision, test_recall, test_F1, test_out, test_labels = multi.test(
        test_set)

    # Metrics
    results['best_epochs'].append(multi.best_epochs)
    results['best_lr'].append(multi.best_lr)
    results['train_times'].append(multi.train_times[:multi.best_epochs])
    results['train_loss'].append(multi.train_loss[:multi.best_epochs])
    results['grad_norms'].append(multi.grad_norms[:multi.best_epochs])
    results['grad_entropies'].append(multi.grad_entropies[:multi.best_epochs])
    results['grad_vars'].append(multi.grad_vars[:multi.best_epochs])
    results['val_times'].append(multi.val_times[:multi.best_epochs])
    results['val_loss'].append(multi.val_loss[:multi.best_epochs])
    results['val_conf_matrix'].append(
        multi.val_conf_matrix[:multi.best_epochs])
    results['val_acc'].append(multi.val_acc[:multi.best_epochs])
    results['val_precision'].append(multi.val_precision[:multi.best_epochs])
    results['val_recall'].append(multi.val_recall[:multi.best_epochs])
    results['val_F1'].append(multi.val_F1[:multi.best_epochs])
    results['test_conf_matrix'].append(test_conf_matrix)
    results['test_acc'].append(test_acc)
    results['test_precision'].append(test_precision)
    results['test_recall'].append(test_recall)
    results['test_F1'].append(test_F1)
    results['test_out'].append(test_out)
    results['test_labels'].append(test_labels)

    prev_b = b
    prev_c = c

    for b, c in results['dims'][1:results['low_start_idx']]:
        b_str = utils.size_to_str(b)
        c_str = utils.size_to_str(c)

        # Calculate_pos
        train_set.calculate_pos(b, c)
        val_set.calculate_pos(b, c)

        # Finegrain
        prev_model_dict = multi.best_model_dict
        if b[0] != prev_model_dict['module.W'].size(
                1) or b[1] != prev_model_dict['module.W'].size(2):
            prev_model_dict['module.W'] = utils.finegrain(
                prev_model_dict['module.W'], b, 1)
        if c[0] != prev_model_dict['module.W'].size(
                3) or c[1] != prev_model_dict['module.W'].size(4):
            prev_model_dict['module.W'] = utils.finegrain(
                prev_model_dict['module.W'], c, 3)

        # Train
        # hyper['lr'] = multi.best_lr / ((b[0] / prev_b[0]) * (c[0] / prev_c[0]))
        hyper['lr'] = multi.best_lr
        multi = BasketballMulti(device)
        multi.init_full_model(train_set)
        multi.model.load_state_dict(prev_model_dict)
        multi.init_params(**hyper)
        multi.init_loaders(train_set, val_set)
        multi.train_and_evaluate(save_dir)

        # Test
        # Create dataset
        test_set.calculate_pos(b, c)
        multi.model.load_state_dict(multi.best_model_dict)
        test_conf_matrix, test_acc, test_precision, test_recall, test_F1, test_out, test_labels = multi.test(
            test_set)

        # Metrics
        results['best_epochs'].append(multi.best_epochs)
        results['best_lr'].append(multi.best_lr)
        results['train_times'].append(multi.train_times[:multi.best_epochs])
        results['train_loss'].append(multi.train_loss[:multi.best_epochs])
        results['grad_norms'].append(multi.grad_norms[:multi.best_epochs])
        results['grad_entropies'].append(
            multi.grad_entropies[:multi.best_epochs])
        results['grad_vars'].append(multi.grad_vars[:multi.best_epochs])
        results['val_times'].append(multi.val_times[:multi.best_epochs])
        results['val_loss'].append(multi.val_loss[:multi.best_epochs])
        results['val_conf_matrix'].append(
            multi.val_conf_matrix[:multi.best_epochs])
        results['val_acc'].append(multi.val_acc[:multi.best_epochs])
        results['val_precision'].append(
            multi.val_precision[:multi.best_epochs])
        results['val_recall'].append(multi.val_recall[:multi.best_epochs])
        results['val_F1'].append(multi.val_F1[:multi.best_epochs])
        results['test_conf_matrix'].append(test_conf_matrix)
        results['test_acc'].append(test_acc)
        results['test_precision'].append(test_precision)
        results['test_recall'].append(test_recall)
        results['test_F1'].append(test_F1)
        results['test_out'].append(test_out)
        results['test_labels'].append(test_labels)

        prev_b = b
        prev_c = c

    # Draw plots for full rank train
    fp_fig = os.path.join(fig_dir, "full_time_vs_loss.png")
    plot.loss_time(results['train_times'],
                   results['train_loss'],
                   results['val_times'],
                   results['val_loss'],
                   fp_fig=fp_fig)

    # CP_decomposition
    prev_model_dict = multi.best_model_dict
    hyper['K'] = params['K']
    W_size = prev_model_dict['module.W'].size()
    W = prev_model_dict['module.W'].view(W_size[0], W_size[1] * W_size[2],
                                         W_size[3] * W_size[4])
    weights, factors = cp_decompose(W, hyper['K'], max_iter=2000)
    factors = [f * torch.pow(weights, 1 / len(factors)) for f in factors]
    prev_model_dict.pop('module.W')
    prev_model_dict['module.A'] = factors[0].clone().detach()
    prev_model_dict['module.B'] = factors[1].clone().detach().view(
        *b, hyper['K'])
    prev_model_dict['module.C'] = factors[2].clone().detach().view(
        *c, hyper['K'])

    # Draw heatmaps after CP decomposition
    fp_fig = os.path.join(fig_dir,
                          "full_{0},{1}_B_heatmap.png".format(b_str, c_str))
    plot.latent_factor_heatmap(prev_model_dict['module.B'],
                               cmap='cmo.balance',
                               draw_court=True,
                               fp_fig=fp_fig)
    fp_fig = os.path.join(fig_dir,
                          "full_{0},{1}_C_heatmap.png".format(b_str, c_str))
    plot.latent_factor_heatmap(prev_model_dict['module.C'],
                               cmap='cmo.balance',
                               draw_court=False,
                               fp_fig=fp_fig)

# Low-rank first resolution
b = results['dims'][results['low_start_idx']][0]
c = results['dims'][results['low_start_idx']][1]
b_str = utils.size_to_str(b)
c_str = utils.size_to_str(c)
train_set.calculate_pos(b, c)
val_set.calculate_pos(b, c)

# Train
multi = BasketballMulti(device)
hyper['K'] = params['K']
hyper['lr'] = params['low_lr']
hyper['reg_coeff'] = params['low_reg']
hyper['stop_threshold'] = params.get('low_stop_threshold')
multi.init_low_model(train_set, hyper['K'])
if args.type == 'multi' or args.type == 'fixed':
    multi.model.load_state_dict(prev_model_dict)
multi.init_params(**hyper)
multi.init_loaders(train_set, val_set)
multi.train_and_evaluate(save_dir)

# Draw heatmaps
fp_fig = os.path.join(fig_dir,
                      "low_{0},{1}_B_heatmap.png".format(b_str, c_str))
plot.latent_factor_heatmap(multi.best_model_dict['module.B'],
                           cmap='cmo.balance',
                           draw_court=True,
                           fp_fig=fp_fig)
fp_fig = os.path.join(fig_dir,
                      "low_{0},{1}_C_heatmap.png".format(b_str, c_str))
plot.latent_factor_heatmap(multi.best_model_dict['module.C'],
                           cmap='cmo.balance',
                           draw_court=False,
                           fp_fig=fp_fig)

# Test
# Create dataset
test_set.calculate_pos(b, c)
multi.model.load_state_dict(multi.best_model_dict)
test_conf_matrix, test_acc, test_precision, test_recall, test_F1, test_out, test_labels = multi.test(
    test_set)

# Metrics
results['best_epochs'].append(multi.best_epochs)
results['best_lr'].append(multi.best_lr)
results['train_times'].append(multi.train_times[:multi.best_epochs])
results['train_loss'].append(multi.train_loss[:multi.best_epochs])
results['grad_norms'].append(multi.grad_norms[:multi.best_epochs])
results['grad_entropies'].append(multi.grad_entropies[:multi.best_epochs])
results['grad_vars'].append(multi.grad_vars[:multi.best_epochs])
results['val_times'].append(multi.val_times[:multi.best_epochs])
results['val_loss'].append(multi.val_loss[:multi.best_epochs])
results['val_conf_matrix'].append(multi.val_conf_matrix[:multi.best_epochs])
results['val_acc'].append(multi.val_acc[:multi.best_epochs])
results['val_precision'].append(multi.val_precision[:multi.best_epochs])
results['val_recall'].append(multi.val_recall[:multi.best_epochs])
results['val_F1'].append(multi.val_F1[:multi.best_epochs])
results['test_conf_matrix'].append(test_conf_matrix)
results['test_acc'].append(test_acc)
results['test_precision'].append(test_precision)
results['test_recall'].append(test_recall)
results['test_F1'].append(test_F1)
results['test_out'].append(test_out)
results['test_labels'].append(test_labels)

prev_b = b
prev_c = c
for b, c in results['dims'][results['low_start_idx'] + 1:]:
    b_str = utils.size_to_str(b)
    c_str = utils.size_to_str(c)

    # Calculate_pos
    train_set.calculate_pos(b, c)
    val_set.calculate_pos(b, c)

    # Finegrain
    prev_model_dict = multi.best_model_dict
    if b[0] != prev_model_dict['module.B'].size(
            0) or b[1] != prev_model_dict['module.B'].size(1):
        prev_model_dict['module.B'] = utils.finegrain(
            prev_model_dict['module.B'], b, 0)
    if c[0] != prev_model_dict['module.C'].size(
            0) or c[1] != prev_model_dict['module.C'].size(1):
        prev_model_dict['module.C'] = utils.finegrain(
            prev_model_dict['module.C'], c, 0)

    # Train
    # hyper['lr'] = multi.best_lr / ((b[0] / prev_b[0]) * (c[0] / prev_c[0]))
    hyper['lr'] = multi.best_lr
    multi = BasketballMulti(device)
    multi.init_low_model(train_set, hyper['K'])
    multi.model.load_state_dict(prev_model_dict)
    multi.init_params(**hyper)
    multi.init_loaders(train_set, val_set)
    multi.train_and_evaluate(save_dir)

    # Draw heatmaps
    fp_fig = os.path.join(fig_dir,
                          "low_{0},{1}_B_heatmap.png".format(b_str, c_str))
    plot.latent_factor_heatmap(multi.best_model_dict['module.B'],
                               cmap='cmo.balance',
                               draw_court=True,
                               fp_fig=fp_fig)
    fp_fig = os.path.join(fig_dir,
                          "low_{0},{1}_C_heatmap.png".format(b_str, c_str))
    plot.latent_factor_heatmap(multi.best_model_dict['module.C'],
                               cmap='cmo.balance',
                               draw_court=False,
                               fp_fig=fp_fig)

    # Test
    # Create dataset
    test_set.calculate_pos(b, c)
    multi.model.load_state_dict(multi.best_model_dict)
    test_conf_matrix, test_acc, test_precision, test_recall, test_F1, test_out, test_labels = multi.test(
        test_set)

    # Metrics
    results['best_epochs'].append(multi.best_epochs)
    results['best_lr'].append(multi.best_lr)
    results['train_times'].append(multi.train_times[:multi.best_epochs])
    results['train_loss'].append(multi.train_loss[:multi.best_epochs])
    results['grad_norms'].append(multi.grad_norms[:multi.best_epochs])
    results['grad_entropies'].append(multi.grad_entropies[:multi.best_epochs])
    results['grad_vars'].append(multi.grad_vars[:multi.best_epochs])
    results['val_times'].append(multi.val_times[:multi.best_epochs])
    results['val_loss'].append(multi.val_loss[:multi.best_epochs])
    results['val_conf_matrix'].append(
        multi.val_conf_matrix[:multi.best_epochs])
    results['val_acc'].append(multi.val_acc[:multi.best_epochs])
    results['val_precision'].append(multi.val_precision[:multi.best_epochs])
    results['val_recall'].append(multi.val_recall[:multi.best_epochs])
    results['val_F1'].append(multi.val_F1[:multi.best_epochs])
    results['test_conf_matrix'].append(test_conf_matrix)
    results['test_acc'].append(test_acc)
    results['test_precision'].append(test_precision)
    results['test_recall'].append(test_recall)
    results['test_F1'].append(test_F1)
    results['test_out'].append(test_out)
    results['test_labels'].append(test_labels)

    prev_b = b
    prev_c = c

if args.type == 'multi' or args.type == 'fixed':
    # Draw loss curve for low rank
    fp_fig = os.path.join(fig_dir, "low_time_vs_loss.png")
    plot.loss_time(results['train_times'][results['low_start_idx']:],
                   results['train_loss'][results['low_start_idx']:],
                   results['val_times'][results['low_start_idx']:],
                   results['val_loss'][results['low_start_idx']:],
                   fp_fig=fp_fig)

# Draw loss curve of all
fp_fig = os.path.join(fig_dir, "all_time_vs_loss.png")
plot.loss_time(results['train_times'],
               results['train_loss'],
               results['val_times'],
               results['val_loss'],
               low_index=results['low_start_idx'],
               fp_fig=fp_fig)

# Draw F1 scores of all
fp_fig = os.path.join(fig_dir, "all_time_vs_F1.png")
plot.F1_time(results['val_times'],
             results['val_F1'],
             low_index=results['low_start_idx'],
             fp_fig=fp_fig)

# Save results
torch.save(results, os.path.join(save_dir, "results.pt"))

main_logger.info('FINISH')
