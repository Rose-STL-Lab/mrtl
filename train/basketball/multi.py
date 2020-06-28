import logging
import os
import time
from math import ceil

import torch

import utils
from config import config
from train.basketball import model
from train.basketball.model import DataParallelPassthrough

logger = logging.getLogger(config.parent_logger_name).getChild(__name__)


class BasketballMulti:
    def __init__(self, device):
        # Model
        self.dims = []
        self.scale = None
        self.model = None
        self.loss_fn = None
        self.optimizer = None
        self.scheduler = None
        self.device = device

        # Hyperparameters
        self.params = None
        self.K_B = None
        self.K_C = None

        # Data
        self.train_loader = None
        self.val_loader = None
        self.train_T = None
        self.eval_T = None

        # Results
        self.train_times = []
        self.val_times = []
        self.train_loss = []
        self.val_loss = []
        self.accum_gradients = []
        self.gradients = []
        self.grad_norms = []
        self.grad_entropies = []
        self.grad_vars = []

        # Metrics
        self.decision_threshold = 0.5
        self.val_conf_matrix = []
        self.val_acc = []
        self.val_precision = []
        self.val_recall = []
        self.val_F1 = []

        # Best
        self.best_epochs = 0
        self.best_model_dict = None
        self.best_lr = 0
        self.best_val_conf_matrix = None
        self.best_val_acc = None
        self.best_F1 = -1.
        self.best_val_loss = float('inf')

    def init_full_model(self, train_set):
        counts = utils.class_counts(train_set)
        self.dims = [train_set.b_dims, train_set.c_dims]

        self.model = model.Full(train_set.a_dims, train_set.b_dims,
                                train_set.c_dims, counts)
        if torch.cuda.device_count() > 1:
            logger.info(f'Using {torch.cuda.device_count()} GPUs')
            self.model = DataParallelPassthrough(self.model)
        self.model.to(self.device)

        self.accum_gradients.append(
            torch.zeros_like(self.model.W.cpu(),
                             dtype=torch.float64).to(self.device))
        self.gradients.append(
            torch.zeros_like(self.model.W.cpu(),
                             dtype=torch.float64).to(self.device))

        self.scale = (train_set.b_dims[1] / 5.) * (train_set.c_dims[0] / 6.)

    def init_low_model(self, train_set, K):
        counts = utils.class_counts(train_set)
        self.dims = [train_set.b_dims, train_set.c_dims]

        self.model = model.Low(train_set.a_dims, train_set.b_dims,
                               train_set.c_dims, K, counts)
        if torch.cuda.device_count() > 1:
            logger.info(f'Using {torch.cuda.device_count()} GPUs')
            self.model = DataParallelPassthrough(self.model)
        self.model.to(self.device)

        self.accum_gradients.append(
            torch.zeros_like(self.model.A,
                             dtype=torch.float64).to(self.device))
        self.accum_gradients.append(
            torch.zeros_like(self.model.B,
                             dtype=torch.float64).to(self.device))
        self.accum_gradients.append(
            torch.zeros_like(self.model.C,
                             dtype=torch.float64).to(self.device))
        self.gradients.append(
            torch.zeros_like(self.model.A,
                             dtype=torch.float64).to(self.device))
        self.gradients.append(
            torch.zeros_like(self.model.B,
                             dtype=torch.float64).to(self.device))
        self.gradients.append(
            torch.zeros_like(self.model.C,
                             dtype=torch.float64).to(self.device))

        self.scale = (train_set.b_dims[1] / 5.) * (train_set.c_dims[0] / 6.)

    def init_params(self, **kwargs):
        self.params = kwargs.copy()

        assert 'lr' in self.params, "lr is a required param"

        # Optimizer and scheduler
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.params['lr'])
        if 'step_size' in self.params and 'gamma' in self.params:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.params['step_size'],
                gamma=self.params['gamma'])

        if 'sigma' in self.params:
            # Precompute kernel matrices
            self.K_B = utils.create_kernel(self.dims[0], self.params['sigma'],
                                           self.device)
            self.K_C = utils.create_kernel(self.dims[1], self.params['sigma'],
                                           self.device)

    def init_loaders(self, train_set, val_set):

        # Pos_weight
        self.train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=self.params['batch_size'],
            shuffle=True,
            num_workers=config.num_workers)

        self.val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=self.params['batch_size'],
            shuffle=False,
            num_workers=config.num_workers)

        self.train_T = ceil(
            len(self.train_loader.sampler) / (self.params['batch_size'] * 12))
        self.eval_T = ceil(
            len(self.val_loader.sampler) / (self.params['batch_size'] * 4))

        counts = utils.class_counts(train_set)
        self.loss_fn = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(counts[0] / counts[1]))

    def train_and_evaluate(self, save_dir=None):
        logger.info('TRAIN BEGIN | {0}: {1},{2},{3}'.format(
            type(self.model.module).__name__, self.model.a_dims, self.dims[0],
            self.dims[1]))
        logger.info('TRAIN | Optim: {0}, params:{1}'.format(
            type(self.optimizer).__name__, self.params))
        logger.info('TRAIN | Nonneg:{0}'.format(
            self.params.get('nonnegative_weights')))
        if self.scheduler is not None:
            logger.info(
                'TRAIN | sched: {0}, step_size: {1}, gamma: {2}'.format(
                    type(self.scheduler).__name__, self.scheduler.step_size,
                    self.params.get('gamma')))

        epochs = 0
        prev = float('inf')
        start_time = time.time()
        while epochs < config.max_epochs:
            logger.info('TRAIN | lr: {0}'.format(
                self.optimizer.param_groups[0]['lr']))
            self.train_one_epoch(start_time)
            self.evaluate_one_epoch(start_time)
            epochs += 1

            logger.info(
                '[{0:.2f}s] Epoch: {1} | Train loss={2:0.6f}, Val loss={3:0.6f}'
                .format(time.time() - start_time, epochs,
                        self.train_loss[-1][-1], self.val_loss[-1][-1]))
            logger.info(
                '[{0:.2f}s] Epoch: {1} | Val Acc=[{2:0.6f}, {3:0.6f}], Val F1={4:0.6f}'
                .format(time.time() - start_time, epochs,
                        self.val_acc[-1][-1][0], self.val_acc[-1][-1][1],
                        self.val_F1[-1][-1]))
            logger.info(
                '[{0:.2f}s] Epoch: {1} | GN={2:0.6e}, GE={3:0.6f}, GV={4:0.6e}'
                .format(time.time() - start_time, epochs,
                        self.grad_norms[-1][-1], self.grad_entropies[-1][-1],
                        self.grad_vars[-1][-1]))

            # Create model checkpoints every 10 epochs
            if save_dir is not None and epochs % 10 == 0:
                torch.save(
                    self.best_model_dict,
                    os.path.join(
                        save_dir, "model_{0}_{1},{2}_epoch{3}.pt".format(
                            type(self.model.module).__name__.lower(),
                            utils.size_to_str(self.dims[0]),
                            utils.size_to_str(self.dims[1]), epochs)))

            # Save best model (Val F1)
            if self.val_F1[-1][-1] > self.best_F1:
                logger.info('[{0:.2f}s] Max F1: {1:0.6f}'.format(
                    time.time() - start_time, self.val_F1[-1][-1]))
                self.best_val_loss = self.val_loss[-1][-1]
                self.best_epochs = epochs
                self.best_model_dict = self.model.state_dict()
                self.best_lr = self.optimizer.param_groups[0]['lr']
                self.best_val_conf_matrix = self.val_conf_matrix[-1]
                self.best_val_acc = self.val_acc[-1][-1]
                self.best_F1 = self.val_F1[-1][-1]
                if save_dir is not None:
                    torch.save(
                        self.best_model_dict,
                        os.path.join(
                            save_dir, "model_{0}_{1},{2}_best.pt".format(
                                type(self.model.module).__name__.lower(),
                                utils.size_to_str(self.dims[0]),
                                utils.size_to_str(self.dims[1]))))

            # # Save best model (Val loss)
            # if self.val_loss[-1] < self.best_val_loss:
            #     logger.info('[{0:.2f}s] Min Val Loss: {1:0.6f}'.format(time.time() - start_time, self.val_loss[-1]))
            #     self.best_val_loss = self.val_loss[-1]
            #     self.best_epochs = epochs
            #     self.best_model_dict = self.model.state_dict()
            #     self.best_lr = self.optimizer.param_groups[0]['lr']
            #     self.best_val_conf_matrix = self.val_conf_matrix[-1]
            #     self.best_val_acc = self.val_acc[-1]
            #     self.best_F1 = self.val_F1[-1]
            #     torch.save(self, os.path.join(save_dir, "multi_{0}_{1},{2}_best.pt".format(
            #         type(self.model.module).__name__.lower(),
            #         utils.size_to_str(self.dims[0]), utils.size_to_str(self.dims[1]))))

            if type(self.model.module).__name__.startswith('Full'):
                # # Number of epochs = 5
                # if epochs >= 5:
                #     logger.info('TRAIN FINISH | {0}: {1},{2} | Epochs: {3}'.format(
                #         type(self.model.module).__name__, self.dims[0], self.dims[1], epochs))
                #     break
                # if epochs >= 2:
                #     break

                if self.params['stop_cond'] == 'val_loss_increase':
                    if self.val_loss[-1][-1] > prev:
                        logger.info(
                            'TRAIN FINISH | {0}: {1},{2} | Epochs: {3} | Stop criterion: {4}'
                            .format(
                                type(self.model.module).__name__, self.dims[0],
                                self.dims[1], epochs,
                                self.params['stop_cond']))
                        break
                    else:
                        prev = self.val_loss[-1][-1]

                elif self.params['stop_cond'] == 'gradient_entropy':
                    if self.grad_entropies[-1][-1] > prev:
                        logger.info(
                            'TRAIN FINISH | {0}: {1},{2} | Epochs: {3} | Stop criterion: {4}'
                            .format(
                                type(self.model.module).__name__, self.dims[0],
                                self.dims[1], epochs,
                                self.params['stop_cond']))
                        break
                    else:
                        prev = self.grad_entropies[-1][-1]

                elif self.params['stop_cond'] == 'gradient_norm':
                    if self.grad_norms[-1][-1] > prev:
                        logger.info(
                            'TRAIN FINISH | {0}: {1},{2} | Epochs: {3} | Stop criterion: {4}'
                            .format(
                                type(self.model.module).__name__, self.dims[0],
                                self.dims[1], epochs,
                                self.params['stop_cond']))
                        break
                    else:
                        prev = self.grad_norms[-1][-1]

                elif self.params['stop_cond'] == 'gradient_variance':
                    if self.grad_vars[-1][-1] > prev:
                        logger.info(
                            'TRAIN FINISH | {0}: {1},{2} | Epochs: {3} | Stop criterion: {4}'
                            .format(
                                type(self.model.module).__name__, self.dims[0],
                                self.dims[1], epochs,
                                self.params['stop_cond']))
                        break
                    else:
                        prev = self.grad_vars[-1][-1]

            else:
                # Min epochs and val F1 decreases on average
                # if epochs >= 10 and np.mean(self.val_F1[-3:]) < np.mean(self.val_F1[-6:-3]):
                #     logger.info('TRAIN FINISH | {0}: {1},{2} | Epochs: {3}'.format(
                #         type(self.model.module).__name__, self.dims[0], self.dims[1], epochs))
                #     break

                # # Min epochs and val_loss increases on average
                # if epochs >= 10 and np.mean(self.val_loss[-3:]) > np.mean(self.val_loss[-6:-3]):
                #     logger.info('TRAIN FINISH | {0}: {1},{2} | Epochs: {3}'.format(
                #         type(self.model.module).__name__, self.dims[0], self.dims[1], epochs))
                #     break

                # if epochs >= 2:
                #     break

                # Loss convergence
                if self.params['stop_cond'] == 'val_loss_increase':
                    if self.val_loss[-1][-1] > prev:
                        logger.info(
                            'TRAIN FINISH | {0}: {1},{2} | Epochs: {3} | Stop criterion: {4}'
                            .format(
                                type(self.model.module).__name__, self.dims[0],
                                self.dims[1], epochs,
                                self.params['stop_cond']))
                        break
                    else:
                        prev = self.val_loss[-1][-1]

                elif self.params['stop_cond'] == 'gradient_entropy':
                    if self.grad_entropies[-1][-1] > prev:
                        logger.info(
                            'TRAIN FINISH | {0}: {1},{2} | Epochs: {3} | Stop criterion: {4}'
                            .format(
                                type(self.model.module).__name__, self.dims[0],
                                self.dims[1], epochs,
                                self.params['stop_cond']))
                        break
                    else:
                        prev = self.grad_entropies[-1][-1]

                elif self.params['stop_cond'] == 'gradient_norm':
                    if self.grad_norms[-1][-1] > prev:
                        logger.info(
                            'TRAIN FINISH | {0}: {1},{2} | Epochs: {3} | Stop criterion: {4}'
                            .format(
                                type(self.model.module).__name__, self.dims[0],
                                self.dims[1], epochs,
                                self.params['stop_cond']))
                        break
                    else:
                        prev = self.grad_norms[-1][-1]

                elif self.params['stop_cond'] == 'gradient_variance':
                    if self.grad_vars[-1][-1] > prev:
                        logger.info(
                            'TRAIN FINISH | {0}: {1},{2} | Epochs: {3} | Stop criterion: {4}'
                            .format(
                                type(self.model.module).__name__, self.dims[0],
                                self.dims[1], epochs,
                                self.params['stop_cond']))
                        break
                    else:
                        prev = self.grad_vars[-1][-1]

            if self.scheduler is not None:
                self.scheduler.step()

        if save_dir is not None:
            torch.save(
                self,
                os.path.join(
                    save_dir, "multi_{0}_{1},{2}.pt".format(
                        type(self.model.module).__name__.lower(),
                        utils.size_to_str(self.dims[0]),
                        utils.size_to_str(self.dims[1]))))

    def train_one_epoch(self, start_time):
        train_loss = 0.0

        self.train_times.append([])
        self.train_loss.append([])
        self.grad_norms.append([])
        self.grad_entropies.append([])
        self.grad_vars.append([])

        # Zero out gradients
        for i in range(len(self.gradients)):
            self.accum_gradients[i].zero_()
            self.gradients[i].zero_()

        self.model.train()
        for i, (a, bh_pos, def_pos, y) in enumerate(self.train_loader):
            a = a.to(self.device)
            bh_pos = bh_pos.to(self.device)
            def_pos = def_pos.to(self.device)
            y = y.to(self.device)

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(a, bh_pos, def_pos)

            # Compute loss
            loss = self.loss_fn(outputs, y.float())
            if self.params.get('reg_coeff'):
                if type(self.model.module).__name__.startswith('Full'):
                    reg = self.params['reg_coeff'] * (
                        utils.bball_spatial_regularizer(
                            self.model, self.K_B, self.K_C, self.device) +
                        utils.l2_regularizer(self.model, self.device))
                else:
                    reg = self.params['reg_coeff'] * (
                        utils.bball_spatial_regularizer(
                            self.model, self.K_B, self.K_C, self.device) +
                        utils.l2_regularizer(self.model, self.device))
                if i == 0:
                    logger.info(
                        "TRAIN | Step {0} | Loss={1:0.6f}, Reg={2:0.6f}".
                        format(i + 1, loss, reg))
                loss = loss + reg

            loss.backward()

            # Accumulate gradients
            utils.accum_grad(self.accum_gradients, self.model)

            self.optimizer.step()

            # Constrain weights
            if type(self.model.module).__name__.startswith(
                    'Low') and self.params.get('nonnegative_weights'):
                with torch.no_grad():
                    self.model.constrain()

            # Aggregate train_loss across batches
            train_loss += loss.item()

            # Log interval
            # if i % self.train_T == self.train_T - 1:
            #     curr_loss = train_loss / (i + 1)
            #     logger.debug('TRAIN | Step {0} | Loss={1:0.6f}'.format(i + 1, curr_loss))

            if i == 2 or i % self.train_T == self.train_T - 1 or i == len(
                    self.train_loader) - 1:
                curr_loss = train_loss / (i + 1)
                self.train_times[-1].append(time.time() - start_time)
                self.train_loss[-1].append(curr_loss)

                # Average gradients
                for p in range(len(self.gradients)):
                    self.gradients[p] = self.accum_gradients[p].div(i + 1)

                # Calculate gradient statistics
                grad_norm, grad_entropy, grad_var = utils.grad_stats(
                    self.gradients)
                self.grad_norms[-1].append(grad_norm)
                self.grad_entropies[-1].append(grad_entropy)
                self.grad_vars[-1].append(grad_var)

                logger.info(
                    'TRAIN | Step {0} | Loss={1:0.6f}, GN={2:0.6e}, GE={3:0.6f}, GV={4:0.6e}'
                    .format(i + 1, curr_loss, grad_norm, grad_entropy,
                            grad_var))

    def evaluate_one_epoch(self, start_time):
        val_loss = 0.0

        tn = 0
        fn = 0
        fp = 0
        tp = 0

        self.val_times.append([])
        self.val_loss.append([])
        self.val_acc.append([])
        self.val_precision.append([])
        self.val_recall.append([])
        self.val_F1.append([])

        self.model.eval()
        with torch.no_grad():
            for i, (a, bh_pos, def_pos, y) in enumerate(self.val_loader):
                a = a.to(self.device)
                bh_pos = bh_pos.to(self.device)
                def_pos = def_pos.to(self.device)
                y = y.to(self.device)

                outputs = self.model(a, bh_pos, def_pos)

                # Compute loss
                loss = self.loss_fn(outputs, y.float())
                if self.params.get('reg_coeff'):
                    if type(self.model.module).__name__.startswith('Full'):

                        # reg = self.params['reg_coeff'] * (utils.l1_regularizer(self.model, self.device))
                        # reg = self.params['reg_coeff'] * (utils.l2_regularizer(self.model, self.device))
                        # reg = self.params['reg_coeff'] * (
                        #     utils.spatial_regularizer(self.model, self.K_B, self.K_C, self.device))
                        reg = self.params['reg_coeff'] * (
                            utils.bball_spatial_regularizer(
                                self.model, self.K_B, self.K_C, self.device) +
                            utils.l2_regularizer(self.model, self.device))
                        # reg = self.params['reg_coeff'] * (
                        #             utils.l2_regularizer(self.model, self.device) + utils.l1_regularizer(self.model,
                        #                                                                                  self.device))
                    else:
                        # reg = self.params['reg_coeff'] * (utils.l1_regularizer(self.model, self.device))
                        # reg = self.params['reg_coeff'] * (utils.l2_regularizer(self.model, self.device))
                        # reg = self.params['reg_coeff'] * (
                        #     utils.spatial_regularizer(self.model, self.K_B, self.K_C, self.device))
                        # reg = self.params['reg_coeff'] * (utils.l1_regularizer(self.model, self.device) +
                        #                         utils.l2_regularizer(self.model, self.device))
                        reg = self.params['reg_coeff'] * (
                            utils.bball_spatial_regularizer(
                                self.model, self.K_B, self.K_C, self.device) +
                            utils.l2_regularizer(self.model, self.device))
                        # reg = self.reg_coeff * (utils.spatial_regularizer(self.model, self.K_B, self.K_C, self.device)
                        #                         + utils.l1_regularizer(self.model, self.device))
                    if i == 0:
                        logger.info(
                            "VAL | Step {0} | Loss={1:0.6f}, Reg={2:0.6f}".
                            format(i + 1, loss, reg))
                    loss = loss + reg

                # Aggregate train_loss across batches
                val_loss += loss.item()

                # Update confusion matrix
                preds = (outputs > self.decision_threshold).bool()
                tn += torch.sum((preds == 0) & (y == 0)).item()
                fn += torch.sum((preds == 0) & (y == 1)).item()
                fp += torch.sum((preds == 1) & (y == 0)).item()
                tp += torch.sum((preds == 1) & (y == 1)).item()

                # Log interval
                # if i % self.eval_T == self.eval_T - 1:
                #     logger.debug('VAL | Step {0} | Loss={1:0.6f}'.format(i + 1, curr_loss))

                if i == 2 or i % self.eval_T == self.eval_T - 1 or i == len(
                        self.val_loader) - 1:
                    curr_loss = val_loss / (i + 1)
                    self.val_times[-1].append(time.time() - start_time)
                    self.val_loss[-1].append(curr_loss)

                    # Class accuracy
                    self.val_acc[-1].append([tn / (tn + fp), tp / (tp + fn)])

                    F1, precision, recall = utils.calc_F1(fp, fn, tp)
                    self.val_precision[-1].append(precision)
                    self.val_recall[-1].append(recall)
                    self.val_F1[-1].append(F1)

                    logger.info(
                        'VAL | Step {0} | Loss={1:0.6f}, P={2:0.6f}, R={3:0.6f}, F1={4:0.6f}'
                        .format(i + 1, curr_loss, precision, recall, F1))

        # Conf Matrix
        self.val_conf_matrix.append([[tn, fp], [fn, tp]])

        logger.info("VAL | Loss={0:6f}, Conf. matrix={1}".format(
            self.val_loss[-1][-1], [[tn, fp], [fn, tp]]))

    def test(self, test_set):
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=self.params['batch_size'],
            shuffle=False,
            num_workers=config.num_workers)

        tn = 0
        fn = 0
        fp = 0
        tp = 0

        out = []
        labels = []

        self.model.eval()
        with torch.no_grad():
            for i, (a, bh_pos, def_pos, y) in enumerate(test_loader):
                a = a.to(self.device)
                bh_pos = bh_pos.to(self.device)
                def_pos = def_pos.to(self.device)
                y = y.to(self.device)

                outputs = self.model(a, bh_pos, def_pos)

                # Update confusion matrix
                preds = (outputs > self.decision_threshold).bool()
                tn += torch.sum((preds == 0) & (y == 0)).item()
                fn += torch.sum((preds == 0) & (y == 1)).item()
                fp += torch.sum((preds == 1) & (y == 0)).item()
                tp += torch.sum((preds == 1) & (y == 1)).item()

                out.append(outputs)
                labels.append(y)

        # Class accuracy
        test_acc = [tn / (tn + fp), tp / (tp + fn)]

        F1, precision, recall = utils.calc_F1(fp, fn, tp)

        logger.info(
            "TEST | Conf. matrix={0}, Acc=[{1:0.6f}, {2:0.6f}], P={3:0.6f}, R={4:0.6f}, F1={5:0.6f}"
            .format([[tn, fp], [fn, tp]], test_acc[0], test_acc[1], precision,
                    recall, F1))

        return [[tn, fp], [fn,
                           tp]], test_acc, precision, recall, F1, out, labels
