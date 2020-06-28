# Source: https://github.com/ktcarr/salinity-corn-yields/tree/master/mrtl
import time

import numpy as np
import torch
import utils

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Multi:
    def __init__(self, batch_size=10, stop_cond='val_loss', device=DEVICE):
        # Model
        self.model = None
        self.loss_fn = torch.nn.MSELoss(reduction='mean')
        self.optimizer = None
        self.scheduler = None
        self.device = device
        self.stop_cond = stop_cond
        self.counter = 0
        self.counter_thresh = 5

        # Data
        self.batch_size = batch_size
        self.train_loader = None
        self.val_loader = None

        # Hyperparams
        self.lr = 0.0001
        self.reg = None
        self.reg_coef = .01

        self.spatial_reg = False  # do spatial regularization?
        self.spatial_reg_coef = .01
        self.K = None  # kernel matrix

        # Metrics
        self.train_times = {'full': [], 'low': []}
        self.val_times = {'full': [], 'low': []}
        self.train_loss = {'full': [], 'low': []}
        self.val_loss = {'full': [], 'low': []}
        self.grad_norms = {'full': [], 'low': []}
        self.grad_entropies = {'full': [], 'low': []}
        self.grad_vars = {'full': [], 'low': []}
        self.finegrain_times = {'full': [], 'low': []}

    def check_stop_cond(self, cond):
        '''Function decides when to stop training.
            - cond is a string indicating the stopping criterion to use.'''

        # Check if training Full or Low model
        if 'low' in type(self.model).__name__:
            stage = 'low'
        else:
            stage = 'full'

        if cond == 'max_epochs':  # main loop will handle this
            return False
        elif cond == 'val_loss':
            if len(self.val_loss[stage]) >= 2:
                #                 return(self.val_loss[-1] > self.val_loss[-2])
                if self.val_loss[stage][-1] > self.val_loss[stage][-2]:
                    self.counter += 1
                    return (self.counter >= self.counter_thresh)
                else:
                    self.counter = 0
                    return (False)
            else:
                return False

        else:  # gradient statistics
            if cond == 'grad_norm':
                x = self.grad_norms[stage]
            if cond == 'grad_entropy':
                x = self.grad_entropies[stage]
            if cond == 'grad_var':
                x = self.grad_vars[stage]

            if len(x) >= 2:
                if (x[-1][-1] > x[-2][-1]):
                    self.counter += 1
                    return (self.counter >= self.counter_thresh)
                else:
                    self.counter = 0
                    return False

            else:
                return False

    def init_loaders(self, train_set, val_set):
        '''Function to initialize data loaders'''
        self.train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(
            val_set, batch_size=self.batch_size, shuffle=False)

    def getPreds(self, dataset, batch_size=10):
        # Function to get predictions and actual values for a given dataset
        loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=self.batch_size,
                                             shuffle=False)

        preds = torch.tensor([]).to(self.device)
        actual = torch.tensor([]).to(self.device)

        self.model.eval()
        with torch.no_grad():
            for X, y in loader:
                X = X.to(self.device)
                y = y.to(self.device)
                preds = torch.cat((preds, self.model(X)), dim=0)
                actual = torch.cat((actual, y), dim=0)

        return actual.detach().cpu(), preds.detach().cpu()

    def train(self,
              train_set,
              val_set,
              epochs=500,
              period=50,
              plot_weights=False):
        '''Function to train model (and perform validation)'''

        # Create arrays to track gradients
        if 'low' not in type(self.model).__name__:
            accum_gradients = [
                torch.zeros_like(self.model.w.cpu(),
                                 dtype=torch.float64).to(self.device)
            ]
            gradients = [
                torch.zeros_like(self.model.w.cpu(),
                                 dtype=torch.float64).to(self.device)
            ]
        else:
            accum_gradients = [
                torch.zeros_like(self.model.A,
                                 dtype=torch.float64).to(self.device),
                torch.zeros_like(self.model.B,
                                 dtype=torch.float64).to(self.device),
                torch.zeros_like(self.model.C,
                                 dtype=torch.float64).to(self.device)
            ]
            gradients = [
                torch.zeros_like(self.model.A,
                                 dtype=torch.float64).to(self.device),
                torch.zeros_like(self.model.B,
                                 dtype=torch.float64).to(self.device),
                torch.zeros_like(self.model.C,
                                 dtype=torch.float64).to(self.device)
            ]

        # Check if training Full or Low model
        if 'low' in type(self.model).__name__:
            stage = 'low'
        else:
            stage = 'full'

        # Check if this is the first resolution for that stage
        if len(self.train_times[stage]) == 0:
            first_time = 0
        else:
            first_time = self.train_times[stage][-1]

        start_time = time.time()
        for epoch in np.arange(epochs) + 1:
            # TRAINING
            se = 0  # squared error
            n_obs = 0  # number of samples
            curr_train_loss = 0.
            curr_val_loss = 0.
            #             self.train_times.append([])
            #             self.train_loss.append([])
            self.grad_norms[stage].append([])
            self.grad_entropies[stage].append([])
            self.grad_vars[stage].append([])

            # Zero out gradients
            for i in range(len(gradients)):
                accum_gradients[i].zero_()
                gradients[i].zero_()

            self.model.train()
            for X, y in self.train_loader:
                X = X.to(self.device)
                y = y.to(self.device)

                self.optimizer.zero_grad()

                output = self.model(X)
                loss = self.loss_fn(output.flatten(), y)

                # REGULARIZATION
                if self.reg == 'l1':
                    if 'low' not in type(self.model).__name__:
                        loss += self.reg_coef * torch.norm(
                            self.model.w, p=1)  # / self.model.w.numel()
                    else:
                        for param in self.model.parameters():
                            loss += self.reg_coef * torch.norm(param, p=1)
                elif self.reg == 'l2':
                    if 'low' not in type(self.model).__name__:
                        loss += self.reg_coef * torch.norm(
                            self.model.w, p=2)**2  # / self.model.w.numel()
                    else:
                        loss += self.reg_coef * torch.norm(
                            self.model.C, p=2)**2  # / self.model.C.numel()
                if self.spatial_reg:
                    loss += self.spatial_reg_coef * utils.climate_spatial_regularizer(
                        self.model, self.K, device=self.device)

                # UPDATE
                loss.backward()

                # Accumulate gradients
                utils.accum_grad(accum_gradients, self.model)

                self.optimizer.step()

                # ACCUMULATE STATS
                se += torch.sum((output - y)**2)  # update squared error
                n_obs += len(y)  # update number of observations
                curr_train_loss += loss.item()

                # COMPUTE GRADIENT STATISTICS
                for p in range(len(gradients)):  # Average gradients
                    gradients[p] = accum_gradients[p].div(i + 1)

                # Calculate gradient statistics
                grad_norm, grad_entropy, grad_var = utils.grad_stats(gradients)
                if 'low' not in type(self.model).__name__:
                    self.grad_norms[stage][-1].append(grad_norm /
                                                      self.model.w.shape[-1])
                    self.grad_entropies[stage][-1].append(
                        grad_entropy / self.model.w.shape[-1])
                    self.grad_vars[stage][-1].append(grad_var /
                                                     self.model.w.shape[-1])
                else:
                    self.grad_norms[stage][-1].append(grad_norm /
                                                      self.model.C.shape[0])
                    self.grad_entropies[stage][-1].append(
                        grad_entropy / self.model.C.shape[0])
                    self.grad_vars[stage][-1].append(grad_var /
                                                     self.model.C.shape[0])

            # Get Loss for epoch
            self.train_loss[stage].append((se / n_obs).item())

            # Record time
            self.train_times[stage].append(first_time + time.time() -
                                           start_time)

            # VALIDATION
            se = 0  # squared error
            n_obs = 0  # number of samples

            self.model.eval()
            with torch.no_grad():
                for X, y in self.val_loader:
                    X = X.to(self.device)
                    y = y.to(self.device)

                    output = self.model(X)
                    loss = self.loss_fn(output.flatten(), y)

                    se += torch.sum((output - y)**2)  # update squared error
                    n_obs += len(y)
                    curr_val_loss += loss.item()

            # Get loss for epoch
            self.val_loss[stage].append((se / n_obs).item())

            if epoch % period == 0:
                print('\nTrain loss after epoch {0}: {1}'.format(
                    epoch, self.train_loss[stage][-1]))
                print('Val   loss after epoch {0}: {1}'.format(
                    epoch, self.val_loss[stage][-1]))

            # Check stopping criteria
            if self.check_stop_cond(self.stop_cond):
                break

            if self.scheduler is not None:
                self.scheduler.step()

        # Track end of training
        self.finegrain_times[stage].append(self.train_times[stage][-1])
