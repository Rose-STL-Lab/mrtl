#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set(style="ticks", font_scale=1.5)

fixed_res = pickle.load(open('../results/fixed_res.pkl', 'rb'))
val_loss = pickle.load(open('../results/val_loss_temp.pkl', 'rb'))
grad_norm = pickle.load(open('../results/grad_norm_temp.pkl', 'rb'))
grad_var = pickle.load(open('../results/grad_var_temp.pkl', 'rb'))


def plot_multi_fixed(fixed_x, fixed_y, multi_x, multi_y, finegrain_times, fp_fig=None):
    # Multi
    ax = sns.lineplot(multi_x, multi_y)

    # Fixed
    ax = sns.lineplot(fixed_x, fixed_y)
    ax.lines[1].set_linestyle('dashed')

    for i in finegrain_times:
        plt.axvline(i, color='gray', linestyle='dotted')

    ax.set(xlabel='Time[s]', ylabel='MSE')

    ax.legend(['MRTL', 'Fixed'])

    plt.tight_layout()
    fig = ax.get_figure()
    if fp_fig is not None:
        fig.savefig(fp_fig)
    plt.show()


def plot_stop_cond(val_loss, grad_norm, grad_var, full=True, fp_fig=None):
    val_loss_low_idx = np.argmax(val_loss['time'] > val_loss['begin_low_rank_time'])
    grad_norm_low_idx = np.argmax(grad_norm['time'] > grad_norm['begin_low_rank_time'])
    grad_var_low_idx = np.argmax(grad_var['time'] > grad_var['begin_low_rank_time'])

    if full:
        val_stop_cond = ['val_loss'] * val_loss_low_idx
        grad_norm_stop_cond = ['grad_norm'] * grad_norm_low_idx
        grad_var_stop_cond = ['grad_var'] * grad_var_low_idx

        df = pd.DataFrame({
            'val_time': np.concatenate((val_loss['time'][:val_loss_low_idx], grad_norm['time'][:grad_norm_low_idx],
                                        grad_var['time'][:grad_var_low_idx]), axis=0),
            'val_loss': np.concatenate((val_loss['val_loss'][:val_loss_low_idx], grad_norm['val_loss'][:grad_norm_low_idx],
                                        grad_var['val_loss'][:grad_var_low_idx]), axis=0),
            'stop_cond': np.concatenate((val_stop_cond, grad_norm_stop_cond, grad_var_stop_cond), axis=0)
        })
    else:
        val_stop_cond = ['val_loss'] * (len(val_loss['time']) - val_loss_low_idx)
        grad_norm_stop_cond = ['grad_norm'] * (len(grad_norm['time']) - grad_norm_low_idx)
        grad_var_stop_cond = ['grad_var'] * (len(grad_var['time']) - grad_var_low_idx)

        df = pd.DataFrame({
            'val_time': np.concatenate((val_loss['time'][val_loss_low_idx:] - val_loss['begin_low_rank_time'],
                                        grad_norm['time'][grad_norm_low_idx:] - grad_norm['begin_low_rank_time'],
                                        grad_var['time'][grad_var_low_idx:] - grad_var['begin_low_rank_time']),
                                       axis=0),
            'val_loss': np.concatenate((val_loss['val_loss'][val_loss_low_idx:], grad_norm['val_loss'][grad_norm_low_idx:],
                                        grad_var['val_loss'][grad_var_low_idx:]), axis=0),
            'stop_cond': np.concatenate((val_stop_cond, grad_norm_stop_cond, grad_var_stop_cond), axis=0)
        })

    ax = sns.lineplot(x='val_time', y='val_loss', data=df, hue='stop_cond', style='stop_cond',
                      hue_order=['val_loss', 'grad_norm', 'grad_var'],
                      style_order=['val_loss', 'grad_norm', 'grad_var'])
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[1:], labels=['Validation Loss', 'Gradient Norm', 'Gradient Variance'], )

    ax.set(xlabel='Time[s]', ylabel='MSE')

    plt.tight_layout()
    fig = ax.get_figure()
    if fp_fig is not None:
        fig.savefig(fp_fig)
    plt.show()


# fixed_low_start_idx = np.argmax(fixed_res['time'] > fixed_res['begin_low_rank_time'])
# multi_low_start_idx = np.argmax(val_loss['time'] > val_loss['begin_low_rank_time'])
# multi_finegrain_low_start_idx = np.argmax(val_loss['finegrain_times'] > val_loss['begin_low_rank_time'])
# # Full
# plot_multi_fixed(fixed_res['time'][:fixed_low_start_idx], fixed_res['val_loss'][:fixed_low_start_idx],
#                  val_loss['time'][:multi_low_start_idx], val_loss['val_loss'][:multi_low_start_idx],
#                  val_loss['finegrain_times'][:multi_finegrain_low_start_idx],
#                  fp_fig='multi_fixed_full_val_loss.png')

# # Low
# plot_multi_fixed(fixed_res['time'][fixed_low_start_idx:] - fixed_res['begin_low_rank_time'],
#                  fixed_res['val_loss'][fixed_low_start_idx:],
#                  val_loss['time'][multi_low_start_idx:]  - val_loss['begin_low_rank_time'],
#                  val_loss['val_loss'][multi_low_start_idx:],
#                  val_loss['finegrain_times'][multi_finegrain_low_start_idx:-1]-val_loss['begin_low_rank_time'],
#                  fp_fig='multi_fixed_low_val_loss.png')

# # Full
# plot_stop_cond(val_loss, grad_norm, grad_var, full=True, fp_fig='stop_cond_full_val_loss.png')
# # Low
# plot_stop_cond(val_loss, grad_norm, grad_var, full=False, fp_fig='stop_cond_low_val_loss.png')


# ### Process results from trials

# In[2]:


f = open('../results/val_loss_temp.pkl','rb')
val_loss = pickle.load(f)
f.close()

f = open('../results/grad_var_temp.pkl','rb')
grad_var = pickle.load(f)
f.close()

f = open('../results/grad_norm_temp.pkl','rb')
grad_norm = pickle.load(f)
f.close()

f = open('../results/fixed_res_temp.pkl','rb')
fixed_res = pickle.load(f)
f.close()

import pandas as pd
plot_data=dict()
# for trial in val_loss.keys():
#     if trial == 9:
multi_trial=2
fixed_trial=9
fixed_res[fixed_trial]['begin_low_rank_time'] = fixed_res[fixed_trial]['finegrain_times'][1]
begin_low_rank_idx_fixed = np.argmax(fixed_res[fixed_trial]['begin_low_rank_time'] < fixed_res[fixed_trial]['time'])
begin_low_rank_idx_multi = np.argmax(val_loss[multi_trial]['begin_low_rank_time'] < val_loss[multi_trial]['time'])
begin_low_rank_idx_multi_finegrain = np.argmax(val_loss[multi_trial]['begin_low_rank_time'] <                                                val_loss[multi_trial]['finegrain_times'])
# Fixed vs. MRTL
# Full
print('Full')
plot_multi_fixed(fixed_res[fixed_trial]['time'][:begin_low_rank_idx_fixed], 
                 fixed_res[fixed_trial]['val_loss'][:begin_low_rank_idx_fixed],
                 val_loss[multi_trial]['time'][:begin_low_rank_idx_multi], 
                 val_loss[multi_trial]['val_loss'][:begin_low_rank_idx_multi],
                 val_loss[multi_trial]['finegrain_times'][:begin_low_rank_idx_multi_finegrain],
                 fp_fig=None)
#                  fp_fig='multi_fixed_full_val_loss.png')

print('\n\nLow')
plot_multi_fixed(fixed_res[fixed_trial]['time'][begin_low_rank_idx_fixed:]-fixed_res[fixed_trial]['begin_low_rank_time'], 
                 fixed_res[fixed_trial]['val_loss'][begin_low_rank_idx_fixed:],
                 val_loss[multi_trial]['time'][begin_low_rank_idx_multi:]-val_loss[multi_trial]['begin_low_rank_time'], 
                 val_loss[multi_trial]['val_loss'][begin_low_rank_idx_multi:],
                 val_loss[multi_trial]['finegrain_times'][begin_low_rank_idx_multi_finegrain:],
                 fp_fig=None)
#                  fp_fig='multi_fixed_low_val_loss.png')


# In[3]:


plt.plot(val_loss[1]['time'])


# In[4]:


f = open('../results/val_loss_temp.pkl','rb')
val_loss = pickle.load(f)
f.close()

f = open('../results/grad_var_temp.pkl','rb')
grad_var = pickle.load(f)
f.close()

f = open('../results/grad_norm_temp.pkl','rb')
grad_norm = pickle.load(f)
f.close()

f = open('../results/fixed_res_temp.pkl','rb')
fixed_res = pickle.load(f)
f.close()

import pandas as pd
plot_data=dict()
# for trial in val_loss.keys():
#     if trial == 9:
multi_trial=2
fixed_trial=9
fixed_res[fixed_trial]['begin_low_rank_time'] = fixed_res[fixed_trial]['finegrain_times'][1]
begin_low_rank_idx_fixed = np.argmax(fixed_res[fixed_trial]['begin_low_rank_time'] < fixed_res[fixed_trial]['time'])
begin_low_rank_idx_multi = np.argmax(val_loss[multi_trial]['begin_low_rank_time'] < val_loss[multi_trial]['time'])
begin_low_rank_idx_multi_finegrain = np.argmax(val_loss[multi_trial]['begin_low_rank_time'] <                                                val_loss[multi_trial]['finegrain_times'])
# Fixed vs. MRTL
# Full
print('Full')
plot_multi_fixed(fixed_res[fixed_trial]['time'][:begin_low_rank_idx_fixed], 
                 fixed_res[fixed_trial]['val_loss'][:begin_low_rank_idx_fixed],
                 val_loss[multi_trial]['time'][:begin_low_rank_idx_multi], 
                 val_loss[multi_trial]['val_loss'][:begin_low_rank_idx_multi],
                 val_loss[multi_trial]['finegrain_times'][:begin_low_rank_idx_multi_finegrain],
                 fp_fig=None)
#                  fp_fig='multi_fixed_full_val_loss.png')

print('\n\nLow')
plot_multi_fixed(fixed_res[fixed_trial]['time'][begin_low_rank_idx_fixed:]-fixed_res[fixed_trial]['begin_low_rank_time'], 
                 fixed_res[fixed_trial]['val_loss'][begin_low_rank_idx_fixed:],
                 val_loss[multi_trial]['time'][begin_low_rank_idx_multi:]-val_loss[multi_trial]['begin_low_rank_time'], 
                 val_loss[multi_trial]['val_loss'][begin_low_rank_idx_multi:],
                 val_loss[multi_trial]['finegrain_times'][begin_low_rank_idx_multi_finegrain:],
                 fp_fig=None)
#                  fp_fig='multi_fixed_low_val_loss.png')


# In[7]:


fixed_res


# In[5]:


data = dict()
# Get loss for full and low rank
fixed_low_times = []
multi_low_times = []
fixed_full_times = []
multi_full_times = []

fixed_low_loss = []
multi_low_loss = []
fixed_full_loss = []
multi_full_loss = []

for trial in fixed_res.keys():
    fixed_res[trial]['begin_low_rank_time'] = fixed_res[trial]['finegrain_times'][1]
#     val_loss[trial]['begin_low_rank_time'] = val_loss[trial]['time'][2]
    begin_low_rank_idx_fixed = np.argmax(fixed_res[fixed_trial]['begin_low_rank_time'] < fixed_res[fixed_trial]['time'])
    begin_low_rank_idx_multi = np.argmax(val_loss[multi_trial]['begin_low_rank_time'] < val_loss[multi_trial]['time'])
    
    fixed_full_times.append(fixed_res[trial]['time'][:begin_low_rank_idx_fixed][-1])
#     multi_full_times.append(val_loss[trial]['time'][:begin_low_rank_idx_multi][-1])
    multi_full_times.append(val_loss[trial]['time'][4])
    
    fixed_low_times.append((fixed_res[trial]['time'][begin_low_rank_idx_fixed:]-                           fixed_res[trial]['begin_low_rank_time'])[-1])
#     multi_low_times.append((val_loss[trial]['time'][begin_low_rank_idx_multi:]-\
#                            val_loss[trial]['begin_low_rank_time'])[-1])
    multi_low_times.append(val_loss[trial]['time'][-1]-val_loss[trial]['time'][4])
    
    fixed_full_loss.append(fixed_res[trial]['val_loss'][:begin_low_rank_idx_fixed][-1])
    multi_full_loss.append(fixed_res[trial]['val_loss'][:begin_low_rank_idx_multi][-1])
    fixed_low_loss.append(fixed_res[trial]['val_loss'][begin_low_rank_idx_fixed:][-1])
    multi_low_loss.append(val_loss[trial]['val_loss'][:begin_low_rank_idx_multi:][-1])


# In[8]:


val_loss[trial]['time']


# In[6]:


print('Full')
print('Mean fixed time', np.mean(fixed_full_times), 'std fixed time', np.std(fixed_full_times))
print('Mean multi time', np.mean(multi_full_times), 'std multi time', np.std(multi_full_times))

print('Mean fixed loss', np.mean(fixed_full_loss), 'std fixed loss', np.std(fixed_full_loss))
print('Mean multi loss', np.mean(multi_full_loss), 'std multi loss', np.std(multi_full_loss))

print('\n\nLow')
print('Mean fixed times', np.mean(fixed_low_times), 'std fixed times', np.std(fixed_low_times))
print('Mean multi times', np.mean(multi_low_times), 'std multi times', np.std(multi_low_times))

print('Mean fixed loss', np.mean(fixed_low_loss), 'std fixed loss', np.std(fixed_low_loss))
print('Mean multi loss', np.mean(multi_low_loss), 'std multi loss', np.std(multi_low_loss))

