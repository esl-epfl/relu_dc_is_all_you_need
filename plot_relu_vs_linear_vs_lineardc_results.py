import tensorflow as tf
import numpy as np
import scipy
import matplotlib.pyplot as plt

from sklearn.utils import shuffle

import pickle

import seaborn as sns

sns.set_theme()

cm = 1/2.54
save_figure = True
fontsize = 6

plt.rc('font', size = fontsize)          # controls default text sizes
plt.rc('axes', titlesize = fontsize)     # fontsize of the axes title
plt.rc('axes', labelsize = fontsize)    # fontsize of the x and y labels
plt.rc('xtick', labelsize = fontsize)    # fontsize of the tick labels
plt.rc('ytick', labelsize = fontsize)    # fontsize of the tick labels
plt.rc('legend', fontsize = fontsize)    # legend fontsize
plt.rc('figure', titlesize = fontsize)  # fontsize of the figure title

plt.rcParams.update({"font.family" : "Times New Roman"})

fig_height = 3
fig_width = 3
        
def list_of_lists_to_list_of_arrays(l):
    new_l = []
    for item in l:
        new_l.append(np.array(item))
    
    return new_l

with open('./results/results.pickle', 'rb') as handle:
    results = pickle.load(handle)


all_history_linear_noDC = results['history_linear_nodc']
all_history_linear_noDC = list_of_lists_to_list_of_arrays(all_history_linear_noDC)
all_history_linear_noDC = np.stack(all_history_linear_noDC)
m_loss_linear_nodc = all_history_linear_noDC.mean(axis = 0)
std_loss_linear_nodc = all_history_linear_noDC.std(axis = 0)

all_history_linear = results['history_linear']
all_history_linear = list_of_lists_to_list_of_arrays(all_history_linear)
all_history_linear = np.stack(all_history_linear)
m_loss_linear = all_history_linear.mean(axis = 0)
std_loss_linear = all_history_linear.std(axis = 0)

all_history_relu = results['history_relu']
all_history_relu = list_of_lists_to_list_of_arrays(all_history_relu)
all_history_relu = np.stack(all_history_relu)
m_loss_relu = all_history_relu.mean(axis = 0)
std_loss_relu = all_history_relu.std(axis = 0)

callbacks_linear_noDC_layer1 = results['callbacks_linear_nodc_layer1']
callbacks_linear_noDC_layer1 = list_of_lists_to_list_of_arrays(callbacks_linear_noDC_layer1)
callbacks_linear_noDC_layer1 = np.stack(callbacks_linear_noDC_layer1)
m_dist_linear_nodc_layer1 = callbacks_linear_noDC_layer1.mean(axis = 0)
std_dist_linear_nodc_layer1 = callbacks_linear_noDC_layer1.std(axis = 0)

callbacks_linear_layer1 = results['callbacks_linear_layer1']
callbacks_linear_layer1 = list_of_lists_to_list_of_arrays(callbacks_linear_layer1)
callbacks_linear_layer1 = np.stack(callbacks_linear_layer1)
m_dist_linear_layer1 = callbacks_linear_layer1.mean(axis = 0)
std_dist_linear_layer1 = callbacks_linear_layer1.std(axis = 0)

callbacks_relu_layer1 = results['callbacks_relu_layer1']
callbacks_relu_layer1 = list_of_lists_to_list_of_arrays(callbacks_relu_layer1)
callbacks_relu_layer1 = np.stack(callbacks_relu_layer1)
m_dist_relu_layer1 = callbacks_relu_layer1.mean(axis = 0)
std_dist_relu_layer1 = callbacks_relu_layer1.std(axis = 0)

callbacks_linear_noDC_layer2 = results['callbacks_linear_nodc_layer2']
callbacks_linear_noDC_layer2 = list_of_lists_to_list_of_arrays(callbacks_linear_noDC_layer2)
callbacks_linear_noDC_layer2 = np.stack(callbacks_linear_noDC_layer2)
m_dist_linear_nodc_layer2 = callbacks_linear_noDC_layer2.mean(axis = 0)
std_dist_linear_nodc_layer2 = callbacks_linear_noDC_layer2.std(axis = 0)

callbacks_linear_layer2 = results['callbacks_linear_layer2']
callbacks_linear_layer2 = list_of_lists_to_list_of_arrays(callbacks_linear_layer2)
callbacks_linear_layer2 = np.stack(callbacks_linear_layer2)
m_dist_linear_layer2 = callbacks_linear_layer2.mean(axis = 0)
std_dist_linear_layer2 = callbacks_linear_layer2.std(axis = 0)

callbacks_relu_layer2 = results['callbacks_relu_layer2']
callbacks_relu_layer2 = list_of_lists_to_list_of_arrays(callbacks_relu_layer2)
callbacks_relu_layer2 = np.stack(callbacks_relu_layer2)
m_dist_relu_layer2 = callbacks_relu_layer2.mean(axis = 0)
std_dist_relu_layer2 = callbacks_relu_layer2.std(axis = 0)

t = np.arange(60)

plt.figure(figsize = (fig_width * cm, fig_height * cm))
plt.plot(t, m_loss_relu, label = 'ReLU')
plt.fill_between(t, m_loss_relu - std_loss_relu, 
                 m_loss_relu + std_loss_relu,
                 alpha = 0.25)

plt.plot(t, m_loss_linear_nodc, label = 'Linear')
plt.fill_between(t, m_loss_linear_nodc - std_loss_linear_nodc, 
                 m_loss_linear_nodc + std_loss_linear_nodc,
                 alpha = 0.25)

plt.plot(t, m_loss_linear, label = 'Linear w\ DC')
plt.fill_between(t, m_loss_linear - std_loss_linear, 
                 m_loss_linear + std_loss_linear,
                 alpha = 0.25)
# plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')


if save_figure:
    plt.savefig('./results/linear_vs_relu/loss.svg',
                bbox_inches = 'tight')

plt.figure(figsize = (fig_width * cm, fig_height * cm))
plt.plot(t, m_dist_relu_layer1, label = 'ReLU')
plt.fill_between(t, m_dist_relu_layer1 - std_dist_relu_layer1, 
                 m_dist_relu_layer1 + std_dist_relu_layer1,
                 alpha = 0.25)

plt.plot(t, m_dist_linear_nodc_layer1, label = 'Linear')
plt.fill_between(t, m_dist_linear_nodc_layer1 - std_dist_linear_nodc_layer1, 
                 m_dist_linear_nodc_layer1 + std_dist_linear_nodc_layer1,
                 alpha = 0.25)

plt.plot(t, m_dist_linear_layer1, label = 'Linear w\ DC')
plt.fill_between(t, m_dist_linear_layer1 - std_dist_linear_layer1, 
                 m_dist_linear_layer1 + std_dist_linear_layer1,
                 alpha = 0.25)
plt.xlabel('Epochs')
plt.ylabel('Distance')

if save_figure:
    plt.savefig('./results/linear_vs_relu/layer1_distance.svg',
                bbox_inches = 'tight')


plt.figure(figsize = (fig_width * cm, fig_height * cm))
plt.plot(t, m_dist_relu_layer2, label = 'ReLU')
plt.fill_between(t, m_dist_relu_layer2 - std_dist_relu_layer2, 
                 m_dist_relu_layer2 + std_dist_relu_layer2,
                 alpha = 0.25)

plt.plot(t, m_dist_linear_nodc_layer2, label = 'Linear')
plt.fill_between(t, m_dist_linear_nodc_layer2 - std_dist_linear_nodc_layer2, 
                 m_dist_linear_nodc_layer2 + std_dist_linear_nodc_layer2,
                 alpha = 0.25)

plt.plot(t, m_dist_linear_layer2, label = 'Linear w\ DC')
plt.fill_between(t, m_dist_linear_layer2 - std_dist_linear_layer2, 
                 m_dist_linear_layer2 + std_dist_linear_layer2,
                 alpha = 0.25)
plt.xlabel('Epochs')
plt.ylabel('Distance')

if save_figure:
    plt.savefig('./results/linear_vs_relu/layer2_distance.svg',
                bbox_inches = 'tight')
