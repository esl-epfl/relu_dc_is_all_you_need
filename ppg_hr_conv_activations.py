import numpy as np
from config import Config


# aliases
val_mae = 'val_mean_absolute_error'
mae = 'mean_absolute_error'

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.utils import shuffle
from sklearn.model_selection import LeaveOneGroupOut

from preprocessing import preprocessing_Dalia_aligned_preproc as pp

import tensorflow_probability as tfp
tfd = tfp.distributions
from self_attention_ppg_only_models import build_attention_model

import pandas as pd

import time

import pickle

import matplotlib.pyplot as plt
import scipy
from scipy import fftpack
from tqdm import tqdm

import seaborn as sns

sns.set_theme()

cm = 1 / 2.54

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


def plot_fft(y, fs = 32.0, linewidth = None, color = None,
             label = None, true_hr = None, true_hr_color = None,
             linestyle = None):
    N = y.size
    
    # sample spacing
    T = 1/fs
    x = np.linspace(0.0, N*T, N)
    yf = scipy.fftpack.fft(y)
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2) * 60
    
    plt.plot(xf, 2.0/N * np.abs(yf[:N//2]), linewidth = linewidth,
             color = color, label = label, linestyle = linestyle)
    
    if true_hr != None:
        index = np.argwhere(xf >= true_hr).flatten()[0]
        plt.plot(xf[index], 2.0 / N * np.abs(yf[:N//2][index]), 'o',
                markersize = 5, color = true_hr_color, markerfacecolor = 'none',
                markeredgewidth = linewidth)

fs = 32.0

n_epochs = 200
batch_size = 256
n_ch = 1

test_subject_id = 5

# Setup config
cf = Config(search_type = 'NAS', root = './data/')

# Load data
X, y, groups, activity = pp.preprocessing(cf.dataset, cf)


X_train = X[groups != test_subject_id]
y_train = y[groups != test_subject_id]

X_test = X[groups == test_subject_id]
y_test = y[groups == test_subject_id]

X_validate = X_test
y_validate = y_test
activity_validate = activity[groups == test_subject_id]

# Build Model
model = build_attention_model((cf.input_shape, n_ch))

model.load_weights('./saved_models/adaptive_w_attention/model_weights/model_S' + str(int(test_subject_id)) + '.h5')

X_validate = X_validate[:, :1, :]

X_validate = np.transpose(X_validate, axes = (0, 2, 1))

mInput = tf.keras.Input(shape = (cf.input_shape, n_ch))

m = mInput
conv_outputs = []
relu_outputs = []
for i in range(3):
    conv = tf.keras.layers.Conv1D(input_shape = (cf.input_shape, n_ch),
                               filters = 32,
                               kernel_size = 5,
                               dilation_rate = 2,
                                padding = 'causal',
                               activation = 'linear')
    m = conv(m)
    conv_outputs.append(m)
    m = tf.keras.layers.Activation('relu')(m)
    relu_outputs.append(m)

submodel = tf.keras.models.Model(inputs = mInput, 
                                 outputs = [conv_outputs, relu_outputs])

submodel.layers[1].set_weights(model.layers[1].layers[1].get_weights())
submodel.layers[3].set_weights(model.layers[1].layers[2].get_weights())
submodel.layers[5].set_weights(model.layers[1].layers[3].get_weights())

with tf.device('/cpu:0'):
    y_pred = submodel.predict(X_validate)

conv_activations = y_pred[0]
relu_activations = y_pred[1]

sample_index = 200
channel_index = 20

plt.figure()
plt.plot(conv_activations[2][sample_index, :, channel_index])
plt.plot(relu_activations[2][sample_index, :, channel_index])
plt.plot(X_validate[sample_index, :, 0] / 50)

plt.figure(figsize = (8.4 * cm, 4 * cm))
for channel_index in range(16):
    plt.subplot(4, 4, channel_index + 1)
    
    y_input_demo = X_validate[sample_index, :, 0]
    y_input_demo = y_input_demo / y_input_demo.std()
    y_demo_conv = conv_activations[2][sample_index, :, channel_index]
    y_demo_conv = y_demo_conv / np.abs(y_demo_conv).std()
    y_demo_relu = relu_activations[2][sample_index, :, channel_index]
    y_demo_relu = y_demo_relu / np.abs(y_demo_relu).std()
    
    # plot_fft(y_demo_conv)
    plot_fft(y_input_demo, true_hr = y_test[sample_index],
             linestyle = 'dashed', linewidth = 1.0)
    plot_fft(y_demo_relu, linewidth = 1.0)

    plt.xlim([-10, 600])
    plt.xticks([])
    plt.yticks([])
    plt.ylim([-0.1, 1.3])    

if save_figure:
    plt.savefig('./results/ppg_conv_figures/block1_conv3_activations.svg',
                bbox_inches = 'tight')    


mInput = tf.keras.Input(shape = (cf.input_shape, n_ch))

m = model.layers[1](mInput)
conv_outputs = []
relu_outputs = []
for i in range(3):
    conv = tf.keras.layers.Conv1D(input_shape = (cf.input_shape//4, 32),
                               filters = 48,
                               kernel_size = 5,
                               dilation_rate = 2,
                                padding = 'causal',
                               activation = 'linear')
    m = conv(m)
    conv_outputs.append(m)
    m = tf.keras.layers.Activation('relu')(m)
    relu_outputs.append(m)

submodel2 = tf.keras.models.Model(inputs = mInput, 
                                 outputs = [conv_outputs, relu_outputs])

submodel2.layers[2].set_weights(model.layers[2].layers[1].get_weights())
submodel2.layers[4].set_weights(model.layers[2].layers[2].get_weights())
submodel2.layers[6].set_weights(model.layers[2].layers[3].get_weights())

with tf.device('/cpu:0'):
    y_pred2 = submodel2.predict(X_validate)
    
conv_activations2 = y_pred2[0]
relu_activations2 = y_pred2[1]

t = np.linspace(0, 8, 256)
t1 = np.linspace(0, 8, 64)

channel_index = 22

plt.figure()
plt.plot(t1, conv_activations2[2][sample_index, :, channel_index])
plt.plot(t1, relu_activations2[2][sample_index, :, channel_index])
plt.plot(t, X_validate[sample_index, :, 0] / 50)

plt.figure(figsize = (8.4 * cm, 4 * cm))

for channel_index in range(16):
    plt.subplot(4, 4, channel_index + 1)
    
    y_input_demo = X_validate[sample_index, :, 0]
    std = y_input_demo.std()
    if std > 0 :
        y_input_demo = y_input_demo / std
    y_demo_conv = conv_activations2[2][sample_index, :, channel_index]
    std = y_demo_conv.std()
    if std > 0:
        y_demo_conv = y_demo_conv / std
    
    y_demo_relu = relu_activations2[2][sample_index, :, channel_index]
    std = y_demo_relu.std()
    if std > 0:
        y_demo_relu = y_demo_relu / std
    
    plot_fft(y_input_demo, true_hr = y_test[sample_index],
             linestyle = 'dashed', linewidth = 1.0)
    plot_fft(y_demo_relu, fs = fs / 4.0, linewidth = 1.0)

    plt.xlim([-10, 600])
    plt.xticks([])
    plt.yticks([])
    plt.ylim([-0.1, 1.3])

if save_figure:
    plt.savefig('./results/ppg_conv_figures/block2_conv3_activations.svg',
                bbox_inches = 'tight')
    
    