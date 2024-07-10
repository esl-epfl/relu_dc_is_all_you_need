import tensorflow as tf
import numpy as np
import scipy
import matplotlib.pyplot as plt

from sklearn.utils import shuffle

import seaborn as sns

class WeightResponseCallback(tf.keras.callbacks.Callback):
    
    def __init__(self, n_convs, fs):
        self.n_convs = n_convs
        self.fs = fs
        self.activation_weights = []
        
    def on_train_begin(self, logs = None):
        w, activation_weights = get_conv_freq_response(self.model, 
                                                       self.n_convs, 
                                                       self.fs)
        self.w = w
        self.activation_weights.append(activation_weights[-1])
    
    def on_epoch_end(self, epoch, logs = None):
        w, activation_weights = get_conv_freq_response(self.model, 
                                                       self.n_convs, 
                                                       self.fs)
        self.w = w
        self.activation_weights.append(activation_weights[-1])
        

def plot_fft(y, fs = 32.0, linewidth = None, color = None,
             label = None, true_hr = None, true_hr_color = None,
             linestyle = None):
    N = y.size
    
    # sample spacing
    T = 1/fs
    x = np.linspace(0.0, N*T, N)
    yf = scipy.fftpack.fft(y)
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
    
    plt.plot(xf, 2.0/N * np.abs(yf[:N//2]), linewidth = linewidth,
             color = color, label = label, linestyle = linestyle)
    
    if true_hr != None:
        index = np.argwhere(xf >= true_hr).flatten()[0]
        plt.plot(xf[index], 2.0 / N * np.abs(yf[:N//2][index]), 'o',
                markersize = 7, color = true_hr_color, markerfacecolor = 'none',
                markeredgewidth = linewidth)


def get_conv_freq_response(model, n_convs, fs):
    activation_weights = []

    for i in range(n_convs):
        
        cur_weights = model.layers[i + 1].get_weights()[0]
        
        cur_response = np.zeros((cur_weights.shape[-1], 512),
                                dtype = np.complex64)
        
        if i == 0:
            for channel in range(cur_weights.shape[2]):
                w, h = scipy.signal.freqz(cur_weights[:, 0, channel], fs = fs)
                cur_response[channel, :] = h
        else:
            for channel in range(cur_weights.shape[2]):
                hh = 0
                for channel_in in range(cur_weights.shape[1]):
                    w, h = scipy.signal.freqz(cur_weights[:, channel_in, channel], 
                                              fs = fs)
                    hh += h * activation_weights[i - 1][channel_in]
                cur_response[channel, :] = hh
                
        activation_weights.append(cur_response)
    return w, activation_weights

def plot_weights_freq_response(w, activation_weights):
    plt.figure()
    for i in range(activation_weights.shape[0]):
        plt.subplot(4, 8, i + 1)
        plt.plot(w, 20 * np.log10(np.abs(activation_weights[i, :])),
                 color = 'black')
        
        plt.ylim([-25, 75])
        # plt.ylixm([-25, 25])
        ymin, ymax = plt.ylim()
        plt.vlines(f0, ymin, ymax, linestyle = 'dashed', color = 'C0')
        plt.vlines(f1, ymin, ymax, linestyle = 'dashed', color = 'C1')
        plt.vlines(f2, ymin, ymax, linestyle = 'dashed', color = 'C2')

        plt.title('Channel ' + str(i))


tf.keras.utils.set_random_seed(0) 
tf.config.experimental.enable_op_determinism()


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

fs = 32.0
n_convs = 2
n_samples_per_class = 1024
kernel_size = 3
n_filters = 32
activation = 'linear'


mInput = tf.keras.Input(shape = (256, 1))
m = tf.keras.layers.Conv1D(filters = n_filters, 
                           kernel_size = kernel_size,
                           padding = 'causal',
                           activation = activation,
                           use_bias = False,)(mInput)


for i in range(n_convs - 1):
    m = tf.keras.layers.Conv1D(filters = n_filters, 
                               kernel_size = kernel_size,
                               padding = 'causal',
                               activation = activation,
                               use_bias = False,)(m)

m = tf.keras.layers.GlobalAveragePooling1D()(m)
# m = tf.keras.layers.Dense(8, activation = 'relu')(m)
m = tf.keras.layers.Dense(3)(m)

model = tf.keras.models.Model(inputs = mInput, 
                              outputs = m)

optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3)

model.compile(optimizer = optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

f0 = 3.0
f1 = 5.0
f2 = 10.0

f0s_0 = np.random.normal(loc = f0, scale = 0.1, size = n_samples_per_class)
f0s_1 = np.random.normal(loc = f1, scale = 0.1, size = n_samples_per_class)
f0s_2 = np.random.normal(loc = f2, scale = 0.1, size = n_samples_per_class)

f0s = np.concatenate([f0s_0, f0s_1, f0s_2])

y = np.concatenate([np.zeros((n_samples_per_class,)), 
                    np.ones((n_samples_per_class,)),
                    2 * np.ones((n_samples_per_class,))])

a_n = np.concatenate([np.ones((n_samples_per_class,)), 
                    5 * np.ones((n_samples_per_class,)),
                    19 * np.ones((n_samples_per_class,))])

t = np.arange(256)/fs

X = np.zeros((3 * n_samples_per_class, 256))
for i in range(X.shape[0]):
    X[i, :] = np.cos(2 * np.pi * t * f0s[i]) #+ a_n[i] * np.sqrt(2)/4
X = X[..., None]

X, y, f0s = shuffle(X, y, f0s)

mInput = tf.keras.Input(shape = (256, 1))
m1 = tf.keras.layers.Conv1D(filters = 1, 
                           kernel_size = 2,
                           padding = 'causal',
                           activation = 'relu',
                           use_bias = False,)(mInput)
m = tf.keras.layers.GlobalAveragePooling1D()(m1)
manual_model = tf.keras.models.Model(inputs = mInput, outputs = [m, m1])

with tf.device('/cpu:0'):
    y_pred_manual = manual_model.predict(X)
    
plt.figure(figsize = (4.5 * cm, 2.5 * cm))

for i in range(3):   
    plt.plot(y_pred_manual[0][y == i], f0s[y == i], 'o',
             markersize = 1.0)

plt.xticks([])
plt.ylabel('Frequency')
plt.xlabel('DC Component')

if save_figure:
    plt.savefig('./results/minimum_classifying_network/freq_vs_dc_component_classification.svg',
                bbox_inches = 'tight')

w, h = scipy.signal.freqz(manual_model.layers[1].get_weights()[0].flatten(),
                          fs = fs)

plt.figure(figsize = (4.5 * cm, 2.5 * cm))
plt.plot(w, 20 * np.log10(np.abs(h)), color = 'black')
ymin, ymax = plt.ylim()
plt.vlines(f0, ymin, ymax, linestyle = 'dashed', color = 'C0')
plt.vlines(f1, ymin, ymax, linestyle = 'dashed', color = 'C1')
plt.vlines(f2, ymin, ymax, linestyle = 'dashed', color = 'C2')
plt.xlabel('Freq. (Hz)')
plt.ylabel('Filter response (dB)')

if save_figure:
    plt.savefig('./results/minimum_classifying_network/filter_response.svg',
                bbox_inches = 'tight')
