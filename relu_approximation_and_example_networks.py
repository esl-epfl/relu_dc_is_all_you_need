import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import scipy

import seaborn as sns

def plot_fft(y, fs = 32.0, linewidth = None, color = None,
             label = None, true_hr = None, true_hr_color = None):
    N = y.size
    
    # sample spacing
    T = 1/fs
    x = np.linspace(0.0, N*T, N)
    yf = scipy.fftpack.fft(y)
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
    
    plt.plot(xf, 2.0/N * np.abs(yf[:N//2]), linewidth = linewidth,
             color = color, label = label)
    
    if true_hr != None:
        index = np.argwhere(xf >= true_hr).flatten()[0]
        plt.plot(xf[index], 2.0 / N * np.abs(yf[:N//2][index]), 'o',
                markersize = 7, color = true_hr_color, markerfacecolor = 'none',
                markeredgewidth = linewidth)

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

f0 = 10
f1 = f0/2
f2 = f0 * 2
f3 = f0 * 4

fs = 1024
t = np.linspace(0, 8, fs * 8)


a = np.array([1, 1, 1, 1])
y1 = a[0] * np.cos(2 * np.pi * f0 * t)
y2 = a[1] * np.cos(2 * np.pi * f1 * t)
y3 = a[2] * np.cos(2 * np.pi * f2 * t)
y4 = a[3] * np.cos(2 * np.pi * f3 * t)

y = y1 + y2 + y3 + y4
y_original = y.copy()
y = y * 0.0001

A = 0.5 * np.sum(a * 0.0001) ** 2


N = 50

g_approx = 0

m = y**2 / A - 1
# m = m / 100
ans = np.zeros((N,))
for n in range(N):
    an = ((-1)**n * np.math.factorial(2 * n)) / ((1 - 2 * n)*np.math.factorial(n)**2 * 4**n)
    g_approx += an * (m)**n
    ans[n] = an
    
plt.figure()
plt.plot(g_approx * np.sqrt(A))
plt.plot(np.sqrt(m + 1) * np.sqrt(A))
plt.plot(np.sqrt(y**2))

plt.figure()
plt.plot(ans, '-o', linewidth = 2)

y_approx = (g_approx * np.sqrt(A)/2 + y/2) / 0.0001

plt.figure(figsize = (5 * cm, 4 * cm))
plt.plot(t, y_original, label = 'Input Signal', linewidth = 1)
plt.plot(t, y_approx, label = 'Approximation', linewidth = 1)
plt.plot(t, tf.nn.relu(y_original), label = 'ReLU', linewidth = 1)
plt.xlim([t[0], t[1024]])
plt.yticks([])
plt.xticks([])

if save_figure:
    plt.savefig('./results/relu_approximation_figures/time_domain_relu_approx.svg',
                bbox_inches = 'tight')

plt.figure(figsize = (2 * cm, 2 * cm))
plt.plot(t, y_original, label = 'Input Signal')
plt.plot(t, y_approx, label = 'Approximation')
plt.plot(t, tf.nn.relu(y_original), label = 'ReLU')
plt.xlim([0.21, 0.26])
plt.ylim([-0.25, 0.8])
plt.xticks([])
plt.yticks([])

if save_figure:
    plt.savefig('./results/relu_approximation_figures/time_domain_relu_approx_detail.svg',
                bbox_inches = 'tight')

plt.figure(figsize = (5 * cm, 4 * cm))
plot_fft(y_original, fs = fs, label = 'Input Signal')
plot_fft(y_approx, fs = fs, label = 'ReLU Approximation')
plot_fft(tf.nn.relu(y_original).numpy(), fs = fs, label = 'ReLU')
plt.xlabel('Freq. (Hz)')
plt.xlim([-10, 150])

if save_figure:
    plt.savefig('./results/relu_approximation_figures/frequency_domain_relu_approx.svg',
                bbox_inches = 'tight')


kernel_size = 9

kernel = np.ones([kernel_size, 1, 1])
kernel = kernel / kernel.sum()
bias = np.zeros((1,))

weights = [kernel, bias]

mInput = tf.keras.Input(shape = (y.shape[0], 1))
m = tf.keras.layers.Activation('relu')(mInput)
m = tf.keras.layers.Conv1D(filters = 1, kernel_size = kernel_size)(m)
m = tf.keras.layers.Activation('relu')(m)
m = tf.keras.layers.Conv1D(filters = 1, kernel_size = kernel_size)(m)
m = tf.keras.layers.Activation('relu')(m)
m = tf.keras.layers.Conv1D(filters = 1, kernel_size = kernel_size)(m)
m = tf.keras.layers.Activation('relu')(m)
m = tf.keras.layers.Conv1D(filters = 1, kernel_size = kernel_size)(m)
m = tf.keras.layers.Activation('relu')(m)
m = tf.keras.layers.Conv1D(filters = 1, kernel_size = kernel_size)(m)
m = tf.keras.layers.Activation('relu')(m)

model = tf.keras.models.Model(inputs = mInput, 
                              outputs = m)

for i in range(len(model.layers)):
    if model.layers[i].__class__.__name__ == 'Conv1D':
        model.layers[i].set_weights(weights) 

with tf.device('/cpu:0'):
    y_pred_demo = model.predict(y_original[None, :, None]).flatten()
    
plt.figure()
plot_fft(y_pred_demo, fs = fs)
plot_fft(y_original, fs = fs)
plt.xlim([-10, 150])

yy1 = tf.nn.relu(y).numpy()
yy1 = tf.nn.relu(np.diff(yy1) * fs).numpy()
yy1 = tf.nn.relu(np.diff(yy1) * fs).numpy()

plt.figure(figsize = (4.4 * cm, 2 * cm))
plot_fft(yy1, fs = fs, label = 'relu(diff())', linewidth = 1.0, color = 'C1')
plot_fft(y_pred_demo, fs = fs, label = 'relu(avg())', linewidth = 1.0, color = 'C2')
plot_fft(y_original, fs = fs, label = 'Input', linewidth = 1.0,
         color = 'C0')
plt.xlim([-10, 150])
plt.xticks([])
plt.yticks([])

if save_figure:
    plt.savefig('./results/relu_approximation_figures/relu_dif_vs_relu_avg.svg',
                bbox_inches = 'tight')
