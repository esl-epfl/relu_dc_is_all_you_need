# DC is all you need: describing ReLU from a signal processing standpoint

Reproduction repository for the "DC is all you need: describing ReLU from a signal processing standpoint"

<img src="./figures/repo_banner.svg" width="1920">

# Abstract 

Non-linear activation functions are crucial in Convolutional Neural Networks. However, until now they have not been well described in the frequency domain . In this letter we study the spectral behavior of the ReLU, a popular activation function. We use the ReLU's Taylor expansion to derive its frequency domain behavior. We demonstrate that ReLU introduces higher frequency oscillations in the signal and a constant DC component. Furthermore, we investigate the importance of this DC component demonstrating that it helps the model extract meaningful features related to the inputs' frequency content. We accompany our theoretical derivations with experiments and real-world examples. First we numerically validate our frequency response model. Then we observe ReLU's spectral behavior on two example models and a real-world one. Finally, we experimentally investigate the role of the DC component introduced by ReLU in the CNN's representations.

# Run Experiments

The code has been tested on Python 3.10.8. For the experiments of Section V.B the PPGDalia should be downloaded and placed in ```./data/```. The python scripts are organized as follows:

|Module Name | Manuscript Sections |
|------------|---------------------|
| [relu_approximation_and_example_networks.py](relu_approximation_and_example_networks.py) | Section V.A |
| [ppg_hr_conv_activations.py](ppg_hr_conv_activations.py) | Section V.B|
| [relu_vs_linear_vs_lineardc_train.py](relu_vs_linear_vs_lineardc_train.py) | Section V.C|
| [plot_relu_vs_linear_vs_lineardc_results.py](plot_relu_vs_linear_vs_lineardc_results.py) | Section V.C|
| [activation_demo_exploration_minimal_demo.py](activation_demo_exploration_minimal_demo.py) | Section V.D|

# Reference
```
TODO
```