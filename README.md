# Image Captioning with Attention

This repository provides a **Tensorflow 2.0** implementation of the image captioning model described in Google's "Show, Attend and Tell: Neural Image Caption Generation with Visual Attention" by Xu et al. (ICML2015).

This version uses:

- Python 3.7
- TensorFlow 2.0-Alpha

## Introduction

This neural system for image captioning is based on the paper "Show, Attend and Tell: Neural Image Caption Generation with Visual Attention" by Xu et al. (ICML2015). The input is an image, and the output is a sentence describing the content of the image. This system adheres to the encoder-decoder architecture with the addition of a soft attention mechanism. The encoder uses a convolutional neural network to extract visual features from the image. The decoder uses a recurrent neural network (GRU or LSTM) to generate sentences from the image features, guiding the process with a soft attention mechanism, as the one described in . A soft attention mechanism is also included to improve the quality of the generated captions.

This project is implemented using the recently released Tensorflow 2.0 library (Alpha version), and allows end-to-end training of both the CNN encoder and the RNN decoder parts.

## Prerequisites

- **Tensorflow 2.0 (alpha)** ([instructions](https://www.tensorflow.org/alpha/))
- **NumPy** ([instructions](https://scipy.org/install.html))

## References

- [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/abs/1502.03044). Kelvin Xu, Jimmy Ba, Ryan Kiros, Kyunghyun Cho, Aaron Courville, Ruslan Salakhutdinov, Richard Zemel, Yoshua Bengio. ICML 2015.