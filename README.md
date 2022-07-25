# Jax Basic Neural Network

This is a continuation of Samson Zhang's work in the notebook, [simple-mnist-nn-from-scratch](https://www.kaggle.com/code/wwsalmon/simple-mnist-nn-from-scratch-numpy-no-tf-keras/notebook). The main aim is explore Jax's features by using them in a simple neural network. Jax is designed to be very flexible so you can choose your prefered level of abstraction when creating a model. Jax also has GPU and TPU optimization, with no code changes required, making the models in Jax easy to scale up.

## Introduction
The notebooks start by explicitly defining the model, then Jax's features are added to explore how to use them on the model. It's recommended to go through the notebooks sequentially starting from notebook 1, 1_numpy_basic_nn through, through to, 8_jax_basic_nn_haiku_optimized. This will provide you with a fundamental understanding of a neural network while slowly introducing you to the most fundamental concepts in Jax.

## Supplementary Meterial
These notebook's may be too advanced for some readers. If you feel out of your depth about the neural network concepts, [this youtube playlist](https://youtube.com/playlist?list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0) will help get you up to speed. Samson also created a video which will help: [Building a neural network FROM SCRATCH](https://youtu.be/w8yWXqWQYmU). The [Jax docs](https://jax.readthedocs.io/en/latest/index.html) and [Haiku docs](https://dm-haiku.readthedocs.io/en/latest/index.html) will also be needed to understand their concepts in more detail.

## Why Use Jax?
Jax's core features, speed and flexibility make it ideal for creating cutting edge models and getting them into production quickly. Jax models can be converted into a TensorFlow saved_model, which means they can be used in TensorFlow, [more info](https://github.com/google/jax/tree/main/jax/experimental/jax2tf). The saved_model can then be converted into ONNX to be used on edge devices, [more info](https://dm-haiku.readthedocs.io/en/latest/notebooks/jax2tf.html).

