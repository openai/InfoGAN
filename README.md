**Status:** Archive (code is provided as-is, no updates expected)

# InfoGAN

Code for reproducing key results in the paper [InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets](https://arxiv.org/abs/1606.03657) by Xi Chen, Yan Duan, Rein Houthooft, John Schulman, Ilya Sutskever, Pieter Abbeel.

## Dependencies

This project currently requires the dev version of TensorFlow available on Github: https://github.com/tensorflow/tensorflow. As of the release, the latest commit is [79174a](https://github.com/tensorflow/tensorflow/commit/79174afa30046ecdc437b531812f2cb41a32695e).

In addition, please `pip install` the following packages:
- `prettytensor`
- `progressbar`
- `python-dateutil`

## Running in Docker

```bash
$ git clone git@github.com:openai/InfoGAN.git
$ docker run -v $(pwd)/InfoGAN:/InfoGAN -w /InfoGAN -it -p 8888:8888 gcr.io/tensorflow/tensorflow:r0.9rc0-devel
root@X:/InfoGAN# pip install -r requirements.txt
root@X:/InfoGAN# python launchers/run_mnist_exp.py
```

## Running Experiment

We provide the source code to run the MNIST example:

```bash
PYTHONPATH='.' python launchers/run_mnist_exp.py
```

You can launch TensorBoard to view the generated images:

```bash
tensorboard --logdir logs/mnist
```
