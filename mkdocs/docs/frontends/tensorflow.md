# TensorFlow frontend

IREE supports compiling and running TensorFlow programs represented as
`tf.Module` [classes](https://www.tensorflow.org/api_docs/python/tf/Module)
or stored in the `SavedModel`
[format](https://www.tensorflow.org/guide/saved_model).

<!-- TODO(??): notes about TensorFlow 2.0, supported features? -->

## Prerequisites

Install TensorFlow by following the
[official documentation](https://www.tensorflow.org/install):

```shell
python -m pip install tf-nightly
```

Install IREE pip packages, either from pip or by
[building from source](../building-from-source/python.md):

```shell
python -m pip install \
  iree-compiler-snapshot \
  iree-runtime-snapshot \
  iree-tools-tf-snapshot \
  -f https://github.com/google/iree/releases
```

## Importing models

<!-- TODO(??): overview of APIs available, code snippets (lift from Colab?) -->

## Training

<!-- TODO(??): discuss training -->

## Samples

| Colab notebooks |  |
| -- | -- |
Training an MNIST digits classifier | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google/iree/blob/main/colab/mnist_training.ipynb)
Edge detection module | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google/iree/blob/main/colab/edge_detection.ipynb)
Pretrained ResNet50 inference | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google/iree/blob/main/colab/resnet.ipynb)

End-to-end execution tests can be found in IREE's
[integrations/tensorflow/e2e/](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e)
directory.
