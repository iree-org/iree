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

### From SavedModel on TensorFlow Hub

IREE supports importing and using SavedModels from
[TensorFlow Hub](https://www.tensorflow.org/hub). Here we use [MobileNet
v2](https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification) as
an example.


#### Using the command-line tool

IREE compilers transform a model into its final deployable format in many
sequential steps. The first step is to import the model authored with Python
in TensorFlow into a format (i.e., [MLIR](https://mlir.llvm.org/)) expected
by main IREE compilers. This can be done via the `iree-tf-import` tool.

!!! note
    `iree-tf-import` is installed as `/path/to/python/site-packages/iree/tools/tf/iree-tf-import`
    via the `iree-tools-tf-snapshot` Python package. You can find out the full
    path to the `site-packages` directory via the `python -m site` command.

First download the SavedModel and re-save it with a [serving signature](https://www.tensorflow.org/guide/saved_model#specifying_signatures_during_export).

!!! info
    This is necessary because by default TensorFlow Hub SavedModels might not
    have serving signatures, which are required for IREE compilation flow.

``` python
import tensorflow.compat.v2 as tf
loaded_model = tf.saved_model.load('/path/to/downloaded/model/')
call = loaded_model.__call__.get_concrete_function(
         tf.TensorSpec([1, 224, 224, 3], tf.float32))
signatures = {'predict': call}
tf.saved_model.save(loaded_model,
  '/path/to/resaved/model/', signatures=signatures)
```

The above will create a new SavedModel with a serving signature, `predict`, and
save it to `/path/to/resaved/model/` Then you can import it via
`iree-tf-import`:

``` shell
iree-tf-import
  --tf-savedmodel-exported-names=predict \
  --tf-import-type=savedmodel_v1 \
  /path/to/resaved/model -o iree_input.mlir
```

Afterwards you can further compile the model for [CPU](/backends/cpu-llvm/) or
[GPU](/backends/gpu-vulkan/).

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
