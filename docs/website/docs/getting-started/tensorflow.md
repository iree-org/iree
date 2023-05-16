# TensorFlow Integration

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
[building from source](../building-from-source/getting-started.md#python-bindings):

```shell
python -m pip install \
  iree-compiler \
  iree-runtime \
  iree-tools-tf
```

!!! Caution
    The TensorFlow package is currently only available on Linux and macOS. It
    is not available on Windows yet (see
    [this issue](https://github.com/openxla/iree/issues/6417)).

## Importing models

IREE compilers transform a model into its final deployable format in several
sequential steps. The first step for a TensorFlow model is to use either the
`iree-import-tf` command-line tool or IREE's Python APIs to import the model
into a format (i.e., [MLIR](https://mlir.llvm.org/)) compatible with the generic
IREE compilers.

### From SavedModel on TensorFlow Hub

IREE supports importing and using SavedModels from
[TensorFlow Hub](https://www.tensorflow.org/hub).

#### Using the command-line tool

First download the SavedModel and load it to get the serving signature, which
is used as the entry point for IREE compilation flow:

``` python
import tensorflow.compat.v2 as tf
loaded_model = tf.saved_model.load('/path/to/downloaded/model/')
print(list(loaded_model.signatures.keys()))
```

!!! note
    If there are no serving signatures in the original SavedModel, you may add
    them by yourself by following
    ["Missing serving signature in SavedModel"](#missing-serving-signature-in-savedmodel).

Then you can import the model with `iree-import-tf`. You can read the options
supported via `iree-import-tf -help`. Using
[MobileNet v2](https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification)
as an example and assuming the serving signature is `predict`:

``` shell
iree-import-tf
  --tf-import-type=savedmodel_v1 \
  --tf-savedmodel-exported-names=predict \
  /path/to/savedmodel -o iree_input.mlir
```

!!! tip

    `iree-import-tf` is installed as
    `/path/to/python/site-packages/iree/tools/tf/iree-import-tf`.
    You can find out the full path to the `site-packages` directory via the
    `python -m site` command.

    `-tf-import-type` needs to match the SavedModel version. You can try both v1
    and v2 if you see one of them gives an empty dump.

Afterwards you can further compile the model in `iree_input.mlir` for
[CPU](../deployment-configurations/cpu.md) or
[GPU](../deployment-configurations/gpu-vulkan.md).

<!-- TODO(??): overview of APIs available, code snippets (lift from Colab?) -->

## Training

!!! todo
    Discuss training

## Samples

| Colab notebooks |  |
| -- | -- |
Training an MNIST digits classifier | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openxla/iree/blob/main/samples/colab/mnist_training.ipynb)
Edge detection module | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openxla/iree/blob/main/samples/colab/edge_detection.ipynb)
Pretrained ResNet50 inference | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openxla/iree/blob/main/samples/colab/resnet.ipynb)
TensorFlow Hub Import | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openxla/iree/blob/main/samples/colab/tensorflow_hub_import.ipynb)

End-to-end execution tests can be found in IREE's
[integrations/tensorflow/e2e/](https://github.com/openxla/iree/tree/main/integrations/tensorflow/e2e)
directory.

## Troubleshooting

### Missing serving signature in SavedModel

Sometimes SavedModels are exported without explicit
[serving signatures](https://www.tensorflow.org/guide/saved_model#specifying_signatures_during_export).
This happens by default for TensorFlow Hub SavedModels. However, serving
signatures are required as entry points for IREE compilation flow. You
can use Python to load and re-export the SavedModel to give it serving
signatures. For example, for
[MobileNet v2](https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification),
assuming we want the serving signature to be `predict` and operating on a
224x224 RGB image:

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
save it to `/path/to/resaved/model/`.
