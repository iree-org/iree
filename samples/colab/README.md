# Google Colaboratory (Colab) Notebooks

These [Colab](https://colab.google/) notebooks contain interactive sample
applications using IREE's Python bindings and ML framework integrations.

## Notebooks

Framework | Notebook file | Description | Link
--------  | ------------- | ----------- | ----
Generic | [low_level_invoke_function](low_level_invoke_function.ipynb) | Shows low level IREE python binding concepts | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openxla/iree/blob/main/samples/colab/low_level_invoke_function.ipynb)
PyTorch | [pytorch_aot_simple](pytorch_aot_simple.ipynb) | Uses [SHARK-Turbine](https://github.com/nod-ai/SHARK-Turbine) to export a simple PyTorch program | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openxla/iree/blob/main/samples/colab/pytorch_aot_simple.ipynb)
PyTorch | [pytorch_jit](pytorch_jit.ipynb) | Uses [SHARK-Turbine](https://github.com/nod-ai/SHARK-Turbine) for eager execution in a PyTorch session | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openxla/iree/blob/main/samples/colab/pytorch_jit.ipynb)
TensorFlow | [tensorflow_edge_detection](tensorflow_edge_detection.ipynb) |Performs image edge detection | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openxla/iree/blob/main/samples/colab/tensorflow_edge_detection.ipynb)
TensorFlow | [tensorflow_hub_import](tensorflow_hub_import.ipynb) | Imports a [MobileNet V2](https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification) model from [TensorFlow Hub](https://tfhub.dev/) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openxla/iree/blob/main/samples/colab/tensorflow_hub_import.ipynb)
TensorFlow | [tensorflow_mnist_training](tensorflow_mnist_training.ipynb) | Compiles, trains, and executes a neural network | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openxla/iree/blob/main/samples/colab/tensorflow_mnist_training.ipynb)
TensorFlow | [tensorflow_resnet](tensorflow_resnet.ipynb) | Compiles and runs a pretrained [ResNet50](https://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet50) model | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openxla/iree/blob/main/samples/colab/tensorflow_resnet.ipynb)
TFLite | [tflite_text_classification](tflite_text_classification.ipynb) | Compiles and runs a pretrained [text classification](https://www.tensorflow.org/lite/examples/text_classification/overview) model | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openxla/iree/blob/main/samples/colab/tflite_text_classification.ipynb)

## Working with GitHub

Refer to
[Colab's GitHub demo](https://colab.research.google.com/github/googlecolab/colabtools/blob/main/notebooks/colab-github-demo.ipynb)
for general information about using Colab with GitHub.

To make changes to a notebook in this repository, one possible workflow is:

*   Open or create the notebook in Colab
*   Connect to a hosted or local runtime
*   Make your changes, run the notebook, etc.
*   Download the modified notebook using `File > Download .ipynb`
*   Move the downloaded notebook file into a clone of this repository and submit
    a pull request

## Testing

This notebooks are tested continuously by the samples.yml CI job.
