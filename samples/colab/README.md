# Google Colaboratory (Colab) Notebooks

These [Colab](https://colab.google/) notebooks contain interactive sample
applications using IREE's Python bindings and ML framework integrations.

## Notebooks

Framework | Notebook file | Description | Link
--------  | ------------- | ----------- | ----
Generic | [low_level_invoke_function\.ipynb](low_level_invoke_function.ipynb) | Shows off some concepts of the low level IREE python bindings | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openxla/iree/blob/main/samples/colab/low_level_invoke_function.ipynb)
PyTorch | [pytorch_jit\.ipynb](pytorch_jit.ipynb) | Uses [SHARK-Turbine](https://github.com/nod-ai/SHARK-Turbine) for eager execution in a PyTorch session | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openxla/iree/blob/main/samples/colab/pytorch_jit.ipynb)
TensorFlow | [edge_detection\.ipynb](edge_detection.ipynb) |Performs image edge detection using TF and IREE | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openxla/iree/blob/main/samples/colab/edge_detection.ipynb)
TensorFlow | [mnist_training\.ipynb](mnist_training.ipynb) | Compile, train, and execute a neural network with IREE | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openxla/iree/blob/main/samples/colab/mnist_training.ipynb)
TensorFlow | [resnet\.ipynb](resnet.ipynb) | Loads a pretrained [ResNet50](https://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet50) model and runs it using IREE | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openxla/iree/blob/main/samples/colab/resnet.ipynb)
TensorFlow | [tensorflow_hub_import\.ipynb](tensorflow_hub_import.ipynb) | Runs a pretrained [MobileNet V2](https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification) model using IREE | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openxla/iree/blob/main/samples/colab/tensorflow_hub_import.ipynb)
TFLite | [tflite_text_classification\.ipynb](tflite_text_classification.ipynb) | Runs a pretrained [text classification](https://www.tensorflow.org/lite/examples/text_classification/overview) model using IREE | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openxla/iree/blob/main/samples/colab/tflite_text_classification.ipynb)

## Working with GitHub

Refer to
[Colab's GitHub demo](https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb)
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
