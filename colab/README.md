# Google Colaboratory (Colab) Notebooks

To run these notebooks with a local runtime, refer to the
[Using Colab docs](../docs/using_colab.md).

Hosted/remote runtimes are not yet supported.

## Notebooks

### [edge_detection\.ipynb](edge_detection.ipynb)

Constructs a TF module for performing image edge detection and runs it using
IREE

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google/iree/blob/main/colab/edge_detection.ipynb)

### [low_level_invoke_function\.ipynb](low_level_invoke_function.ipynb)

Shows off some concepts of the low level IREE python bindings

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google/iree/blob/main/colab/low_level_invoke_function.ipynb)

### [mnist_tensorflow\.ipynb](mnist_tensorflow.ipynb)

Trains a TensorFlow 2.0 model for recognizing handwritten digits and runs it
using IREE

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google/iree/blob/main/colab/mnist_tensorflow.ipynb)

### [resnet\.ipynb](resnet.ipynb)

Loads a pretrained
[ResNet50](https://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet50)
model and runs it using IREE

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google/iree/blob/main/colab/resnet.ipynb)

### [simple_tensorflow_module_import\.ipynb](simple_tensorflow_module_import.ipynb)

Defines a simple TF module, saves it and loads it in IREE

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google/iree/blob/main/colab/simple_tensorflow_module_import.ipynb)

## Working with GitHub

Refer to
[Colab's GitHub demo](https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb)
for general information about using Colab with GitHub.

To make changes to a notebook in this repository, one possible workflow is:

*   Open or create the notebook in Colab
*   Connect to your local runtime
*   Make your changes, run the notebook, etc.
*   Download the modified notebook using `File > Download .ipynb`
*   Move the downloaded notebook file into a clone of this repository and submit
    a pull request
