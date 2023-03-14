# Google Colaboratory (Colab) Notebooks

## Notebooks

### [edge_detection\.ipynb](edge_detection.ipynb)

Constructs a TF module for performing image edge detection and runs it using
IREE

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openxla/iree/blob/main/samples/colab/edge_detection.ipynb)

### [low_level_invoke_function\.ipynb](low_level_invoke_function.ipynb)

Shows off some concepts of the low level IREE python bindings

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openxla/iree/blob/main/samples/colab/low_level_invoke_function.ipynb)

### [mnist_training\.ipynb](mnist_training.ipynb)

Compile, train and execute a TensorFlow Keras neural network with IREE

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openxla/iree/blob/main/samples/colab/mnist_training.ipynb)

### [resnet\.ipynb](resnet.ipynb)

Loads a pretrained
[ResNet50](https://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet50)
model and runs it using IREE

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openxla/iree/blob/main/samples/colab/resnet.ipynb)

### [tensorflow_hub_import\.ipynb](tensorflow_hub_import.ipynb)

Downloads a pretrained
[MobileNet V2](https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification)
model, pre-processes it for import, then compiles it using IREE

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openxla/iree/blob/main/samples/colab/tensorflow_hub_import.ipynb)

### [tflite_text_classification\.ipynb](tflite_text_classification.ipynb)

Downloads a pretrained
[TFLite text classification](https://www.tensorflow.org/lite/examples/text_classification/overview)
model, and runs it using TFLite and IREE

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openxla/iree/blob/main/samples/colab/tflite_text_classification.ipynb)

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
