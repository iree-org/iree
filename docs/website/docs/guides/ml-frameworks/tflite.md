---
hide:
  - tags
tags:
  - Python
  - TensorFlow
icon: simple/tensorflow
---

# TensorFlow Lite integration

## :octicons-book-16: Overview

IREE supports compiling and running TensorFlow Lite (TFLite) programs stored as
[TFLite FlatBuffers](https://www.tensorflow.org/lite/guide). These files can be
imported into an IREE-compatible format then compiled to a series of backends.

``` mermaid
graph LR
  accTitle: TFLite to runtime deployment workflow overview
  accDescr {
    Programs start as TensorFlow Lite FlatBuffers.
    Programs are imported into MLIR's TOSA dialect using iree-import-tflite.
    The IREE compiler uses the imported MLIR.
    Compiled programs are used by the runtime.
  }

  subgraph A[TFLite]
    A1[FlatBuffer]
  end

  subgraph B[MLIR]
    B1[TOSA]
  end

  C[IREE compiler]
  D[Runtime deployment]

  A -- iree-import-tflite --> B
  B --> C
  C --> D
```

## :octicons-download-16: Prerequisites

1. Install TensorFlow by following the
    [official documentation](https://www.tensorflow.org/install):

    ```shell
    python -m pip install tf-nightly
    ```

2. Install IREE packages, either by
    [building from source](../../building-from-source/getting-started.md#python-bindings)
    or from pip:

    === "Stable releases"

        Stable release packages are
        [published to PyPI](https://pypi.org/user/google-iree-pypi-deploy/).

        ``` shell
        python -m pip install \
          iree-compiler \
          iree-runtime \
          iree-tools-tflite
        ```

    === ":material-alert: Nightly releases"

        Nightly releases are published on
        [GitHub releases](https://github.com/openxla/iree/releases).

        ``` shell
        python -m pip install \
          --find-links https://openxla.github.io/iree/pip-release-links.html \
          --upgrade \
          iree-compiler \
          iree-runtime \
          iree-tools-tflite
        ```

## :octicons-package-dependents-16: Importing and Compiling

IREE's tooling is divided into two components: import and compilation.

1. The import tool converts the TFLite FlatBuffer to an IREE compatible form,
  validating that only IREE compatible operations remain. Containing a combination
  of TOSA and IREE operations.
2. The compilation stage generates the bytecode module for a list of targets,
  which can be executed by IREE.

### Using Command Line Tools

These two stages can be completed entirely via the command line.

``` shell
WORKDIR="/tmp/workdir"
TFLITE_URL="https://storage.googleapis.com/iree-model-artifacts/tflite-integration-tests/posenet_i8.tflite"
TFLITE_PATH=${WORKDIR}/model.tflite
IMPORT_PATH=${WORKDIR}/tosa.mlir
MODULE_PATH=${WORKDIR}/module.vmfb

# Fetch the sample model
wget ${TFLITE_URL} -O ${TFLITE_PATH}

# Import the sample model to an IREE compatible form
iree-import-tflite ${TFLITE_PATH} -o ${IMPORT_PATH}

# Compile for the CPU backend
iree-compile \
    --iree-input-type=tosa \
    --iree-hal-target-backends=llvm-cpu \
    ${IMPORT_PATH} \
    -o ${MODULE_PATH}
```

### Using the Python API

The example below demonstrates downloading, compiling, and executing a TFLite
model using the Python API. This includes some initial setup to declare global
variables, download the sample module, and download the sample inputs.

Declaration of absolute paths for the sample repo and import all required
libraries. The default setup uses the CPU backend as the only target. This can
be reconfigured to select alternative targets.

``` python
import iree.compiler.tflite as iree_tflite_compile
import iree.runtime as iree_rt
import numpy
import os
import urllib.request

from PIL import Image

workdir = "/tmp/workdir"
os.makedirs(workdir, exist_ok=True)

tfliteFile = "/".join([workdir, "model.tflite"])
jpgFile = "/".join([workdir, "input.jpg"])
tfliteIR = "/".join([workdir, "tflite.mlir"])
tosaIR = "/".join([workdir, "tosa.mlir"])
bytecodeModule = "/".join([workdir, "iree.vmfb"])

backends = ["llvm-cpu"]
config = "local-task"
```

The TFLite sample model and input are downloaded locally.

``` python
tfliteUrl = "https://storage.googleapis.com/iree-model-artifacts/tflite-integration-tests/posenet_i8.tflite"
jpgUrl = "https://storage.googleapis.com/iree-model-artifacts/tflite-integration-tests/posenet_i8_input.jpg"

urllib.request.urlretrieve(tfliteUrl, tfliteFile)
urllib.request.urlretrieve(jpgUrl, jpgFile)
```

Once downloaded we can compile the model for the selected backends. Both the
TFLite and TOSA representations of the model are saved for debugging purposes.
This is optional and can be omitted.

``` python
iree_tflite_compile.compile_file(
  tfliteFile,
  input_type="tosa",
  output_file=bytecodeModule,
  save_temp_tfl_input=tfliteIR,
  save_temp_iree_input=tosaIR,
  target_backends=backends,
  import_only=False)
```

After compilation is completed we configure the VmModule using the local-task
configuration and compiled IREE module.

``` python
config = iree_rt.Config("local-task")
context = iree_rt.SystemContext(config=config)
with open(bytecodeModule, 'rb') as f:
  vm_module = iree_rt.VmModule.from_flatbuffer(config.vm_instance, f.read())
  context.add_vm_module(vm_module)
```

Finally, the IREE module is loaded and ready for execution. Here we load the
sample image, manipulate to the expected input size, and execute the module. By
default TFLite models include a single function named 'main'. The final results
are printed.

``` python
im = numpy.array(Image.open(jpgFile).resize((192, 192))).reshape((1, 192, 192, 3))
args = [im]

invoke = context.modules.module["main"]
iree_results = invoke(*args)
print(iree_results)
```

## :octicons-code-16: Samples

* The
[tflitehub folder](https://github.com/iree-org/iree-samples/tree/main/tflitehub)
in the [iree-samples repository](https://github.com/iree-org/iree-samples)
contains test scripts to compile, run, and compare various TensorFlow Lite
models sourced from [TensorFlow Hub](https://tfhub.dev/).

* An example smoke test of the
[TensorFlow Lite C API](https://github.com/openxla/iree/tree/main/runtime/bindings/tflite)
is available
[here](https://github.com/openxla/iree/blob/main/runtime/bindings/tflite/smoke_test.cc).

| Colab notebooks |  |
| -- | -- |
Text classification with TFLite and IREE | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openxla/iree/blob/main/samples/colab/tflite_text_classification.ipynb)

## :octicons-question-16: Troubleshooting

Failures during the import step usually indicate a failure to lower from
TensorFlow Lite's operations to TOSA, the intermediate representation used by
IREE. Many TensorFlow Lite operations are not fully supported, particularly
those than use dynamic shapes. Please reach out on one of IREE's
[communication channels](../../index.md#communication-channels) if you notice
something missing.
