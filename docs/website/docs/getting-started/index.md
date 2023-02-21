# Getting Started Guide

## Setup

Use the following command for the default installation, or check out the
comprehensive installation [guide](../bindings/python.md) if your needs are
more complex.

``` bash
python -m pip install \
  iree-compiler \
  iree-runtime \
  iree-tools-tf \
  iree-tools-tflite \
  iree-tools-xla
```

## Supported frameworks

See end-to-end examples of how to execute a variety models on IREE. This covers
the import, compilation, and execution of the provided model.

* [TensorFlow](./tensorflow.md)
* [TensorFlow Lite](./tflite.md)
* [JAX](./jax.md)
* [PyTorch](./pytorch.md)

Importing from other frameworks is planned - stay tuned!

## Samples

Check out the samples in IREE's
[samples/colab/ directory](https://github.com/openxla/iree/tree/main/colab),
as well as the [iree-samples repository](https://github.com/iree-org/iree-samples),
which contains workflow comparisons across frameworks.

## Import

Importing models takes known file types and imports into a form that the core
IREE compiler is able to ingest. This import process is specific to each
frontend and typically involves a number of stages:

* Load the source format
* Legalize operations specific each specific frontend to legal IR
* Validate only IREE compatible operations remain
* Write the remaining IR to a file

This fully legalized form can then be compiled without dependencies on the
source model language.

## Compilation

During compilation we load an MLIR file and compile for the specified set of
backends (CPU, GPU, etc).  Each of these backends creates custom native code to
execute on the target device.  Once compiled, the resulting bytecode is
exported to an IREE bytecode file that can be executed on the specified devices.

## Execution

The final stage is executing the now compiled module. This involves selecting
what compute devices should be used, loading the module, and executing the
module with the intended inputs. For testing, IREE includes a Python API.
However, on mobile and embedded devices you will want to use the
[C API](../deployment-configurations/index.md).
