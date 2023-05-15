# Getting Started Guide

IREE supports popular machine learning frameworks using the same underlying
technology.

## :octicons-list-unordered-16: Supported frameworks

See end-to-end examples of how to use each framework with IREE:

* [TensorFlow](./tensorflow.md) and [TensorFlow Lite](./tflite.md)
* [JAX](./jax.md)
* [PyTorch](./pytorch.md)

Importing from other frameworks is planned - stay tuned!

## :octicons-code-16: Samples

Check out the samples in IREE's
[`samples/` directory](https://github.com/openxla/iree/tree/main/samples),
as well as the
[iree-samples repository](https://github.com/iree-org/iree-samples).

## :octicons-package-dependents-16: Export/Import

Each machine learning framework has some "export" mechanism that snapshots the
structure and data in your program. These exported programs can then be
"imported" into IREE's compiler by using either a stable import format or one of
IREE's importer tools. This export/import process is specific to each frontend
and typically involves a number of stages:

1. Capture/trace/freeze the ML model into a graph
2. Write that graph to an interchange format (e.g. SavedModel, TorchScript)
3. Load the saved program into an import tool and convert to MLIR
4. Legalize the graph's operations so only IREE-compatible operations remain
5. Write the imported MLIR to a file

This fully imported form can then be compiled indepedently of the source
language and framework.

## :octicons-gear-16: Compilation

During compilation we load an MLIR file and compile for the specified set of
backends (CPU, GPU, etc).  Each of these backends creates custom native code to
execute on the target device.  Once compiled, the resulting artifact can be
executed on the specified devices using IREE's runtime.

## :octicons-rocket-16: Execution

The final stage is executing the now compiled module. This involves selecting
what compute devices should be used, loading the module, and executing the
module with the intended inputs. IREE provides several
[language bindings](../bindings/index.md) for it's runtime API.
