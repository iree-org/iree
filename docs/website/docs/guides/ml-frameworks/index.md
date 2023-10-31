# ML frameworks

IREE supports popular machine learning frameworks using the same underlying
technology.

``` mermaid
graph LR
  accTitle: ML framework to runtime deployment workflow overview
  accDescr {
    Programs start in some ML framework.
    Programs are imported into MLIR.
    The IREE compiler uses the imported MLIR.
    Compiled programs are used by the runtime.
  }

  A[ML frameworks]
  B[Imported MLIR]
  C[IREE compiler]
  D[Runtime deployment]

  A --> B
  B --> C
  C --> D
```

## :octicons-list-unordered-16: Supported frameworks

See end-to-end examples of how to use each framework with IREE:

* [:simple-tensorflow: TensorFlow](./tensorflow.md) and
  [:simple-tensorflow: TensorFlow Lite](./tflite.md)
* [:simple-python: JAX](./jax.md)
* [:simple-pytorch: PyTorch](./pytorch.md)

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
IREE's importer tools.

This export/import process is specific to each frontend and typically involves a
number of stages:

1. Capture/trace/freeze the ML model into a graph
2. Write that graph to an interchange format (e.g. SavedModel, TorchScript)
3. Load the saved program into an import tool and convert to MLIR
4. Legalize the graph's operations so only IREE-compatible operations remain
5. Write the imported MLIR to a file

This fully imported form can then be compiled indepedently of the source
language and framework.

## :octicons-gear-16: Compilation

IREE compiles MLIR files for specified sets of backends (CPU, GPU, etc). Each
backend generates optimized native code custom to the input program and
intended target platform. Once compiled, modules can be executed using IREE's
runtime.

See the [deployment configuration guides](../deployment-configurations/index.md)
for details on selecting a compiler backend and tuning options for your choice
of target platform(s) or device(s).

## :octicons-rocket-16: Execution

Compiled modules can be executed by selecting what compute devices to use,
loading the module, and then executing it with the intended inputs. IREE
provides several [language bindings](../../reference/bindings/index.md) for its
runtime API.
