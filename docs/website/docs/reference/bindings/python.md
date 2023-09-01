---
hide:
  - tags
tags:
  - Python
icon: simple/python
---

# Python bindings

## Overview

IREE offers Python bindings split into several packages, covering different
components:

| PIP package name             | Description                                                                 |
|------------------------------|-----------------------------------------------------------------------------|
| `iree-compiler`     | IREE's generic compiler tools and helpers                                   |
| `iree-runtime`      | IREE's runtime, including CPU and GPU backends                              |
| `iree-tools-tf`     | Tools for importing from [TensorFlow](https://www.tensorflow.org/)          |
| `iree-tools-tflite` | Tools for importing from [TensorFlow Lite](https://www.tensorflow.org/lite) |
| `iree-jax`          | Tools for importing from [JAX](https://github.com/google/jax)               |

Collectively, these packages allow for importing from frontends, compiling
towards various targets, and executing compiled code on IREE's backends.

!!! Caution "Caution - Operating system support"
    Packages are currently only available on Linux and macOS. They are not
    available on Windows yet (see
    [this issue](https://github.com/openxla/iree/issues/13484)).

## :octicons-download-16: Prerequisites

To use IREE's Python bindings, you will first need to install
[Python 3](https://www.python.org/downloads/) and
[pip](https://pip.pypa.io/en/stable/installing/), as needed.

???+ Tip "Tip - Virtual environments"
    We recommend using virtual environments to manage python packages, such as
    through `venv`
    ([about](https://docs.python.org/3/library/venv.html),
    [tutorial](https://docs.python.org/3/tutorial/venv.html)):

    === "Linux"

        ``` shell
        python -m venv .venv
        source .venv/bin/activate
        ```

    === "macOS"

        ``` shell
        python -m venv .venv
        source .venv/bin/activate
        ```

    === "Windows"

        ``` powershell
        python -m venv .venv
        .venv\Scripts\activate.bat
        ```

    When done, run `deactivate`.

## Installing IREE packages

### :octicons-package-16: Prebuilt packages

=== "Stable releases"

    Stable release packages are
    [published to PyPI](https://pypi.org/user/google-iree-pypi-deploy/).

    ``` shell
    python -m pip install \
      iree-compiler \
      iree-runtime
    ```

=== ":material-alert: Nightly releases"

    Nightly releases are published on
    [GitHub releases](https://github.com/openxla/iree/releases).

    ``` shell
    python -m pip install \
      --find-links https://openxla.github.io/iree/pip-release-links.html \
      --upgrade \
      iree-compiler \
      iree-runtime
    ```

### :material-hammer-wrench: Building from source

See [Building Python bindings](../../building-from-source/getting-started.md#python-bindings)
page for instructions for building from source.

## Usage

!!! info "Info - API reference pages"

    API reference pages for IREE's runtime and compiler Python APIs are hosted on
    [readthedocs](https://iree-python-api.readthedocs.io/en/latest/).

    Documentation for the MLIR compiler Python APIs can be found at
    [https://mlir.llvm.org/docs/Bindings/Python/](https://mlir.llvm.org/docs/Bindings/Python/).

### Compile a program

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openxla/iree/blob/main/samples/colab/low_level_invoke_function.ipynb)

```python
from iree import compiler as ireec

# Compile a module.
INPUT_MLIR = """
module @arithmetic {
  func.func @simple_mul(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
    %0 = arith.mulf %arg0, %arg1 : tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}
"""

# Compile using the vmvx (reference) target:
compiled_flatbuffer = ireec.tools.compile_str(
    INPUT_MLIR,
    target_backends=["vmvx"])
```

### Run a compiled program

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openxla/iree/blob/main/samples/colab/low_level_invoke_function.ipynb)

```python
from iree import runtime as ireert

# Register the module with a runtime context.
# Use the "local-task" CPU driver, which can load the vmvx executable:
config = ireert.Config("local-task")
ctx = ireert.SystemContext(config=config)
vm_module = ireert.VmModule.copy_buffer(ctx.instance, compiled_flatbuffer)
ctx.add_vm_module(vm_module)

# Invoke the function and print the result.
print("INVOKE simple_mul")
arg0 = np.array([1., 2., 3., 4.], dtype=np.float32)
arg1 = np.array([4., 5., 6., 7.], dtype=np.float32)
f = ctx.modules.arithmetic["simple_mul"]
results = f(arg0, arg1).to_host()
print("Results:", results)
```

### Samples

Check out the samples in IREE's
[samples/colab/ directory](https://github.com/openxla/iree/tree/main/samples/colab)
and the [iree-samples repository](https://github.com/iree-org/iree-samples) for
examples using the Python APIs.
