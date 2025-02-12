---
hide:
  - tags
tags:
  - Python
icon: simple/python
---

# Python bindings

## :octicons-book-16: Overview

IREE offers several Python packages, including API bindings, utilities, and
integrations with frameworks:

PIP package name | Description
-- | --
[`iree-base-compiler`](https://pypi.org/project/iree-base-compiler/) | IREE's generic compiler tools and helpers
[`iree-base-runtime`](https://pypi.org/project/iree-base-runtime/) | IREE's runtime, including CPU and GPU backends
[`iree-tools-tf`](https://pypi.org/project/iree-tools-tf/) | Tools for importing from [TensorFlow](https://www.tensorflow.org/)
[`iree-tools-tflite`](https://pypi.org/project/iree-tools-tflite/) | Tools for importing from [TensorFlow Lite](https://www.tensorflow.org/lite)
[`iree-turbine`](https://pypi.org/project/iree-turbine/) | IREE's frontend for [PyTorch](https://pytorch.org/)

Collectively, these packages allow for importing from frontends, compiling
towards various targets, and executing compiled code on IREE's backends.

???+ Info "Note - deprecated package names"
    The Python packages
    [`iree-compiler`](https://pypi.org/project/iree-compiler/) and
    [`iree-runtime`](https://pypi.org/project/iree-runtime/) are deprecated.
    The packages were renamed to
    [`iree-base-compiler`](https://pypi.org/project/iree-base-compiler/) and
    [`iree-base-runtime`](https://pypi.org/project/iree-base-runtime/)
    respectively.

    To clean old installed packages, run
    `pip uninstall iree-compiler iree-runtime`.

## :octicons-download-16: Prerequisites

To use IREE's Python bindings, you will first need to install
[Python 3](https://www.python.org/downloads/) and
[pip](https://pip.pypa.io/en/stable/installing/), as needed.

???+ Tip "Tip - Virtual environments"
    We recommend using virtual environments to manage python packages, such as
    through `venv`
    ([about](https://docs.python.org/3/library/venv.html),
    [tutorial](https://docs.python.org/3/tutorial/venv.html)):

    === ":fontawesome-brands-linux: Linux"

        ``` shell
        python -m venv .venv
        source .venv/bin/activate
        ```

    === ":fontawesome-brands-apple: macOS"

        ``` shell
        python -m venv .venv
        source .venv/bin/activate
        ```

    === ":fontawesome-brands-windows: Windows"

        ``` powershell
        python -m venv .venv
        .venv\Scripts\activate.bat
        ```

    When done, run `deactivate`.

## Installing IREE packages

### :octicons-download-16: Prebuilt packages

=== ":octicons-package-16: Stable releases"

    Stable release packages are [published to PyPI](https://pypi.org/).

    ``` shell
    python -m pip install \
      iree-base-compiler \
      iree-base-runtime
    ```

=== ":octicons-beaker-16: Nightly pre-releases"

    Nightly pre-releases are published on
    [GitHub releases](https://github.com/iree-org/iree/releases).

    ``` shell hl_lines="2-4"
    python -m pip install \
      --find-links https://iree.dev/pip-release-links.html \
      --pre \
      --upgrade \
      iree-base-compiler \
      iree-base-runtime
    ```

--8<-- "docs/website/docs/snippets/_iree-dev-packages.md"

### :material-hammer-wrench: Building from source

See [Building Python bindings](../../building-from-source/getting-started.md#python-bindings)
page for instructions for building from source.

## Usage

### :material-file-document-multiple: API reference pages

Description | URL
-- | --
IREE Python APIs | <https://iree-python-api.readthedocs.io/>
IREE Turbine APIs | <https://iree-turbine.readthedocs.io/>
MLIR Python APIs | <https://mlir.llvm.org/docs/Bindings/Python/>

### Compile a program

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iree-org/iree/blob/main/samples/colab/low_level_invoke_function.ipynb)

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

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iree-org/iree/blob/main/samples/colab/low_level_invoke_function.ipynb)

```python
from iree import runtime as ireert
import numpy as np

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

### :octicons-code-16: Samples

Check out the samples in IREE's
[samples/colab/ directory](https://github.com/iree-org/iree/tree/main/samples/colab)
and the
[iree-experimental repository](https://github.com/iree-org/iree-experimental)
for examples using the Python APIs.

### :material-console: Console scripts

The Python packages include console scripts for most of IREE's native tools
like `iree-compile` and `iree-run-module`.  After installing a package from
pip, these should be added to your path automatically:

```console
$ python -m pip install iree-base-runtime
$ which iree-run-module

/projects/.venv/Scripts/iree-run-module
```

## :material-chart-line: Profiling

The tools in the `iree-base-runtime` package support variants:

| Variant name | Description |
| ------------ | ----------- |
default | Standard runtime tools
tracy | Runtime tools instrumented using the [Tracy](https://github.com/wolfpld/tracy) profiler

Switch between variants of the installed tools using the `IREE_PY_RUNTIME`
environment variable:

```bash
IREE_PY_RUNTIME=tracy iree-run-module ...
```

See the developer documentation page on
[Profiling with Tracy](../../developers/performance/profiling-with-tracy.md)
for information on using Tracy.

!!! tip - "Tip - flushing profile data"

    When writing a Python-based program that you want to profile you may need to
    insert IREE runtime calls to periodically flush the profile data:

    ```python
    device = ... # HalDevice
    device.flush_profiling()
    ```
