# Python bindings

IREE offers Python bindings split into several packages, covering different
components:

| PIP package name             | Description                                                                 |
|------------------------------|-----------------------------------------------------------------------------|
| `iree-compiler-snapshot`     | IREE's generic compiler tools and helpers                                   |
| `iree-runtime-snapshot`      | IREE's runtime, including CPU and GPU backends                              |
| `iree-tools-tf-snapshot`     | Tools for importing from [TensorFlow](https://www.tensorflow.org/)          |
| `iree-tools-tflite-snapshot` | Tools for importing from [TensorFlow Lite](https://www.tensorflow.org/lite) |
| `iree-tools-xla-snapshot`    | Tools for importing from [XLA](https://www.tensorflow.org/xla)              |
| `iree-jax-snapshot`          | Tools for importing from [JAX](https://github.com/google/jax)               |

Collectively, these packages allow for importing from frontends, compiling
towards various targets, and executing compiled code on IREE's backends.

!!! warning
    The TensorFlow, TensorFlow Lite, and XLA packages are currently only
    available on Linux and macOS. They are not available on Windows yet (see
    [this issue](https://github.com/google/iree/issues/6417)).

<!-- TODO(??): API references for packages/modules -->
<!-- TODO(??): at least link to source code and sample Colab notebooks for now -->
<!-- TODO(??): link to frontend docs -->

## Prerequisites

To use IREE's Python bindings, you will first need to install
[Python 3](https://www.python.org/downloads/) and
[pip](https://pip.pypa.io/en/stable/installing/), as needed.

???+ tip
    We recommend using virtual environments to manage python packages, such as
    through `venv`
    ([about](https://docs.python.org/3/library/venv.html),
    [tutorial](https://docs.python.org/3/tutorial/venv.html)):

    === "Linux and MacOS"

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

<!-- TODO(??): use setup.py install_requires for any dependencies IREE needs -->

Next, install packages:

``` shell
python -m pip install --upgrade pip
python -m pip install numpy absl-py
```

## Installing IREE packages

### Prebuilt packages

For now, packages can be installed from our
[GitHub releases](https://github.com/google/iree/releases):

=== "Minimal"

    To install just the core IREE packages:

    ``` shell
    python -m pip install \
      iree-compiler-snapshot \
      iree-runtime-snapshot \
      --find-links https://github.com/google/iree/releases
    ```

=== "All packages"

    To install IREE packages with tools for all frontends:

    ``` shell
    python -m pip install \
      iree-compiler-snapshot \
      iree-runtime-snapshot \
      iree-tools-tf-snapshot \
      iree-tools-tflite-snapshot \
      iree-tools-xla-snapshot \
      --find-links https://github.com/google/iree/releases
    ```

!!! info
    We plan to publish packages on [PyPI](https://pypi.org/) as they become
    more stable.

### Building from source

See [Building Python bindings](../../building-from-source/optional-features/#building-python-bindings)
page for instructions for building from source.

## Using the Python bindings

<!-- TODO(??): general flow: import, compile, run -->
<!-- TODO(??): high level API vs low level API -->
<!-- TODO(??): examples split by TF/TFLite/XLA/JAX (and link to frontend docs) -->

## Troubleshooting

<!-- TODO(scotttodd): update python, update pip, search GitHub issues -->
<!-- TODO(scotttodd): add note about frontend tools not being published for Windows -->
