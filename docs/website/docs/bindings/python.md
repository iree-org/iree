# Python bindings

!!! info

    API reference pages for IREE's runtime and compiler Python APIs are hosted on
    [readthedocs](https://iree-python-api.readthedocs.io/en/latest/).

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
    [this issue](https://github.com/openxla/iree/issues/6417)).

## Prerequisites

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

### Prebuilt packages

Stable release packages are published to
[PyPI](https://pypi.org/user/google-iree-pypi-deploy/).

=== "Minimal"

    To install just the core IREE packages:

    ``` shell
    python -m pip install \
      iree-compiler \
      iree-runtime
    ```

=== "All packages"

    To install IREE packages with tools for all frontends:

    ``` shell
    python -m pip install \
      iree-compiler \
      iree-runtime \
      iree-tools-tf \
      iree-tools-tflite
    ```

!!! Tip "Tip - Nightly releases"

    Unstable packages are also published nightly on
    [GitHub releases](https://github.com/openxla/iree/releases). To use these,
    run `pip install` with this option:

    ```
    --find-links https://openxla.github.io/iree/pip-release-links.html
    ```

### Building from source

See [Building Python bindings](../../building-from-source/getting-started/#python-bindings)
page for instructions for building from source.

## Using the Python bindings

API reference pages for IREE's runtime and compiler Python APIs are hosted on
[readthedocs](https://iree-python-api.readthedocs.io/en/latest/).

Check out the samples in IREE's
[samples/colab/ directory](https://github.com/openxla/iree/tree/main/samples/colab)
and the [iree-samples repository](https://github.com/iree-org/iree-samples) for
examples using the Python APIs.

<!-- ## Troubleshooting -->

<!-- TODO(scotttodd): update python, update pip, search GitHub issues -->
