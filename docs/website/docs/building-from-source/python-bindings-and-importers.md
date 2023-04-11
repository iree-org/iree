# Python bindings and importers

!!! Attention
    These components are more complicated to build from source than the rest of
    the project. If your usage does not require making source changes, prefer to
    [install from the official binary distributions](../bindings/python.md#installing-iree-packages)
    instead.

This page covers how to build IREE's Python-based bindings and import tools from
source. These components are built using CMake as well as other dependencies and
each section extends the basic build steps in the
[getting started](./getting-started.md) page.

## Building Python bindings

This section describes how to build and interactively use built-from-source
Python bindings for the following packages:

| Python Import          | Description                                    |
|------------------------|------------------------------------------------|
| `import iree.compiler` | IREE's generic compiler tools and helpers      |
| `import iree.runtime`  | IREE's runtime, including CPU and GPU backends |

Also see [instructions for installing pre-built binaries](../bindings/python.md).

**Pre-requisites:**

* A relatively recent Python3 installation >=3.8 (we aim to support
  [non-eol Python versions](https://endoflife.date/python)).

**CMake Variables:**

* **`IREE_BUILD_PYTHON_BINDINGS`** : `BOOL`

    Enables building of Python bindings under `runtime/bindings/python` in the
    repository. Defaults to `OFF`.

* **`Python3_EXECUTABLE`** : `PATH`

    Full path to the Python3 executable to build against. If not specified, CMake
    will auto-detect this, which often yields incorrect results on systems
    with multiple Python versions. Explicitly setting this is recommended.
    Note that mixed case of the option.

### Environment setup

We recommend using virtual environments to manage python packages, such
as through `venv`, which may need to be installed via your system
package manager ([about](https://docs.python.org/3/library/venv.html),
[tutorial](https://docs.python.org/3/tutorial/venv.html)):

=== "Linux"

    ``` shell
    # Make sure your 'python' is what you expect. Note that on multi-python
    # systems, this may have a version suffix, and on many Linuxes where
    # python2 and python3 can co-exist, you may also want to use `python3`.
    which python
    python --version

    # Create a persistent virtual environment (first time only).
    python -m venv iree.venv

    # Activate the virtual environment (per shell).
    # Now the `python` command will resolve to your virtual environment
    # (even on systems where you typically use `python3`).
    source iree.venv/bin/activate

    # Upgrade PIP. On Linux, many packages cannot be installed for older
    # PIP versions. See: https://github.com/pypa/manylinux
    python -m pip install --upgrade pip

    # Install IREE build pre-requisites.
    python -m pip install -r ./runtime/bindings/python/iree/runtime/build_requirements.txt
    ```

=== "macOS"

    ``` shell
    # Make sure your 'python' is what you expect. Note that on multi-python
    # systems, this may have a version suffix, and on macOS where python2
    # and python3 can co-exist, you may also want to use `python3`.
    which python
    python --version

    # Create a persistent virtual environment (first time only).
    python -m venv iree.venv

    # Activate the virtual environment (per shell).
    # Now the `python` command will resolve to your virtual environment
    # (even on systems where you typically use `python3`).
    source iree.venv/bin/activate

    # Upgrade PIP.
    python -m pip install --upgrade pip

    # Install IREE build pre-requisites.
    python -m pip install -r ./runtime/bindings/python/iree/runtime/build_requirements.txt
    ```

=== "Windows"

    ``` powershell
    # Make sure your 'python' is what you expect.
    # Also consider using the Python launcher 'py' instead of 'python':
    # https://docs.python.org/3/using/windows.html#python-launcher-for-windows
    which python
    python --version
    py --list-paths

    # Create a persistent virtual environment (first time only).
    python -m venv .venv

    # Activate the virtual environment (per shell).
    # Now the `python` command will resolve to your virtual environment
    # (even on systems where you typically use `python3`).
    .venv\Scripts\activate.bat

    # Upgrade PIP.
    python -m pip install --upgrade pip

    # Install IREE build pre-requisites.
    python -m pip install -r runtime\bindings\python\iree\runtime\build_requirements.txt
    ```

When you are done with the venv, you can close it by closing your shell
or running `deactivate`.

### Building with CMake

From the `iree-build` directory:

=== "Linux"

    ``` shell
    cmake \
        -GNinja \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DIREE_BUILD_PYTHON_BINDINGS=ON \
        -DPython3_EXECUTABLE="$(which python)" \
        .
    cmake --build .

    # Add the bindings/python paths to PYTHONPATH and use the API.
    source .env && export PYTHONPATH
    python -c "import iree.compiler"
    python -c "import iree.runtime"
    ```

=== "macOS"

    ``` shell
    cmake \
        -GNinja \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DIREE_BUILD_PYTHON_BINDINGS=ON \
        -DPython3_EXECUTABLE="$(which python)" \
        .
    cmake --build .

    # Add the bindings/python paths to PYTHONPATH and use the API.
    source .env && export PYTHONPATH
    python -c "import iree.compiler"
    python -c "import iree.runtime"
    ```

=== "Windows"

    ``` powershell
    cmake -GNinja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DIREE_BUILD_PYTHON_BINDINGS=ON .
    cmake --build .

    # Add the bindings/python paths to PYTHONPATH and use the API.
    .env.bat
    python -c "import iree.compiler"
    python -c "import iree.runtime"
    ```

Tests can now be run individually via python or via ctest.

## Building TensorFlow frontend bindings

This section describes how to build compiler tools and Python bindings for
importing models from various TensorFlow-ecosystem frontends, including
TensorFlow, XLA (used for JAX), and TFLite. It extends the instructions in
[Building Python Bindings](#building-python-bindings) above with additional
steps that are TensorFlow specific. There are various ways to achieve these
ends, but this section describes the canonical method that the core
developers recommend. Upon completing these steps, you will have access to
additional Python packages:

| Python Import                       | Description                                                                 |
|-------------------------------------|-----------------------------------------------------------------------------|
| `import iree.compiler.tools.tf`     | Tools for importing from [TensorFlow](https://www.tensorflow.org/)          |
| `import iree.compiler.tools.tflite` | Tools for importing from [TensorFlow Lite](https://www.tensorflow.org/lite) |
| `import iree.compiler.tools.xla`    | Tools for importing from [XLA](https://www.tensorflow.org/xla)              |

These tools packages are needed in order for the frontend specific, high-level
APIs under `import iree.compiler.tf`, `import iree.compiler.tflite`,
`import iree.compiler.xla`, and `import iree.jax` to be fully functional.

!!! Caution

    This section is under construction. Refer to the
    [source documentation](https://github.com/openxla/iree/tree/main/integrations/tensorflow#readme)
    for the latest building from source instructions.

???+ Note
    Due to the difficulties using Bazel and compiling TensorFlow, only
    compilation on Linux with clang is supported. Other OS's and compilers are
    "best effort" with patches to improve support welcome.
