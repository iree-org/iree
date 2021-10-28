# Optional features

This page details the optional features and build modes for the project.
Most of these are controlled by various CMake options, sometimes requiring
extra setup or preparation. Each section extends the basic build steps
in the [getting started](./getting-started.md) page.

## Building Python Bindings

This section describes how to build and interactively use built-from-source
Python bindings for the following packages:

| Python Import             | Description                                                                 |
|------------------------------|-----------------------------------------------------------------------------|
| `import iree.compiler`     | IREE's generic compiler tools and helpers                                   |
| `import iree.runtime`      | IREE's runtime, including CPU and GPU backends                              |

Also see [instructions for installing pre-built binaries](../bindings/python.md).

**Pre-requisites:**

* A relatively recent Python3 installation >=3.7 (we aim to support
  [non-eol Python versions](https://endoflife.date/python)).
* Installation of python dependencies as specified in
  [`bindings/python/build_requirements.txt`](https://github.com/google/iree/blob/main/bindings/python/build_requirements.txt).

**CMake Variables:**

* **`IREE_BUILD_PYTHON_BINDINGS`** : `BOOL`

    Enables building of Python bindings under `bindings/python` in the repository.
    Defaults to `OFF`.

* **`Python3_EXECUTABLE`** : `PATH`

    Full path to the Python3 executable to build against. If not specified, CMake
    will auto-detect this, which often yields incorrect results on systems
    with multiple Python versions. Explicitly setting this is recommended.
    Note that mixed case of the option.

### Setup

We recommend using virtual environments to manage python packages, such
as through `venv`, which may need to be installed via your system
package manager ([about](https://docs.python.org/3/library/venv.html),
[tutorial](https://docs.python.org/3/tutorial/venv.html)):

=== "Linux and MacOS"

    ``` shell
    # Make sure your 'python' is what you expect. Note that on multi-python
    # systems, this may have a version suffix, and on many Linuxes and MacOS where
    # python2 and python3 co-exist, you may also want to use `python3`.
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
    python -m pip install -r ./bindings/python/build_requirements.txt
    ```

=== "Windows"

    ``` powershell
    python -m venv .venv
    .venv\Scripts\activate.bat
    python -m pip install --upgrade pip
    python -m pip install -r bindings\python\build_requirements.txt
    ```

When you are done with the venv, you can close it by closing your shell
or running `deactivate`.

### Usage

From the `iree-build` directory:

=== "Linux and MacOS"

    ``` shell
    cmake \
        -GNinja \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DIREE_BUILD_PYTHON_BINDINGS=ON \
        -DPython3_EXECUTABLE="$(which python)" \
        .
    cmake --build .

    # Add ./bindings/python and compiler-api/python_package to PYTHONPATH and
    # use the API.
    source .env && export PYTHONPATH
    export PYTHONPATH="$PWD/bindings/python"
    python -c "import iree.compiler"
    python -c "import iree.runtime"
    ```

=== "Windows"

    ``` powershell
    cmake -GNinja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DIREE_BUILD_PYTHON_BINDINGS=ON .
    cmake --build .

    # Add bindings\python and compiler-api\python_package to PYTHONPATH and use
    # the API.
    set PYTHONPATH="$pwd\compiler-api\python_package;$pwd\bindings\python;%PYTHONPATH%"
    python -c "import iree.compiler"
    python -c "import iree.runtime"
    ```

Tests can now be run individually via python or via ctest.


## Building TensorFlow Frontend Bindings

This section describes how to build compiler tools and Python bindings for
importing models from various TensorFlow-ecosystem frontends, including
TensorFlow, XLA (used for JAX), and TFLite. It extends the instructions in
[Building Python Bindings](#building-python-bindings) above with additional
steps that are TensorFlow specific. There are various ways to achieve these
ends, but this section describes the canonical method that the core
developers recommend. Upon completing these steps, you will have access to
additional Python packages:

| Python Import             | Description                                                                 |
|------------------------------|-----------------------------------------------------------------------------|
| `import iree.compiler.tools.tf`     | Tools for importing from [TensorFlow](https://www.tensorflow.org/)          |
| `import iree.compiler.tools.tflite` | Tools for importing from [TensorFlow Lite](https://www.tensorflow.org/lite) |
| `import iree.compiler.tools.xla`    | Tools for importing from [XLA](https://www.tensorflow.org/xla)              |

These tools packages are needed in order for the frontend specific, high-level
APIs under `import iree.compiler.tf`, `import iree.compiler.tflite`,
`import iree.compiler.xla`, and `import iree.jax` to be fully functional.

### Setup

A relatively recent `tf-nightly` release is needed to run tests.

=== "Linux and MacOS"

    ``` shell
    python -m pip install -r ./integrations/tensorflow/bindings/python/build_requirements.txt
    ```

=== "Windows"

    ``` powershell
    python -m pip install -r integrations\tensorflow\bindings\python\build_requirements.txt
    ```

### TensorFlow

TensorFlow frontends can only be built with [Bazel](https://bazel.build/),
and this must be done as a manual step (we used to have automation for this,
but Bazel integrates poorly with automation and it made diagnosis and cross
platform usage unreliable). The recommended version of Bazel (used by CI
systems) can be found in the
[.bazelversion](https://github.com/google/iree/blob/main/.bazelversion)
file. In addition, Bazel is hard to use out of tree, so these steps will
involve working from the source tree (instead of the build tree).

???+ Note
    Due to the difficulties using Bazel and compiling TensorFlow, only
    compilation on Linux with clang is supported. Other OS's and compilers are
    "best effort" with patches to improve support welcome.

=== "Linux and MacOS"

    ``` shell
    # From the iree source directory.
    cd integrations/tensorflow
    CC=clang CXX=clang python ../../configure_bazel.py
    bazel build iree_tf_compiler:importer-binaries
    ```

=== "Windows"

    ``` powershell
    # From the iree source directory.
    cd integrations\tensorflow
    python ..\..\configure_bazel.py
    bazel build iree_tf_compiler:importer-binaries
    ```

Importer binaries can be found under `bazel-bin/iree_tf_compiler` and can
be used from the command line if desired.


???+ Note
    Bazel's default configuration tends to build almost everything twice,
    for reasons that can only be surmised to be based in some technically
    correct but practically challenged viewpoint. It is also incompatible with
    ccache and other mechanisms for performing less incremental work. It is
    recommended to address both of these with a `.bazelrc` file in your
    home directory:

    ```
    build --disk_cache=/path/to/home/.bazelcache
    build --nodistinct_host_configuration
    ```

    We can't set these for you because of other inscrutable reasons.

### IREE

The main IREE build will embed binaries built above and enable additional
Python APIs. Within the build, the binaries are symlinked, so can be
rebuilt per above without re-running these steps for edit-and-continue
style work.

``` shell
# From the iree-build/ directory.
cmake -DIREE_BUILD_TENSORFLOW_ALL=ON .
cmake --build .

# Validate.
python -c "import iree.tools.tf as _; print(_.get_tool('iree-import-tf'))"
python -c "import iree.tools.tflite as _; print(_.get_tool('iree-import-tflite'))"
python -c "import iree.tools.xla as _; print(_.get_tool('iree-import-xla'))"
```
