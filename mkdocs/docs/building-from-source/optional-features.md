# Optional Features

This page details the optional features and build modes for the project.
Most of these are controlled by various CMake options, sometimes requiring
extra setup or preparation. Each section extends the basic build steps
in the [getting started](../getting-started/) page.

## Building Python Bindings

This section describes how to build and interactively use built-from-source
Python bindings for the following packages:

| Python Import             | Description                                                                 |
|------------------------------|-----------------------------------------------------------------------------|
| `import iree.compiler`     | IREE's generic compiler tools and helpers                                   |
| `import iree.runtime`      | IREE's runtime, including CPU and GPU backends                              |

Also see [instructions for installing pre-built binaries](../../bindings/python/).

**Pre-requisites:**

* A relatively recent Python3 installation (we aim to support
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

???+ Setup
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
        python -m venv .venv

        # Activate the virtual environment (per shell).
        # Now the `python` command will resolve to your virtual environment
        # (even on systems where you typically use `python3`).
        source .venv/bin/activate

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

    When done, close your shell or run `deactivate`.

???+ Usage
    From the `iree-build` directory:

    === "Linux and MacOS"

        ``` shell
        cmake -DIREE_BUILD_PYTHON_BINDINGS=ON -DPython3_EXECUTABLE="$(which python)" .
        ninja

        # Add ./bindings/python to PYTHONPATH and use the API.
        export PYTHONPATH="$PWD/bindings/python"
        python -c "import iree.compiler"
        python -c "import iree.runtime"
        ```

    === "Windows"

        ``` powershell
        cmake -DIREE_BUILD_PYTHON_BINDINGS=ON .
        ninja

        # Add bindings\python to PYTHONPATH and use the API.
        set PYTHONPATH="$pwd\bindings\python;%PYTHONPATH%"
        python -c "import iree.compiler"
        python -c "import iree.runtime"
        ```

    Tests can now be run individually via python or via ctest.
