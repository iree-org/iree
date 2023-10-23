# Debugging failures in TF/TFLite integration tests.

These are steps to reproduce/address failures in TF/TFLite integration tests.
These instructions are most stable on Linux, though they may work with a few
tweaks on Windows and macOS.

All steps here assume starting from the IREE root directory.

1. First create a Python virtual environment to install packages into:

    ```bash
    python -m venv iree-tf.venv
    source iree-tf.venv/bin/activate

    # Install test requirements
    python -m pip install -r ./integrations/tensorflow/test/requirements.txt
    ```

2. Install IREE's tools and Python bindings or build them from source

    Install distributed packages

    ```bash
    # Install packages from nightly releases
    # This should work for most cases, as the importers change infrequently
    python -m pip install \
      iree-compiler iree-runtime iree-tools-tf iree-tools-tflite \
      --find-links https://openxla.github.io/iree/pip-release-links.html
    ```

    _OR_ build from source

    ```bash
    # Build Python bindings from source
    cmake -G Ninja -B ../iree-build/ -DIREE_BUILD_PYTHON_BINDINGS=ON .
    cmake --build ../iree-build/

    # Add IREE built-from-source Python packages to PYTHONPATH
    source .env

    # Install IREE TF/TFLite Python packages
    python -m pip install integrations/tensorflow/python_projects/iree_tf
    python -m pip install integrations/tensorflow/python_projects/iree_tflite
    ```

3. Run the python test command line

    The command can be obtained from the run file. For example, if
    `iree_tfl_tests/llvmcpu_posenet_i8.run` failed,

    ```bash
    cd integrations/tensorflow/test/
    cat iree_tfl_tests/llvmcpu_posenet_i8.run

    # REQUIRES: llvmcpu
    # RUN: %PYTHON -m iree_tfl_tests.posenet_i8_test --target_backend=llvmcpu --artifacts_dir=%t

    cd python/
    python -m iree_tfl_tests.posenet_i8_test --target_backend=llvmcpu --artifacts_dir=/tmp/posenet_i8_failure
    ```

    Note that the command can only be run under
    `integrations/tensorflow/test/python` directory.

4. Extract intermediate files and use with native tools

    The test will create an `iree_input.mlir` in the temp directory specified.
    Those can then be fed into `iree-compile` (built locally to reproduce the
    error)

    ```bash
    iree-compile \
      --iree-hal-target-backends=llvm-cpu \
      --iree-input-type=stablehlo \
      iree_input.mlir
    ```
