# Debugging failures in TF/TFLite integration tests.

These are steps to reproduce/address failures in TF/TFLite integration tests. All steps here
assume starting from the IREE root directory.

1. First setup the python environment as described [here](https://google.github.io/iree/building-from-source/python-bindings-and-importers/#environment-setup).

```
python -m venv iree.venv
source iree.venv/bin/activate
```

2. Install latest IREE release binaries. The importers are not expected to change much, so using the release binaries should work for most cases

```
python -m pip install iree-compiler iree-runtime iree-tools-tf iree-tools-tflite -f https://github.com/google/iree/releases/latest
```

3. Install TF nightly

```
python -m pip install tf-nightly Pillow
```

4. Run the python test command line. The command can be obtained from the run file. For example, if iree_tfl_tests/llvmaot_posenet_i8.run failed,

```
cd integrations/tensorflow/test/
cat iree_tfl_tests/llvmaot_posenet_i8.run

# REQUIRES: llvmaot
# RUN: %PYTHON -m iree_tfl_tests.posenet_i8_test --target_backend=llvmaot -artifacts_dir=%t

python -m iree_tfl_tests.posenet_i8_test -- target_backend=llvmaot -artifacts_dir=/tmp/posenet_i8_failure
```

5. This will create an `iree_input.mlir` in the temp directory specified. Those can then be fed into `iree-translate` (built locally to reproduce the error)

```
iree-translate -iree-mlir-to-vm-bytecode-module -iree-hal-target-backends=dylib-llvm-aot -iree-input-type=mhlo iree_input.mlir
```

