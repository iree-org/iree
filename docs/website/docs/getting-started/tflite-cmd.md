# TFLite via Command Line

IREE's tooling is divided into two components: import and compilation.

1. The import tool converts the TFLite flatbuffer to an IREE compatible form,
validating that only IREE compatible operations remain. Containing a combination of TOSA
and IREE operations.
2. The compilation stage generates the bytecode module for a list of targets, which can
be executed by IREE.

These two stages can be completed entirely via the command line.

```shell
WORKDIR="/tmp/workdir"
TFLITE_URL="https://storage.googleapis.com/iree-model-artifacts/tflite-integration-tests/posenet_i8.tflite"
TFLITE_PATH=${WORKDIR}/model.tflite
IMPORT_PATH=${WORKDIR}/tosa.mlir
MODULE_PATH=${WORKDIR}/module.vmfb

# Fetch the sample model
wget ${TFLITE_URL} -O ${TFLITE_PATH}

# Import the sample model to an IREE compatible form
iree-import-tflite ${TFLITE_PATH} -o ${IMPORT_PATH}

# Compile for the CPU backend
iree-translate \
    --iree-mlir-to-vm-bytecode-module \
    --iree-input-type=tosa \
    --iree-hal-target-backends=dylib-llvm-aot \
    ${IMPORT_PATH} \
    -o ${MODULE_PATH}
```
