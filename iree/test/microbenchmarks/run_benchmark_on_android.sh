#!/bin/bash

MLIR_FILE_PATH=${1}
ARGS=${@:2}

adb shell 'mkdir /data/local/tmp'
MLIR_FILE_NAME=$(basename $MLIR_FILE_PATH)
TARGET_VM_FILE=/tmp/$MLIR_FILE_NAME.fbvm
ANDROID_TARGET=aarch64-none-linux-android30
${IREE_RELEASE_DIR}/iree/tools/iree-translate --iree-hal-target-backends=dylib-llvm-aot --iree-mlir-to-vm-bytecode-module --iree-llvm-target-triple=${ANDROID_TARGET} ${ARGS} ${MLIR_FILE_PATH} -o ${TARGET_VM_FILE}
adb push ${TARGET_VM_FILE} '/data/local/tmp'
rm ${TARGET_VM_FILE}
adb shell </dev/null taskset 80 "data/local/tmp/iree-benchmark-module --driver=dylib  --dylib_worker_count=1 --module_file=/data/local/tmp/${MLIR_FILE_NAME}.fbvm"
adb shell "rm /data/local/tmp/${MLIR_FILE_NAME}.fbvm"