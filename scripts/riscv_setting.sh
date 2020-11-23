#!/bin/bash

SETTING_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RISCV_LINUX_WORK_DIR=$SETTING_SCRIPT_DIR/..

PREBUILT_DIR=$RISCV_LINUX_WORK_DIR/Prebuilt
BUILD_DIR=$RISCV_LINUX_WORK_DIR/build/

if [[ "$OSTYPE" == "linux-gnu" ]]; then
  RISCV_CLANG_TOOLCHAIN_FILE_ID=1d5AIeSTTOSUTFs6XPEhdcY9THgjtXtjQ
  RISCV_CLANG_TOOLCHAIN_FILE_NAME=riscv-llvm-toolchain.tar.bz2
  QEMU_FILE_ID=1gU4ycMrKtm3nrJ8Kmg4TSTJQefJbQQDQ
  QEMU_FILE_NAME=riscv-qemu-e5994807-linux-ubuntu.tar.gz

  TOOLCHAIN_PATH_PREFIX=$PREBUILT_DIR/toolchain/clang/linux/RISCV
  QEMU_PATH_PREFIX=$PREBUILT_DIR/qemu/linux/RISCV
elif [[ "$OSTYPE" == "darwin"* ]]; then
  RISCV_CLANG_TOOLCHAIN_FILE_ID=empty
  RISCV_CLANG_TOOLCHAIN_FILE_NAME=empty
  QEMU_FILE_ID=empty
  QEMU_FILE_NAME=empty

  TOOLCHAIN_PATH_PREFIX=$PREBUILT_DIR/toolchain/clang/darwin/RISCV
  QEMU_PATH_PREFIX=$PREBUILT_DIR/qemu/darwin/RISCV
else
  echo "$OSTYPE is not supported"
  return 1
fi
