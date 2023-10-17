#!/bin/bash

# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This script will download and setup the riscv-linux required toolchain and tool.

set -e
set -o pipefail

PREBUILT_DIR="${HOME}/riscv"
IREE_ARTIFACT_URL="https://storage.googleapis.com/iree-shared-files"

read -p "Enter the riscv tools root path(press enter to use default path:${PREBUILT_DIR}): " INPUT_PATH
if [[ "${INPUT_PATH}" ]]; then
  PREBUILT_DIR=${INPUT_PATH}
fi
echo "The riscv tool prefix path: ${PREBUILT_DIR}"

BOOTSTRAP_WORK_DIR="${PREBUILT_DIR}/.bootstrap"

if [[ "${OSTYPE}" == "linux-gnu" ]]; then
  RISCV_CLANG_TOOLCHAIN_FILE_NAME="toolchain_iree_manylinux_2_28_20231012.tar.gz"
  RISCV_CLANG_TOOLCHAIN_FILE_SHA="3af56a58551ed5ae7441214822461a5368fee9403d7c883762fa902489bfbff0"

  QEMU_FILE_NAME="qemu-riscv.tar.gz"
  QEMU_FILE_SHA="6e0bca77408e606add8577d6f1b6709f6ef3165b0e241ed2ba191183dfc931ec"

  TOOLCHAIN_PATH_PREFIX=${PREBUILT_DIR}/toolchain/clang/linux/RISCV
  QEMU_PATH_PREFIX=${PREBUILT_DIR}/qemu/linux/RISCV
else
  echo "${OSTYPE} is not supported."
  exit 1
fi

function cleanup {
  if [[ -d "${BOOTSTRAP_WORK_DIR}" ]]; then
    rm -rf "${BOOTSTRAP_WORK_DIR}"
  fi
}

# Call the cleanup function when this tool exits.
trap cleanup EXIT

# Download and install the toolchain from IREE-OSS GCS
download_file() {
  local file_name="$1"
  local install_path="$2"
  local file_sha="$3"

  echo "Install $1 to $2"
  if [[ "$(ls -A $2)" ]]; then
    read -p "The file already exists. Keep it (y/n)? " replaced
    case ${replaced:0:1} in
      y|Y )
        echo "Skip download $1."
        return
      ;;
      * )
        rm -rf "$2"
      ;;
    esac
  fi

  echo "Download ${file_name} ..."
  mkdir -p $install_path
  wget --progress=bar:force:noscroll --directory-prefix="${BOOTSTRAP_WORK_DIR}" \
    "${IREE_ARTIFACT_URL}/${file_name}" && \
    echo "${file_sha} ${BOOTSTRAP_WORK_DIR}/${file_name}" | sha256sum -c -
  echo "Extract ${file_name} ..."
  tar -C "${install_path}" -xf "${BOOTSTRAP_WORK_DIR}/${file_name}" --no-same-owner \
    --strip-components=1
}

mkdir -p "${BOOTSTRAP_WORK_DIR}"

read -p "Install RISCV clang toolchain(y/n)? " answer
case ${answer:0:1} in
  y|Y )
    download_file "${RISCV_CLANG_TOOLCHAIN_FILE_NAME}" \
                  "${TOOLCHAIN_PATH_PREFIX}" \
                  "${RISCV_CLANG_TOOLCHAIN_FILE_SHA}"

    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo " PLEASE run 'export RISCV_TOOLCHAIN_ROOT=${TOOLCHAIN_PATH_PREFIX}'   "
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
  ;;
  * )
    echo "Skip RISCV clang toolchain."
  ;;
esac

read -p "Install RISCV qemu(y/n)? " answer
case ${answer:0:1} in
  y|Y )
    download_file "${QEMU_FILE_NAME}" \
                  "${QEMU_PATH_PREFIX}" \
                  "${QEMU_FILE_SHA}"
  ;;
  * )
    echo "Skip RISCV qemu."
  ;;
esac

echo "Bootstrap finished."
