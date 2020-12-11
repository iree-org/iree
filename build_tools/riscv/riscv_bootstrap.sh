#!/bin/bash

# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script will download and setup the riscv-linux required toolchain and tool.

BOOTSTRAP_SCRIPT_PATH=$(dirname "$0")
BOOTSTRAP_WORK_DIR=${BOOTSTRAP_SCRIPT_PATH}/.bootstrap

PREBUILT_DIR=${HOME}/riscv

read -p "Enter the riscv tools root path(press enter to use default path:${PREBUILT_DIR}): " INPUT_PATH
if [[ "${INPUT_PATH}" ]]; then
  PREBUILT_DIR=${INPUT_PATH}
fi
echo "The riscv tool prefix path: ${PREBUILT_DIR}"

if [[ "${OSTYPE}" == "linux-gnu" ]]; then
  RISCV_CLANG_TOOLCHAIN_FILE_ID=1d5AIeSTTOSUTFs6XPEhdcY9THgjtXtjQ
  RISCV_CLANG_TOOLCHAIN_FILE_NAME=riscv-llvm-toolchain.tar.bz2
  QEMU_FILE_ID=1gU4ycMrKtm3nrJ8Kmg4TSTJQefJbQQDQ
  QEMU_FILE_NAME=riscv-qemu-e5994807-linux-ubuntu.tar.gz

  TOOLCHAIN_PATH_PREFIX=${PREBUILT_DIR}/toolchain/clang/linux/RISCV
  QEMU_PATH_PREFIX=${PREBUILT_DIR}/qemu/linux/RISCV
elif [[ "${OSTYPE}" == "darwin"* ]]; then
  RISCV_CLANG_TOOLCHAIN_FILE_ID=empty
  RISCV_CLANG_TOOLCHAIN_FILE_NAME=empty
  QEMU_FILE_ID=empty
  QEMU_FILE_NAME=empty

  TOOLCHAIN_PATH_PREFIX=${PREBUILT_DIR}/toolchain/clang/darwin/RISCV
  QEMU_PATH_PREFIX=${PREBUILT_DIR}/qemu/darwin/RISCV

  echo "We haven't had the darwin prebuilt binary yet. Skip this script."
  exit 1
else
  echo "${OSTYPE} is not supported."
  exit 1
fi

function cleanup {
  if [[ -d ${BOOTSTRAP_WORK_DIR} ]]; then
    rm -rf ${BOOTSTRAP_WORK_DIR}
  fi
}

set -o pipefail

# Call the cleanup function when this tool exits.
trap cleanup EXIT

execute () {
  eval $1
  if [[ $? -ne 0 ]]; then
    echo "command:\"$1\" error"
    exit 1
  fi
}

# $1: file_id
# $2: file name
# $3: install path
# $4: tar_option
wget_google_drive() {
  execute "wget --save-cookies ${BOOTSTRAP_WORK_DIR}/cookies.txt \"https://docs.google.com/uc?export=download&id=\"$1 -O- | sed -En \"s/.*confirm=([0-9A-Za-z_]+).*/\1/p\" > ${BOOTSTRAP_WORK_DIR}/confirm.txt"
  execute "wget --progress=bar:force:noscroll --load-cookies ${BOOTSTRAP_WORK_DIR}/cookies.txt \"https://docs.google.com/uc?export=download&id=$1&confirm=`cat ${BOOTSTRAP_WORK_DIR}/confirm.txt`\" -O- | tar $4 - --no-same-owner --strip-components=1 -C $3"
}

# $1: server name or google drive file_id
# $2: file name
# $3: install path
# $4: download method(scp or wget_google_drive)
# (optional) $5: the post-processing for the file
download_file() {
  echo "Install $2 to $3"
  if [[ -e $3/file_info.txt ]]; then
    read -p "The file already exists. Keep it (y/n)? " replaced
    case ${replaced:0:1} in
      y|Y )
        echo "Skip download $2."
        return
      ;;
      * )
        rm -rf $3
      ;;
    esac
  fi

  if [[ "${2##*.}" == "gz" ]]; then
    tar_option="zxpf"
  elif [[ "${2##*.}" == "bz2" ]]; then
    tar_option="jxpf"
  fi
  echo "tar option: $tar_option"

  echo "Download $2 ..."
  execute "mkdir -p $3"
  $4 $1 $2 $3 $tar_option

  if [[ $# -eq 5 ]]; then
    $5
  fi

  echo "$1 $2" > $3/file_info.txt
}

execute "mkdir -p ${BOOTSTRAP_WORK_DIR}"

read -p "Install RISCV clang toolchain(y/n)? " answer
case ${answer:0:1} in
  y|Y )
    download_file ${RISCV_CLANG_TOOLCHAIN_FILE_ID} ${RISCV_CLANG_TOOLCHAIN_FILE_NAME} ${TOOLCHAIN_PATH_PREFIX} wget_google_drive
  ;;
  * )
    echo "Skip RISCV clang toolchain."
  ;;
esac

read -p "Install RISCV qemu(y/n)? " answer
case ${answer:0:1} in
  y|Y )
    download_file $QEMU_FILE_ID ${QEMU_FILE_NAME} ${QEMU_PATH_PREFIX} wget_google_drive
  ;;
  * )
    echo "Skip RISCV qemu."
  ;;
esac

echo "Bootstrap finished."
