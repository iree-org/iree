#!/bin/bash
# This script will download and setup the riscv-linux required toolchain and tool.

BOOTSTRAP_SCRIPT_PATH=$(dirname "$0")
BOOTSTRAP_WORK_DIR=$BOOTSTRAP_SCRIPT_PATH/.bootstrap

function cleanup {
  if [ -d $BOOTSTRAP_WORK_DIR ]; then
    rm -rf $BOOTSTRAP_WORK_DIR
  fi
}

clear
set -o pipefail

# Call the cleanup function when this tool exits.
trap cleanup EXIT

source $BOOTSTRAP_SCRIPT_PATH/riscv_setting.sh

execute () {
  eval $1
  if [ $? -ne 0 ]; then
    echo "command:\"$1\" error"
    exit 1
  fi
}

# $1: server name
# $2: file name
# $3: install path
# $4: tar_option
scp_method() {
  execute "scp $1:$2 $BOOTSTRAP_WORK_DIR/$(basename $2)"
  execute "tar $4 $BOOTSTRAP_WORK_DIR/$(basename $2) --no-same-owner --strip-components=1 -C $3"
}

# $1: file_id
# $2: file name
# $3: install path
# $4: tar_option
wget_google_drive() {
  execute "wget --save-cookies $BOOTSTRAP_WORK_DIR/cookies.txt \"https://docs.google.com/uc?export=download&id=\"$1 -O- | sed -En \"s/.*confirm=([0-9A-Za-z_]+).*/\1/p\" > $BOOTSTRAP_WORK_DIR/confirm.txt"
  execute "wget --progress=bar:force:noscroll --load-cookies $BOOTSTRAP_WORK_DIR/cookies.txt \"https://docs.google.com/uc?export=download&id=$1&confirm=`cat $BOOTSTRAP_WORK_DIR/confirm.txt`\" -O- | tar $4 - --no-same-owner --strip-components=1 -C $3"
}

# $1: server name or google drive file_id
# $2: file name
# $3: install path
# $4: download method(scp or wget_google_drive)
# (optional) $5: the post-processing for the file
download_file() {
  echo "Install $2 to $3"
  if [ -e $3/file_info.txt ]; then
    if [ ! -v WITH_CI_ENV ]; then
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
    else
      echo "Skip download $2."
      return
    fi
  fi

  if [ "${2##*.}" == "gz" ]; then
    tar_option="zxpf"
  elif [ "${2##*.}" == "bz2" ]; then
    tar_option="jxpf"
  fi
  echo "tar option: $tar_option"

  echo "Download $2 ..."
  execute "mkdir -p $3"
  $4 $1 $2 $3 $tar_option

  if [ $# -eq 5 ]; then
    $5
  fi

  echo "$1 $2" > $3/file_info.txt
}

execute "mkdir -p $BOOTSTRAP_WORK_DIR"

read -p "Install RISCV clang toolchain(y/n)? " answer
case ${answer:0:1} in
  y|Y )
    download_file $RISCV_CLANG_TOOLCHAIN_FILE_ID $RISCV_CLANG_TOOLCHAIN_FILE_NAME $TOOLCHAIN_PATH_PREFIX wget_google_drive
  ;;
  * )
    echo "Skip RISCV clang toolchain."
  ;;
esac

read -p "Install RISCV qemu(y/n)? " answer
case ${answer:0:1} in
  y|Y )
    download_file $QEMU_FILE_ID $QEMU_FILE_NAME $QEMU_PATH_PREFIX wget_google_drive
  ;;
  * )
    echo "Skip RISCV qemu."
  ;;
esac

echo "Bootstrap finished."
