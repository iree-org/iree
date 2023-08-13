#!/bin/bash
# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# build_windows_packages.sh
# One stop build of IREE Python packages for Windows. This presumes that
# dependencies are installed from install_windows_deps.ps1.
#
# Valid packages:
#   iree-runtime
#   iree-compiler

set -eu -o errtrace

this_dir="$(cd $(dirname $0) && pwd)"
repo_root="$(cd $this_dir/../../ && pwd)"
python_versions="${override_python_versions:-3.11}"
output_dir="${output_dir:-${this_dir}/wheelhouse}"
packages="${packages:-iree-runtime iree-compiler}"

# Canonicalize paths.
mkdir -p "$output_dir"
output_dir="$(cd $output_dir && pwd)"

function run() {
  echo "Using python versions: ${python_versions}"

  local orig_path="$PATH"

  # Build phase.
  for package in $packages; do
    echo "******************** BUILDING PACKAGE ${package} ********************"
    for python_version in $python_versions; do
      if [[ $(py --list) != *${python_version}* ]]; then
        echo "ERROR: Could not find python version: ${python_version}"
        continue
      fi

      echo ":::: Version: $(py -${python_version} --version)"
      case "$package" in
        iree-runtime)
          clean_wheels iree_runtime $python_version
          build_iree_runtime $python_version
          ;;
        iree-compiler)
          clean_wheels iree_compiler $python_version
          build_iree_compiler $python_version
          ;;
        *)
          echo "Unrecognized package '$package'"
          exit 1
          ;;
      esac
    done
  done

  echo "******************** BUILD COMPLETE ********************"
  echo "Generated binaries:"
  ls -l $output_dir
}

function build_iree_runtime() {
  local python_version="$1"
  export IREE_RUNTIME_BUILD_TRACY=ON
  IREE_HAL_DRIVER_VULKAN=ON \
  py -${python_version} -m pip wheel -v -w $output_dir $repo_root/runtime/
}

function build_iree_compiler() {
  local python_version="$1"
  py -${python_version} -m pip wheel -v -w $output_dir $repo_root/compiler/
}

function clean_wheels() {
  local wheel_basename="$1"
  local python_version="$2"
  echo ":::: Clean wheels $wheel_basename $python_version"
  # python_version is something like "3.11", but we'd want something like "cp311".
  local cpython_version_string="cp${python_version%.*}${python_version#*.}"
  rm -f -v ${output_dir}/${wheel_basename}-*-${cpython_version_string}-*.whl
}

run
