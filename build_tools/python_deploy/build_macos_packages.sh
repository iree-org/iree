#!/bin/bash
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# build_macos_packages.sh
# One stop build of IREE Python packages for MacOS. This presumes that
# dependencies are installed from install_macos_deps.sh. This will build
# for a list of Python versions synchronized with that script and corresponding
# with directory names under:
#   /Library/Frameworks/Python.framework/Versions
#
# MacOS convention is to refer to this as major.minor (i.e. "3.9", "3.10").
# Valid packages:
#   iree-runtime
#   iree-compiler

set -eu -o errtrace

this_dir="$(cd $(dirname $0) && pwd)"
repo_root="$(cd $this_dir/../../ && pwd)"
python_versions="${override_python_versions:-3.11}"
output_dir="${output_dir:-${this_dir}/wheelhouse}"
packages="${packages:-iree-runtime iree-compiler}"

# Note that this typically is selected to match the version that the official
# Python distributed is built at.
export MACOSX_DEPLOYMENT_TARGET=13.0

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
      python_dir="/Library/Frameworks/Python.framework/Versions/$python_version"
      if ! [ -x "$python_dir/bin/python3" ]; then
        echo "ERROR: Could not find python3: $python_dir (skipping)"
        continue
      fi
      export PATH=$python_dir/bin:$orig_path
      echo ":::: Python version $(python3 --version)"
      case "$package" in
        iree-runtime)
          clean_wheels iree_runtime $python_version
          build_iree_runtime
          ;;
        iree-compiler)
          clean_wheels iree_compiler $python_version
          build_iree_compiler
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
  # TODO(antiagainst): remove Vulkan once IREE_HAL_DRIVER_METAL is stable
  export IREE_RUNTIME_BUILD_TRACY=ON
  IREE_HAL_DRIVER_VULKAN=ON \
  python3 -m pip wheel -v -w $output_dir $repo_root/runtime/
}

function build_iree_compiler() {
  python3 -m pip wheel -v -w $output_dir $repo_root/compiler/
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
