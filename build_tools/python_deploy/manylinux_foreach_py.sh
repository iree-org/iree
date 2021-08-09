#!/bin/bash
# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Script to do ad-hoc builds for all python versions in a dockcross manylinux
# container. Mostly, there are CI scripts and actions that do this, but they
# generally are not friendly for running directly, and sometimes you just
# want to be able to build a binary without balancing the world. This script
# is for such times.
#
# Typical usage:
#   ./build_tools/python_deploy/manylinux_foreach_py.sh \
#     ./llvm-projects/iree-compiler-api/build_tools/build_python_wheels.sh
set -eu -o errtrace

this_dir="$(cd $(dirname $0) && pwd)"
script_name="$(basename $0)"
dockcross_image="manylinux2014-x64"
python_versions="cp37-cp37m cp38-cp38 cp39-cp39"

function run_on_host() {
  echo "Running on host"
  "$this_dir/setup_dockcross.sh" $dockcross_image

  echo "Running in docker..."
  "$this_dir/$dockcross_image" \
    --args "-v $this_dir:/python_deploy -e __MANYLINUX_BUILD_WHEELS_IN_DOCKER=1 -e MANYLINUX_AUDITWHEEL_REPAIR=1" \
    -- bash /python_deploy/$script_name "$@"
}

function run_in_docker() {
  local script="$1"
  shift
  echo "Running in docker"
  local orig_path="$PATH"
  local script_path="/work/$script"
  for python_version in $python_versions; do
    python_dir="/opt/python/$python_version"
    if ! [ -x "$python_dir/bin/python" ]; then
      echo "ERROR: Could not find python: $python_dir (skipping)"
      continue
    fi
    export PATH=$python_dir/bin:$orig_path
    # CMake has trouble with manylinux versions of python, so set some env
    # vars.
    echo "Running $script_path $@"
    "$script_path" "$@"
  done
}

# Trampoline to the docker container if running on the host.
if [ -z "${__MANYLINUX_BUILD_WHEELS_IN_DOCKER-}" ]; then
  run_on_host "$@"
else
  run_in_docker "$@"
fi
