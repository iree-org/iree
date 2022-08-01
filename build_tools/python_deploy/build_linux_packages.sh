#!/bin/bash
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# build_linux_packages.sh
# One stop build of IREE Python packages for Linux. The Linux build is
# complicated because it has to be done via a docker container that has
# an LTS glibc version, all Python packages and other deps.
# This script handles all of those details.
#
# Usage:
# Build everything (all packages, all python versions):
#   ./build_tools/python_deploy/build_linux_packages.sh
#
# Build specific Python versions and packages to custom directory:
#   override_python_versions="cp38-cp38 cp39-cp39" \
#   packages="iree-runtime iree-runtime-instrumented" \
#   output_dir="/tmp/wheelhouse" \
#   ./build_tools/python_deploy/build_linux_packages.sh
#
# Valid Python versions match a subdirectory under /opt/python in the docker
# image. Typically:
#   cp37-cp37m cp38-cp38 cp39-cp39 cp310-cp310
#
# Valid packages:
#   iree-runtime
#   iree-runtime-instrumented
#   iree-compiler
#
# Note that this script is meant to be run on CI and it will pollute both the
# output directory and in-tree build/ directories (under runtime/ and
# compiler/) with docker created, root owned builds. Sorry - there is
# no good way around it.
#
# It can be run on a workstation but recommend using a git worktree dedicated
# to packaging to avoid stomping on development artifacts.
set -eu -o errtrace

this_dir="$(cd $(dirname $0) && pwd)"
script_name="$(basename $0)"
repo_root="$(cd $this_dir/../../ && pwd)"
script_name="$(basename $0)"
manylinux_docker_image="${manylinux_docker_image:-gcr.io/iree-oss/manylinux2014_x86_64-release@sha256:b09c10868f846308bad2eab253a77d0a3f097816c40342bc289d8e62509bc5f9}"
python_versions="${override_python_versions:-cp37-cp37m cp38-cp38 cp39-cp39 cp310-cp310}"
output_dir="${output_dir:-${this_dir}/wheelhouse}"
packages="${packages:-iree-runtime iree-runtime-instrumented iree-compiler}"

function run_on_host() {
  echo "Running on host"
  echo "Launching docker image ${manylinux_docker_image}"

  # Canonicalize paths.
  mkdir -p "$output_dir"
  output_dir="$(cd $output_dir && pwd)"
  echo "Outputting to ${output_dir}"
  mkdir -p "${output_dir}"
  docker run --rm \
    -v "${repo_root}:/main_checkout/iree" \
    -v "${output_dir}:/wheelhouse" \
    -e __MANYLINUX_BUILD_WHEELS_IN_DOCKER=1 \
    -e "override_python_versions=${python_versions}" \
    -e "packages=${packages}" \
    ${manylinux_docker_image} \
    -- bash /main_checkout/iree/build_tools/python_deploy/build_linux_packages.sh

  echo "******************** BUILD COMPLETE ********************"
  echo "Generated binaries:"
  ls -l $output_dir
}

function run_in_docker() {
  echo "Running in docker"
  echo "Using python versions: ${python_versions}"

  local orig_path="$PATH"

  # Build phase.
  for package in $packages; do
    echo "******************** BUILDING PACKAGE ${package} ********************"
    for python_version in $python_versions; do
      python_dir="/opt/python/$python_version"
      if ! [ -x "$python_dir/bin/python" ]; then
        echo "ERROR: Could not find python: $python_dir (skipping)"
        continue
      fi
      export PATH=$python_dir/bin:$orig_path
      echo ":::: Python version $(python --version)"
      case "$package" in
        iree-runtime)
          clean_wheels iree_runtime $python_version
          build_iree_runtime
          run_audit_wheel iree_runtime $python_version
          ;;
        iree-runtime-instrumented)
          clean_wheels iree_runtime_instrumented $python_version
          build_iree_runtime_instrumented
          run_audit_wheel iree_runtime_instrumented $python_version
          ;;
        iree-compiler)
          clean_wheels iree_compiler $python_version
          build_iree_compiler
          run_audit_wheel iree_compiler $python_version
          ;;
        *)
          echo "Unrecognized package '$package'"
          exit 1
          ;;
      esac
    done
  done
}

function build_iree_runtime() {
  IREE_HAL_DRIVER_CUDA=ON \
  python -m pip wheel -v -w /wheelhouse /main_checkout/iree/runtime/
}

function build_iree_runtime_instrumented() {
  IREE_HAL_DRIVER_CUDA=ON IREE_BUILD_TRACY=ON IREE_ENABLE_RUNTIME_TRACING=ON \
  IREE_RUNTIME_CUSTOM_PACKAGE_SUFFIX="-instrumented" \
  python -m pip wheel -v -w /wheelhouse /main_checkout/iree/runtime/
}

function build_iree_compiler() {
  IREE_TARGET_BACKEND_CUDA=ON \
  python -m pip wheel -v -w /wheelhouse /main_checkout/iree/compiler/
}

function run_audit_wheel() {
  local wheel_basename="$1"
  local python_version="$2"
  generic_wheel="/wheelhouse/${wheel_basename}-*-${python_version}-linux_x86_64.whl"
  echo ":::: Auditwheel $generic_wheel"
  auditwheel repair -w /wheelhouse $generic_wheel
  rm -v $generic_wheel
}

function clean_wheels() {
  local wheel_basename="$1"
  local python_version="$2"
  echo ":::: Clean wheels $wheel_basename $python_version"
  rm -f -v /wheelhouse/${wheel_basename}-*-${python_version}-*.whl
}

# Trampoline to the docker container if running on the host.
if [ -z "${__MANYLINUX_BUILD_WHEELS_IN_DOCKER-}" ]; then
  run_on_host "$@"
else
  run_in_docker "$@"
fi
