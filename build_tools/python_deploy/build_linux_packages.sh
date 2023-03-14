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
set -xeu -o errtrace

this_dir="$(cd $(dirname $0) && pwd)"
script_name="$(basename $0)"
repo_root="$(cd "${this_dir}" && git rev-parse --show-toplevel)"
manylinux_docker_image="${manylinux_docker_image:-gcr.io/iree-oss/manylinux2014_x86_64-release@sha256:794513562cca263480c0c169c708eec9ff70abfe279d6dc44e115b04488b9ab5}"
python_versions="${override_python_versions:-cp37-cp37m cp38-cp38 cp39-cp39 cp310-cp310 cp311-cp311}"
output_dir="${output_dir:-${this_dir}/wheelhouse}"
packages="${packages:-iree-runtime iree-runtime-instrumented iree-compiler}"
package_suffix="${package_suffix:-}"

function run_on_host() {
  echo "Running on host"
  echo "Launching docker image ${manylinux_docker_image}"

  # Canonicalize paths.
  mkdir -p "${output_dir}"
  output_dir="$(cd "${output_dir}" && pwd)"
  echo "Outputting to ${output_dir}"
  mkdir -p "${output_dir}"
  docker run --rm \
    -v "${repo_root}:${repo_root}" \
    -v "${output_dir}:${output_dir}" \
    -e __MANYLINUX_BUILD_WHEELS_IN_DOCKER=1 \
    -e "override_python_versions=${python_versions}" \
    -e "packages=${packages}" \
    -e "package_suffix=${package_suffix}" \
    -e "output_dir=${output_dir}" \
    "${manylinux_docker_image}" \
    -- ${this_dir}/${script_name}

  echo "******************** BUILD COMPLETE ********************"
  echo "Generated binaries:"
  ls -l "${output_dir}"
}

function run_in_docker() {
  echo "Running in docker"
  echo "Using python versions: ${python_versions}"

  local orig_path="${PATH}"

  # Build phase.
  for package in ${packages}; do
    echo "******************** BUILDING PACKAGE ${package} ********************"
    for python_version in ${python_versions}; do
      python_dir="/opt/python/${python_version}"
      if ! [ -x "${python_dir}/bin/python" ]; then
        echo "ERROR: Could not find python: ${python_dir} (skipping)"
        continue
      fi
      export PATH="${python_dir}/bin:${orig_path}"
      echo ":::: Python version $(python --version)"
      # replace dashes with underscores
      package_suffix="${package_suffix//-/_}"
      case "${package}" in
        iree-runtime)
          clean_wheels "iree_runtime${package_suffix}" "${python_version}"
          build_iree_runtime
          run_audit_wheel "iree_runtime${package_suffix}" "${python_version}"
          ;;
        iree-runtime-instrumented)
          clean_wheels "iree_runtime_instrumented${package_suffix}" "${python_version}"
          build_iree_runtime_instrumented
          run_audit_wheel "iree_runtime_instrumented${package_suffix}" "${python_version}"
          ;;
        iree-compiler)
          clean_wheels "iree_compiler${package_suffix}" "${python_version}"
          build_iree_compiler
          run_audit_wheel "iree_compiler${package_suffix}" "${python_version}"
          ;;
        *)
          echo "Unrecognized package '${package}'"
          exit 1
          ;;
      esac
    done
  done
}

function build_wheel() {
  python -m pip wheel --disable-pip-version-check -v -w "${output_dir}" "${repo_root}/$@"
}

function build_iree_runtime() {
  IREE_HAL_DRIVER_CUDA=ON \
  build_wheel runtime/
}

function build_iree_runtime_instrumented() {
  IREE_HAL_DRIVER_CUDA=ON IREE_BUILD_TRACY=ON IREE_ENABLE_RUNTIME_TRACING=ON \
  IREE_RUNTIME_CUSTOM_PACKAGE_SUFFIX="-instrumented" \
  build_wheel runtime/
}

function build_iree_compiler() {
  IREE_TARGET_BACKEND_CUDA=ON \
  build_wheel compiler/
}

function run_audit_wheel() {
  local wheel_basename="$1"
  local python_version="$2"
  # Force wildcard expansion here
  generic_wheel="$(echo "${output_dir}/${wheel_basename}-"*"-${python_version}-linux_x86_64.whl")"
  ls "${generic_wheel}"
  echo ":::: Auditwheel ${generic_wheel}"
  auditwheel repair -w "${output_dir}" "${generic_wheel}"
  rm -v "${generic_wheel}"
}

function clean_wheels() {
  local wheel_basename="$1"
  local python_version="$2"
  echo ":::: Clean wheels ${wheel_basename} ${python_version}"
  rm -f -v "${output_dir}/${wheel_basename}-"*"-${python_version}-"*".whl"
}

# Trampoline to the docker container if running on the host.
if [ -z "${__MANYLINUX_BUILD_WHEELS_IN_DOCKER-}" ]; then
  run_on_host "$@"
else
  run_in_docker "$@"
fi
