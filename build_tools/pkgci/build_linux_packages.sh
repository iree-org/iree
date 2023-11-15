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
#   override_python_versions="cp39-cp39 cp310-cp310" \
#   packages="iree-runtime" \
#   output_dir="/tmp/wheelhouse" \
#   ./build_tools/python_deploy/build_linux_packages.sh
#
# Valid Python versions match a subdirectory under /opt/python in the docker
# image. Typically:
#   cp39-cp39 cp310-cp310
#
# Valid packages:
#   iree-runtime
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

# Function to find the directory the ".git" directory is in.
# We do this instead of using git directly because `git` may complain about
# operating in a directory owned by another user.
function find_git_dir_parent() {
  curr_dir="${PWD}"

  # Loop until we reach the root directory
  while [ "${curr_dir}" != "/" ]; do
    # Check if there is a ".git" directory in the current directory
    if [ -d "${curr_dir}/.git" ]; then
      # Return the path to the directory containing the ".git" directory
      echo "${curr_dir}"
      return
    fi

    # Move up one directory
    curr_dir="$(dirname "${curr_dir}")"
  done

  # If we reach the root directory and there is no ".git" directory, return an empty string
  echo ""
}

this_dir="$(cd $(dirname $0) && pwd)"
script_name="$(basename $0)"
repo_root=$(cd "${this_dir}" && find_git_dir_parent)
manylinux_docker_image="${manylinux_docker_image:-$(uname -m | awk '{print ($1 == "aarch64") ? "quay.io/pypa/manylinux_2_28_aarch64" : "ghcr.io/nod-ai/manylinux_x86_64:main" }')}"
python_versions="${override_python_versions:-cp311-cp311}"
output_dir="${output_dir:-${this_dir}/wheelhouse}"
cache_dir="${cache_dir:-}"
packages="${packages:-iree-runtime iree-compiler}"
package_suffix="${package_suffix:-}"
toolchain_suffix="${toolchain_suffix:-release}"

function run_on_host() {
  echo "Running on host"
  echo "Launching docker image ${manylinux_docker_image}"

  # Canonicalize paths.
  mkdir -p "${output_dir}"
  output_dir="$(cd "${output_dir}" && pwd)"
  echo "Outputting to ${output_dir}"
  extra_args=""
  if ! [ -z "$cache_dir" ]; then
    echo "Setting up host cache dir ${cache_dir}"
    mkdir -p "${cache_dir}/ccache"
    mkdir -p "${cache_dir}/pip"
    extra_args="${extra_args} -v ${cache_dir}:${cache_dir} -e cache_dir=${cache_dir}"
  fi
  docker run --rm \
    -v "${repo_root}:${repo_root}" \
    -v "${output_dir}:${output_dir}" \
    -e __MANYLINUX_BUILD_WHEELS_IN_DOCKER=1 \
    -e "override_python_versions=${python_versions}" \
    -e "packages=${packages}" \
    -e "package_suffix=${package_suffix}" \
    -e "output_dir=${output_dir}" \
    -e "toolchain_suffix=${toolchain_suffix}" \
    ${extra_args} \
    "${manylinux_docker_image}" \
    -- "${this_dir}/${script_name}"
}

function run_in_docker() {
  echo "Running in docker"
  echo "Marking git safe.directory"
  git config --global --add safe.directory '*'

  echo "Using python versions: ${python_versions}"
  local orig_path="${PATH}"

  # Configure toolchain.
  export CMAKE_TOOLCHAIN_FILE="${this_dir}/linux_toolchain_${toolchain_suffix}.cmake"
  echo "Using CMake toolchain ${CMAKE_TOOLCHAIN_FILE}"
  if ! [ -f "$CMAKE_TOOLCHAIN_FILE" ]; then
    echo "CMake toolchain not found (wrong toolchain_suffix?)"
    exit 1
  fi

  # Configure caching.
  if [ -z "$cache_dir" ]; then
    echo "Cache directory not configured. No caching will take place."
  else
    mkdir -p "${cache_dir}"
    cache_dir="$(cd ${cache_dir} && pwd)"
    echo "Caching build artifacts to ${cache_dir}"
    export CCACHE_DIR="${cache_dir}/ccache"
    export CCACHE_MAXSIZE="2G"
    export CMAKE_C_COMPILER_LAUNCHER=ccache
    export CMAKE_CXX_COMPILER_LAUNCHER=ccache
    # Configure pip cache dir.
    # We make it two levels down from within the container because pip likes
    # to know that it is owned by the current user.
    export PIP_CACHE_DIR="${cache_dir}/pip/in/container"
    mkdir -p "${PIP_CACHE_DIR}"
    chown -R "$(whoami)" "${cache_dir}/pip"
  fi

  # Build phase.
  set -o xtrace
  install_native_deps
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
      prepare_python
      # replace dashes with underscores
      package_suffix="${package_suffix//-/_}"
      case "${package}" in
        iree-runtime)
          clean_wheels "iree_runtime${package_suffix}" "${python_version}"
          build_iree_runtime
          run_audit_wheel "iree_runtime${package_suffix}" "${python_version}"
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

  set +o xtrace
  echo "******************** BUILD COMPLETE ********************"
  echo "Generated binaries:"
  ls -l "${output_dir}"
  if ! [ -z "$cache_dir" ]; then
    echo "ccache stats:"
    ccache --show-stats
  fi
}

function build_wheel() {
  python -m pip wheel --disable-pip-version-check -v -w "${output_dir}" "${repo_root}/$@"
}

function build_iree_runtime() {
  echo "::group::build_iree_runtime"
  # We install the needed build deps below for the tools.
  IREE_RUNTIME_BUILD_TRACY=ON IREE_RUNTIME_BUILD_TRACY_TOOLS=ON \
  IREE_EXTERNAL_HAL_DRIVERS="rocm" \
  build_wheel runtime/
  echo "::endgroup::"
}

function build_iree_compiler() {
  echo "::group::build_iree_compiler"
  IREE_TARGET_BACKEND_ROCM=ON IREE_ENABLE_LLD=ON \
  build_wheel compiler/
  echo "::endgroup::"
}

function run_audit_wheel() {
  local wheel_basename="$1"
  local python_version="$2"
  # Force wildcard expansion here
  generic_wheel="$(echo "${output_dir}/${wheel_basename}-"*"-${python_version}-linux_$(uname -m).whl")"
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

function prepare_python() {
  # The 0.17 series of patchelf can randomly corrupt executables. Fixes
  # have landed but not yet been released. Consider removing this pin
  # once 0.19 is released. We just override the system version with
  # a pip side load.
  pip install patchelf==0.16.1.0
  hash -r
  echo "patchelf version: $(patchelf --version) (0.17 is bad: https://github.com/NixOS/patchelf/issues/446)"
}

function install_native_deps() {
  echo ":::: Install Native Deps"

  # Get the output of uname -m
  uname_m=$(uname -m)

  # Check if the output is aarch64

  if [[ "$uname_m" == "aarch64" ]]; then
    echo "The architecture is aarch64 and we use manylinux 2_28 so install deps"
    yum install -y epel-release
    yum update -y
    # Required for Tracy
    yum install -y capstone-devel tbb-devel libzstd-devel
    yum install -y clang lld
  elif [[ "$uname_m" == "x86_64" ]]; then
    # Check if the output is x86_64
    echo "Running on an architecture which has deps in docker image."
  else
    echo "The architecture is unknown. Exiting"
    exit 1
  fi
}


# Trampoline to the docker container if running on the host.
if [ -z "${__MANYLINUX_BUILD_WHEELS_IN_DOCKER-}" ]; then
  run_on_host "$@"
else
  run_in_docker "$@"
fi
