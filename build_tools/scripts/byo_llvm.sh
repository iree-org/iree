#!/usr/bin/env bash
# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This script illustrates how IREE can be built with various forms of LLVM
# installations with the option -DIREE_BUILD_BUNDLED_LLVM=OFF set, disabling
# a source dependency on the bundled LLVM submodule.
#
# Usage:
#   byo_llvm.sh build_llvm && \
#   byo_llvm.sh build_mlir && \
#   byo_llvm.sh build_iree
#
# This script has minimal configurability, which can be extended as needed. The
# defaults should suffice for testing on CI. Different configurations are
# possible (i.e. building MLIR bundled with LLVM vs standalone), and this
# is just normal CMake package management options.
#
# Fully separating LLVM+LLD+CLANG from MLIR and from IREE enables maximum
# flexibility for the cases where multiple teams are responsible for
# different parts. Note that IREE often has a tight dependency on specific
# MLIR commits, and the bundled submodule often carries patches and fixes
# required for full functionality of all backends.

TD="$(cd $(dirname $0) && pwd)"
REPO_ROOT="$(cd $TD/../.. && pwd)"
LLVM_SOURCE_DIR="${LLVM_SOURCE_DIR:-${REPO_ROOT}/third_party/llvm-project}"
LLVM_BUILD_DIR="${LLVM_BUILD_DIR:-${REPO_ROOT}/../iree-llvm-build}"
LLVM_INSTALL_DIR="${LLVM_INSTALL_DIR:-${REPO_ROOT}/../iree-llvm-install}"

command="$1"
shift

# Detect commands.
has_ccache=false
if (command -v ccache &> /dev/null); then
  has_ccache=true
fi
has_clang=false
if (command -v clang &> /dev/null) && (command -v clang++ &> /dev/null); then
  has_clang=true
fi
has_lld=false
if (command -v lld &> /dev/null); then
  has_lld=true
fi
python3_command=""
if (command -v python3 &> /dev/null); then
  python3_command="python3"
elif (command -v python &> /dev/null); then
  python3_command="python"
fi

set -euo pipefail

print_toolchain_config() {
  if $has_ccache; then
    echo -n "-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache "
  fi
  if $has_clang; then
    echo "-DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++"
  fi
}

do_build_llvm() {
  echo "*********************** BUILDING LLVM *********************************"
  main_build_dir="${LLVM_BUILD_DIR}/llvm"
  main_install_dir="${LLVM_INSTALL_DIR}/llvm"
  targets_to_build="${LLVM_TARGETS_TO_BUILD:-X86}"
  enable_projects="${LLVM_ENABLE_TARGETS:-clang;lld}"

  cmake_options="-DLLVM_ENABLE_PROJECTS='${enable_projects}' -DLLVM_TARGETS_TO_BUILD='${targets_to_build}'"
  cmake_options="${cmake_options} -DLLVM_BUILD_EXAMPLES=OFF"
  cmake_options="${cmake_options} -DLLVM_INSTALL_UTILS=ON"
  cmake_options="${cmake_options} -DCMAKE_BUILD_TYPE=Release"
  cmake_options="${cmake_options} -DLLVM_ENABLE_ASSERTIONS=ON"
  cmake_options="${cmake_options} $(print_toolchain_config)"
  if $has_lld; then
    cmake_options="${cmake_options} -DLLVM_ENABLE_LLD=ON"
  fi

  echo "Source Directory: ${LLVM_SOURCE_DIR}"
  echo "Build Directory: ${main_build_dir}"
  echo "CMake Options: ${cmake_options}"
  cmake -GNinja -S "${LLVM_SOURCE_DIR}/llvm" -B "${main_build_dir}" \
    $cmake_options
  cmake --build "${main_build_dir}"
  cmake -DCMAKE_INSTALL_PREFIX="${main_install_dir}" -P "${main_build_dir}/cmake_install.cmake"
}

do_build_mlir() {
  echo "*********************** BUILDING MLIR *********************************"
  main_install_dir="${LLVM_INSTALL_DIR}/llvm"
  mlir_build_dir="${LLVM_BUILD_DIR}/mlir"
  mlir_install_dir="${LLVM_INSTALL_DIR}/mlir"

  cmake_options="-DLLVM_DIR='${main_install_dir}/lib/cmake/llvm'"
  cmake_options="${cmake_options} -DPython3_EXECUTABLE='$(which $python3_command)'"
  cmake_options="${cmake_options} -DLLVM_INSTALL_TOOLCHAIN_ONLY=OFF"
  cmake_options="${cmake_options} -DLLVM_BUILD_TOOLS=ON"
  cmake_options="${cmake_options} -DLLVM_ENABLE_ASSERTIONS=ON"
  cmake_options="${cmake_options} -DMLIR_ENABLE_BINDINGS_PYTHON=ON"
  cmake_options="${cmake_options} $(print_toolchain_config)"
  if $has_lld; then
    cmake_options="${cmake_options} -DLLVM_ENABLE_LLD=ON"
  fi

  echo "Source Directory: ${LLVM_SOURCE_DIR}"
  echo "Build Directory: ${mlir_build_dir}"
  echo "CMake Options: ${cmake_options}"
  cmake -GNinja -S "${LLVM_SOURCE_DIR}/mlir" -B "${mlir_build_dir}" \
    $cmake_options
  cmake --build "${mlir_build_dir}"
  cmake -DCMAKE_INSTALL_PREFIX="${mlir_install_dir}" -P "${mlir_build_dir}/cmake_install.cmake"
}

print_iree_config() {
  llvm_cmake_dir="${LLVM_INSTALL_DIR}/llvm/lib/cmake/llvm"
  lld_cmake_dir="${LLVM_INSTALL_DIR}/llvm/lib/cmake/lld"
  clang_cmake_dir="${LLVM_INSTALL_DIR}/llvm/lib/cmake/clang"
  # TODO: There seem to be utility exports missing from installed MLIR,
  # so using the build tree for now. This isn't great but needs fixing
  # upstream.
  #mlir_cmake_dir="${LLVM_INSTALL_DIR}/mlir/lib/cmake/mlir"
  mlir_cmake_dir="${LLVM_BUILD_DIR}/mlir/lib/cmake/mlir"

  if ! [ -d "$llvm_cmake_dir" ]; then
    echo "WARNING: CMake dir does not exist ($llvm_cmake_dir)" >&2
    return 1
  fi
  if ! [ -d "$lld_cmake_dir" ]; then
    echo "WARNING: CMake dir does not exist ($lld_cmake_dir)" >&2
    return 1
  fi
  if ! [ -d "$clang_cmake_dir" ]; then
    echo "WARNING: CMake dir does not exist ($clang_cmake_dir)" >&2
    return 1
  fi
  if ! [ -d "$mlir_cmake_dir" ]; then
    echo "WARNING: CMake dir does not exist ($mlir_cmake_dir)" >&2
    return 1
  fi

  echo "-DLLVM_DIR='$llvm_cmake_dir' -DLLD_DIR='$lld_cmake_dir' -DCLANG_DIR='$clang_cmake_dir' -DMLIR_DIR='$mlir_cmake_dir' -DIREE_BUILD_BUNDLED_LLVM=OFF"
}

do_build_iree() {
  echo "*********************** BUILDING IREE *********************************"
  iree_build_dir="${LLVM_BUILD_DIR}/iree"
  iree_install_dir="${LLVM_INSTALL_DIR}/iree"

  cmake_options="$(print_iree_config)"
  cmake_options="${cmake_options} -DPython3_EXECUTABLE='$(which $python3_command)'"
  cmake_options="${cmake_options} -DIREE_BUILD_PYTHON_BINDINGS=ON"
  cmake_options="${cmake_options} -DIREE_TARGET_BACKEND_DEFAULTS=OFF"
  cmake_options="${cmake_options} -DIREE_HAL_DRIVER_DEFAULTS=OFF"
  cmake_options="${cmake_options} -DIREE_HAL_DRIVER_LOCAL_SYNC=ON"
  cmake_options="${cmake_options} -DIREE_HAL_DRIVER_LOCAL_TASK=ON"
  cmake_options="${cmake_options} -DCMAKE_BUILD_TYPE=Release"
  cmake_options="${cmake_options} -DIREE_ENABLE_ASSERTIONS=ON"
  cmake_options="${cmake_options} $(print_toolchain_config)"
  if $has_lld; then
    cmake_options="${cmake_options} -DIREE_ENABLE_LLD=ON"
  fi

  echo "Source Directory: ${REPO_ROOT}"
  echo "Build Directory: ${iree_build_dir}"
  echo "CMake Options: ${cmake_options}"
  cmake -GNinja -S "${REPO_ROOT}" -B "${iree_build_dir}" \
    $cmake_options
  cmake --build "${iree_build_dir}"
  cmake -DCMAKE_INSTALL_PREFIX="${iree_install_dir}" -P "${iree_build_dir}/cmake_install.cmake"
}

case "${command}" in
  build_llvm)
    do_build_llvm
    ;;

  build_mlir)
    do_build_mlir
    ;;

  build_iree)
    do_build_iree
    ;;

  *)
    echo "ERROR: Expected command of 'build_llvm', 'clean_install' (got '${command}')"
    exit 1
    ;;
esac
