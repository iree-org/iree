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

# A CMake wrapper for concise command lines and stateless operation.

set -e

# This syntax allows the user to override this by defining environment
# variables.
# Note that these are currently only used on Linux and Android builds, whence
# the lack of concern about hardcoding clang as default. Might still need to
# change this to support Android builds on Windows host?
: ${CC:=clang}
: ${CXX:=clang++}

relative_path_to_this_script="$(realpath --relative-to="$(pwd)" $0)"

if [[ $# < 1 || $1 == "-h" || $1 == "--help" ]]
then
  echo "Usage: ${relative_path_to_this_script} [list of configuration options] [build [list of targets]]"
  echo
  echo "Runs CMake to perform configuration and/or build."
  echo "By default, erases CMakeCache.txt everytime for stateless operation."
  echo
  echo "Available configuration options:"
  echo
  echo "  printonly    Just print cmake commands, do not actually run them."
  echo "  noclear      Do not erase CMakeCache.txt."
  echo "  ccache       Use ccache if found."
  echo "  asan         Build with AddressSanitizer."
  echo "  msan         Build with MemorySanitizer."
  echo "  tsan         Build with ThreadSanitizer."
  echo "  Debug|Release|RelWithDebInfo|MinSizeRel    Set the CMake build type."
  echo "  src <path>   Specify IREE source path. Default: inferred from script path."
  echo "  ndk <path>   Specify Android NDK path. Make an Android build."
  echo "  docs         Build docs."
  echo "  tracy        Build with Tracy profiler instrumentation."
  echo "  py           Build Python bindings."
  echo "  java         Build Java bindings."
  echo "  tf           Build TensorFlow compiler frontend."
  echo "  tflite       Build TFLite compiler frontend."
  echo "  xla          Build XLA compiler frontend."
  echo
  echo "Examples:"
  echo
  echo "  Configure and build with ASan, CCache, and Tracy. Note how 'build' comes last!"
  echo "    ${relative_path_to_this_script} asan ccache tracy build"
  echo
  echo "  Configure and build and Android NDK build, passing the NDK path. Also with CCache and Tracy."
  echo "    ${relative_path_to_this_script} ndk ~/android-ndk-r21d ccache tracy build"
  echo
  echo "  Configure a Debug build, build only iree_tools_iree-translate:"
  echo "    ${relative_path_to_this_script} Debug build iree_tools_iree-translate"
  echo
  echo "  Configure, but do not build, a Release build with Python bindings and TensorFlow compiler:"
  echo "    ${relative_path_to_this_script} Release py tf"
  echo
  echo "  Do not configure, just build iree_tools_iree-translate"
  echo "    ${relative_path_to_this_script} build iree_tools_iree-translate"
  echo
  echo "  Configure and build a build with ASan and CCache, only build iree_tools_iree-translate:"

  exit 1
fi

if [[ -f "CMakeLists.txt" ]]
then
  echo "Error: We seem to be in a source directory. Please cd into a build directory."
  exit 1
fi

# Parse command-line args
args=("$@")
for (( i=0; i < $#; i++ ))
do
  case "${args[i]}" in
    noclear) arg_noclear=1;;
    printonly) arg_printonly=1;;
    ccache) arg_ccache=1;;
    asan) arg_asan=1;;
    msan) arg_msan=1;;
    tsan) arg_tsan=1;;
    py) arg_py=1;;
    java) arg_java=1;;
    tf) arg_tf=1;;
    tflite) arg_tflite=1;;
    xla) arg_xla=1;;
    tracy) arg_tracy=1;;
    docs) arg_docs=1;;
    Debug|Release|RelWithDebInfo|MinSizeRel) arg_build_type="${args[i]}";;
    ndk) arg_ndk="$(realpath -s ${args[$((i+1))]})"; i=$((i+1));;
    src) arg_src="$(realpath -s ${args[$((i+1))]})"; i=$((i+1));;
    build)
      arg_build=1
      if [[ $i < $(($# - 1)) ]]
      then
        arg_targets="${args[@]:$((i+1))}"
      fi  
      break;;
    *) echo "Error: unkown argument ${args[i]}"; exit 1;;
  esac
done

if [[ -z "${arg_build_type}" ]]
then
  # Sane default build type
  # TODO: upstream this sane default to IREE's CMakeLists
  arg_build_type=RelWithDebInfo
fi

if [[ -z "${arg_src}" ]]
then
  # No explicit src directory specified. Infer from this script's location.
  tentative_iree_dir="$(dirname "${relative_path_to_this_script}" | sed 's|\(.*/iree\).*|\1|')"
  if [[ -f "${tentative_iree_dir}/CMakeLists.txt" ]]
  then
    arg_src="${tentative_iree_dir}"
    echo "Inferred IREE source directory from this script's location: ${arg_src} (pass src <path> to override)"
    echo
  fi
fi

# Detect Linux builds (not including Android)
if [[ "$OSTYPE" == "linux-gnu"* && -z "${arg_ndk}" ]]
then
  is_linux_build=1
fi

# Build a list of CMake variables to set.
cmake_var_names=()
cmake_var_values=()

function add_cmake_var() {
  if [[ ! -z "$1" ]]
  then
    cmake_var_names+=("$2")
    cmake_var_values+=("$3")
  fi
}

add_cmake_var "${arg_build_type}" CMAKE_BUILD_TYPE "${arg_build_type}"
add_cmake_var "${arg_ndk}" CMAKE_TOOLCHAIN_FILE "${arg_ndk}/build/cmake/android.toolchain.cmake"
add_cmake_var "${arg_ndk}" ANDROID_ABI "arm64-v8a"
add_cmake_var "${arg_ndk}" ANDROID_PLATFORM "$(ls -1v "${arg_ndk}/platforms/" | tail -n1)"
add_cmake_var "${arg_ndk}" IREE_HOST_C_COMPILER "$(which "$CC")"
add_cmake_var "${arg_ndk}" IREE_HOST_CXX_COMPILER "$(which "$CXX")"
add_cmake_var "${arg_ndk}" IREE_BUILD_COMPILER OFF
add_cmake_var "${arg_ndk}" IREE_BUILD_SAMPLES OFF
add_cmake_var "${arg_ccache}" IREE_ENABLE_CCACHE ON
add_cmake_var "${arg_ccache}" LLVM_CCACHE_BUILD ON
add_cmake_var "${arg_asan}" IREE_ENABLE_ASAN ON
add_cmake_var "${arg_msan}" IREE_ENABLE_MSAN ON
add_cmake_var "${arg_tsan}" IREE_ENABLE_TSAN ON
add_cmake_var "${arg_tf}" IREE_BUILD_TENSORFLOW_COMPILER ON
add_cmake_var "${arg_tflite}" IREE_BUILD_TFLITE_COMPILER ON
add_cmake_var "${arg_xla}" IREE_BUILD_XLA_COMPILER ON
add_cmake_var "${arg_py}" IREE_BUILD_PYTHON_BINDINGS ON
add_cmake_var "${arg_java}" IREE_BUILD_JAVA_BINDINGS ON
add_cmake_var "${arg_tracy}" IREE_ENABLE_RUNTIME_TRACING ON
# On Linux, the default choice of compiler might be GCC, so we need to override
# that.
add_cmake_var "${is_linux_build}" CMAKE_C_COMPILER "$(which "$CC")"
add_cmake_var "${is_linux_build}" CMAKE_CXX_COMPILER "$(which "$CXX")"

# Build the CMake configure command line.
num_cmake_vars=${#cmake_var_names[@]}

# Do we need to set CMake variables? Then we need a CMake configure command.
if [[ "$num_cmake_vars" > 0 ]]
then
  # Then we need a source directory.
  if [[ -z "${arg_src}" ]]
  then
    echo "Please specify the IREE source directory (src <path>)."
    exit 1
  fi

  if [[ ! -f "${arg_src}/CMakeLists.txt" ]]
  then
    echo "Error: ${arg_src} does not look like a source directory: it should contain CMakeLists.txt"
    exit 1
  fi

  configure_cmdline="cmake ${arg_src} -G Ninja"

  for (( i = 0; i < $num_cmake_vars; i++ ))
  do
    configure_cmdline+=$' \\\n'"  -D${cmake_var_names[$i]}=${cmake_var_values[$i]}"
  done

  echo "CMake configuration command line:"
  echo
  echo "$configure_cmdline"
  echo
fi

# Build the CMake build command line.
if [[ ! -z "${arg_build}" ]]
then
  build_cmdline="cmake --build ."
  if [[ ! -z "${arg_targets}" ]]
  then
    build_cmdline+=$' \\\n'"  --target ${arg_targets}"
  fi

  # The CMake build may invoke Bazel to build some directory such as
  # Tensorflow. On Linux, this may require overriding the default choice of
  # compiler, GCC.
  if [[ "${is_linux_build}" == 1 ]]
  then
    build_cmdline="CC=${CC} CXX=${CXX} ${build_cmdline}"
  fi

  echo "CMake build command line:"
  echo
  echo "$build_cmdline"
  echo
fi

if [[ "${arg_printonly}" == 1 ]]
then
  exit 0
fi

if [[ -f "CMakeCache.txt" && -z "${arg_noclear}" ]]
then
  echo "Erasing CMakeCache.txt for stateless operation (pass noclear to override)."
  echo
  rm "CMakeCache.txt"
fi

# Run configure.
if [[ ! -z "configure_cmdline" ]]
then
  eval "$configure_cmdline"

  # Check if CMakeCache.txt has what we requested.
  for (( i = 0; i < $num_cmake_vars; i++ ))
  do
    if ! grep -xq "${cmake_var_names[$i]}\b.*=${cmake_var_values[$i]}"  "CMakeCache.txt"
    then
      echo "Error: After running CMake, CMakeCache.txt does not have ${cmake_var_names[$i]} set to ${cmake_var_values[$i]} as requested."
      echo "Suggestion: rm CMakeCache.txt and rerun this script."
      exit 1
    fi
  done
fi

# Run build.
if [[ ! -z "build_cmdline" ]]
then
  eval "$build_cmdline"
fi
