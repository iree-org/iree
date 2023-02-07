#!/bin/bash
# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -e

#########################
# Script input parameters
#########################

echo "=== A script for building iree-run-module apk ==="
echo "This script wraps iree-run-module together with a"
echo "specific IREE VM module and invocation information."
echo ""
echo "At a minimum, it expects the following env vars:"
echo "* ANDROID_SDK_ROOT"
echo "* ANDROID_NDK_ROOT"
echo "* ANDROID_SDK_BUILD_TOOLS_VERSION"
echo "See the script for details and more controls."
echo "===-------------------------------------------==="
echo ""

print_usage_and_exit() {
  echo "Usage: $0 <artifact-directory> "
  echo "       --device <device>"
  echo "       --module <input-module-file> "
  echo "       --function <entry-function> "
  echo "       --inputs_file <input-buffer-file> "
  exit 1
}

while (( "$#" )); do
  case "$1" in
    --device)
      if [[ -n "$2" ]] && [[ ${2:0:1} != "-" ]]; then
        IREE_DEVICE=$2
        shift 2
      else
        echo "Error: missing argument for $1" >&2
        print_usage_and_exit
      fi
      ;;
    --module)
      if [[ -n "$2" ]] && [[ ${2:0:1} != "-" ]]; then
        IREE_INPUT_MODULE_FILE=$(readlink -f $2)
        shift 2
      else
        echo "Error: missing argument for $1" >&2
        print_usage_and_exit
      fi
      ;;
    --function)
      if [[ -n "$2" ]] && [[ ${2:0:1} != "-" ]]; then
        IREE_ENTRY_FUNCTION=$2
        shift 2
      else
        echo "Error: missing argument for $1" >&2
        print_usage_and_exit
      fi
      ;;
    --inputs_file)
      if [[ -n "$2" ]] && [[ ${2:0:1} != "-" ]]; then
        IREE_INPUT_BUFFER_FILE=$(readlink -f $2)
        shift 2
      else
        echo "Error: missing argument for $1" >&2
        print_usage_and_exit
      fi
      ;;
    -*|--*=) # Unsupported flags
      echo "Error: Unsupported flag $1" >&2
      exit 1
      ;;
    *) # Positional arguments
      if [[ -z "${IREE_ARTIFACT_ROOT+x}" ]]; then
        IREE_ARTIFACT_ROOT=$(readlink -f $1)
      else
        echo "Error: <artifact-directory> already set to ${IREE_ARTIFACT_ROOT}" >&2
        print_usage_and_exit
      fi
      shift
      ;;
  esac
done

if [[ -z "${IREE_ARTIFACT_ROOT}" ]] || [[ -z "${IREE_INPUT_MODULE_FILE}" ]] || \
   [[ -z "${IREE_ENTRY_FUNCTION}" ]] || [[ -z "${IREE_INPUT_BUFFER_FILE}" ]] || \
   [[ -z "${IREE_DEVICE}" ]]; then
  echo "Error: missing necessary parameters" >&2
  print_usage_and_exit
fi

#################################
# IREE Android app configurations
#################################

# The final Android APK name; default to "iree-run-module.apk".
IREE_APK_NAME="${IREE_APK_NAME:-iree-run-module}"
# The CMAKE build type for compiling IREE. default to Release.
IREE_BUILD_TYPE="${IREE_BUILD_TYPE:-Release}"
# The target Android API level; default to Android 10 (API level 29).
IREE_ANDROID_API_LEVEL="${IREE_ANDROID_API_LEVEL:-29}"
# The target Android platform ABI; default to arm64-v8a.
IREE_ANDROID_ABI="${IREE_ANDROID_ABI:-arm64-v8a}"

################################
# Android SDK/NDK configurations
################################

# Android SDK root; must be provided as environment variable.
# By default Android Studio uses
# * $HOME/Android/Sdk for Linux.
# * $HOME/Library/Android/sdk for macOS.
# * $%LOCALAPPDATA%\Android\sdk for Windows.
ANDROID_SDK_ROOT="${ANDROID_SDK_ROOT:?environment variable not set}"
# Android NDK root; must be provided as environment variable.
ANDROID_NDK_ROOT="${ANDROID_NDK_ROOT:?environment variable not set}"
# Android SDK build tools version; must be provided as environment variable.
ANDROID_SDK_BUILD_TOOLS_VERSION="${ANDROID_SDK_BUILD_TOOLS_VERSION?environment variable not set}"
# Key store for signing apk files; default to Android debug keystore created
# by Android Studio.
ANDROID_KEYSTORE="${ANDROID_KEYSTORE:-${HOME}/.android/debug.keystore}"

######################
# IREE build toolchain
######################

CMAKE_BIN="${CMAKE_BIN:-$(which cmake)}"
NINJA_BIN="${NINJA_BIN:-$(which ninja)}"
CC_BIN="${CC_BIN:-$(which clang)}"
CXX_BIN="${CXX_BIN:-$(which clang++)}"
JAVAC_BIN="${JAVAC_BIN:-$(which javac)}"

##################################
# IREE source/artifact directories
##################################

# IREE project source root.
IREE_SOURCE_ROOT="$(git rev-parse --show-toplevel)"
# iree-run-module Android app source root.
IREE_NATIVE_APP_SOURCE_ROOT="${IREE_SOURCE_ROOT?}/tools/android/run_module_app"

# Directory for IREE native code intermediate intermediate artifacts.
IREE_NATIVE_LIB_BUILD_DIR="${IREE_ARTIFACT_ROOT?}/iree/${IREE_BUILD_TYPE?}"
# Directory for holding APK parts.
IREE_APK_PARTS_DIR="${IREE_ARTIFACT_ROOT?}/parts"
# Directory for IREE native libraries.
IREE_NATIVE_LIB_DIR="${IREE_APK_PARTS_DIR?}/libs/lib/${IREE_ANDROID_ABI?}"
# Directory for Android app assets.
IREE_ASSET_DIR="${IREE_APK_PARTS_DIR?}/assets"
# Directory for Android app R.class.
IREE_RESOURCE_GEN_DIR="${IREE_APK_PARTS_DIR?}/rclass"

#########################
# Android build toolchain
#########################

ANDROID_SDK_BUILD_TOOLS_DIR="${ANDROID_SDK_ROOT?}/build-tools/${ANDROID_SDK_BUILD_TOOLS_VERSION?}"
ANDROID_SDK_PLATFORMS_DIR="${ANDROID_SDK_ROOT?}/platforms/android-${IREE_ANDROID_API_LEVEL?}"

AAPT_BIN="${ANDROID_SDK_BUILD_TOOLS_DIR?}/aapt"
DX_BIN="${ANDROID_SDK_BUILD_TOOLS_DIR?}/dx"
ZIPALIGN_BIN="${ANDROID_SDK_BUILD_TOOLS_DIR?}/zipalign"
APKSIGNER_BIN="${ANDROID_SDK_BUILD_TOOLS_DIR?}/apksigner"

AAPT_ADD="${AAPT_BIN?} add"
# Link in the Android framework classes and disable compression for IREE
# bytecode modules. This allows us to mmap the file directly.
AAPT_PACK="${AAPT_BIN?} package -f -I ${ANDROID_SDK_PLATFORMS_DIR?}/android.jar -0 vmfb"
DX="${DX_BIN?} --dex"
ZIPALIGN="${ZIPALIGN_BIN?} -f -p 4"
APKSIGN="${APKSIGNER_BIN?} sign"
JAVAC="${JAVAC_BIN?} -classpath ${ANDROID_SDK_PLATFORMS_DIR?}/android.jar -sourcepath ${IREE_RESOURCE_GEN_DIR?} -d ${IREE_APK_PARTS_DIR?}"

#############################
# Build IREE native libraries
#############################

mkdir -p "${IREE_NATIVE_LIB_BUILD_DIR?}"

echo ">>> Building IREE native libraries <<<"

IREE_NATIVE_LIB_NAME=iree_tools_android_run_module_app_iree_run_module_app

pushd "${IREE_NATIVE_LIB_BUILD_DIR?}"
"${CMAKE_BIN?}" "${IREE_SOURCE_ROOT?}" -G Ninja \
  -DCMAKE_BUILD_TYPE="${IREE_BUILD_TYPE?}" \
  -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK_ROOT?}/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI="${IREE_ANDROID_ABI?}" \
  -DANDROID_PLATFORM="android-${IREE_ANDROID_API_LEVEL?}" \
  -DIREE_HOST_C_COMPILER="${CC_BIN?}" \
  -DIREE_HOST_CXX_COMPILER="${CXX_BIN?}" \
  -DIREE_BUILD_COMPILER=OFF \
  -DIREE_BUILD_TESTS=OFF \
  -DIREE_BUILD_SAMPLES=OFF
"${NINJA_BIN?}" "${IREE_NATIVE_LIB_NAME?}"
popd

#####################
# Package Android app
#####################

# Clean artifacts from previous runs.
rm -rf "${IREE_ARTIFACT_ROOT?}/${IREE_APK_NAME?}.apk" "${IREE_APK_PARTS_DIR?}"

echo ">>> Generating AndroidManifest.xml <<<"

# Create an AndroidManifest.xml with proper target SDK version.
mkdir -p "${IREE_APK_PARTS_DIR?}"
IREE_ANDROID_API_LEVEL="${IREE_ANDROID_API_LEVEL?}" envsubst \
  < "${IREE_NATIVE_APP_SOURCE_ROOT?}/AndroidManifest.xml.template" \
  > "${IREE_APK_PARTS_DIR?}/AndroidManifest.xml"

echo ">>> Preparing shared libraries and vm module information <<<"

# Find the compiled iree_run_module_app shared library and symlink it to a
# known location for packaging.
mkdir -p "${IREE_NATIVE_LIB_DIR?}"
IREE_NATIVE_LIB=$(find ${IREE_NATIVE_LIB_BUILD_DIR?} -name "lib${IREE_NATIVE_LIB_NAME?}.so")
# Note: the target link name must match with
# run_module_app/AndroidManifest.xml.template.
ln -sf "${IREE_NATIVE_LIB?}" "${IREE_NATIVE_LIB_DIR?}/libiree_run_module_app.so"

# Copy the VM FlatBuffer and iree-run-module invocation related information
# over as assets.
mkdir -p "${IREE_ASSET_DIR?}"
# Note: the following files must match with run_module_app/src/main.cc.
cp "${IREE_INPUT_MODULE_FILE?}" "${IREE_ASSET_DIR?}/module.vmfb"
cp "${IREE_INPUT_BUFFER_FILE?}" "${IREE_ASSET_DIR?}/inputs.txt"
echo -n "${IREE_ENTRY_FUNCTION?}" > "${IREE_ASSET_DIR?}/entry_function.txt"
echo -n "${IREE_DEVICE?}" > "${IREE_ASSET_DIR?}/device.txt"

echo ">>> Compiling app resources <<<"

# Generate the R.java for resources.
mkdir -p "${IREE_RESOURCE_GEN_DIR?}"
${AAPT_PACK?} --non-constant-id -m \
  -M "${IREE_APK_PARTS_DIR?}/AndroidManifest.xml" \
  -S "${IREE_NATIVE_APP_SOURCE_ROOT?}/res" \
  -J "${IREE_RESOURCE_GEN_DIR?}" \
  --generate-dependencies

# Compile the R.java and create classes.dex out of it for Android.
echo "Using javac: '${JAVAC_BIN?}'"
${JAVAC?} "${IREE_RESOURCE_GEN_DIR?}"/com/iree-org/iree/run_module/*.java
${DX?} --output="${IREE_APK_PARTS_DIR?}/classes.dex" "${IREE_APK_PARTS_DIR?}"

echo ">>> Packaging apk file <<<"

# Package assets and shared libraries into an apk file.
${AAPT_PACK?} -m \
  -M "${IREE_APK_PARTS_DIR?}/AndroidManifest.xml" \
  -S "${IREE_NATIVE_APP_SOURCE_ROOT?}/res" \
  -A "${IREE_APK_PARTS_DIR?}/assets" \
  -F "${IREE_APK_PARTS_DIR?}/${IREE_APK_NAME?}.unaligned.apk" \
  --shared-lib "${IREE_APK_PARTS_DIR?}/libs"

pushd "${IREE_APK_PARTS_DIR?}"
# Also package the resources into the apk file.
${AAPT_ADD?} "${IREE_APK_NAME?}.unaligned.apk" classes.dex
echo ">>> Aligning apk file <<<"
${ZIPALIGN?} "${IREE_APK_NAME?}.unaligned.apk" "${IREE_APK_NAME?}.apk"
echo ">>> Signing apk file <<<"
echo "NOTE: if you are using the Android Studio's debug keystore, the password is 'android'."
${APKSIGN?} --ks "${ANDROID_KEYSTORE?}" --min-sdk-version 28 "${IREE_APK_NAME?}.apk"
mv "${IREE_APK_NAME?}.apk" "${IREE_ARTIFACT_ROOT?}"
popd

echo ">>> Done: '${IREE_ARTIFACT_ROOT?}/${IREE_APK_NAME?}.apk' <<<"
