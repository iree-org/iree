#!/bin/bash

# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Wrapper script to push build artifacts and run tests on an Android device.
#
# This script expects the following arguments:
#   <test-binary> [<test-args>]..
# Where <test-binary> should be a path relative to /data/local/tmp/ on device.
#
# This script reads the following environment variables:
# - TEST_ANDROID_ABS_DIR: the absolute path on Android device for the build
#   artifacts.
# - TEST_DATA: optional; the files to push to the Android device. Space-separated.
# - TEST_EXECUTABLE: the executable file to push to the Android device.
# - TEST_TMPDIR: optional; temporary directory on the Android device for
#   running tests.
#
# This script pushes $TEST_EXECUTABLE and $TEST_DATA onto the device
# under $TEST_ANDROID_ABS_DIR/ before running <test-binary> with all
# <test-args> under /data/local/tmp.

set -x
set -e

adb push $TEST_EXECUTABLE $TEST_ANDROID_ABS_DIR/$(basename $TEST_EXECUTABLE)

if [ -n "$TEST_DATA" ]; then
  for datafile in $TEST_DATA
  do
    adb push "$datafile" "$TEST_ANDROID_ABS_DIR/$(basename "$datafile")"
  done
fi

if [ -n "$TEST_TMPDIR" ]; then
  adb shell "mkdir -p $TEST_TMPDIR"
  tmpdir="TEST_TMPDIR=$TEST_TMPDIR"
else
  tmpdir=""
fi

# Execute the command with `adb shell` under `/data/local/tmp`.
# We set LD_LIBRARY_PATH for the command so that it can use libvulkan.so under
# /data/local/tmp when running Vulkan tests. This is to workaround an Android
# issue where linking to libvulkan.so is broken under /data/local/tmp.
# See https://android.googlesource.com/platform/system/linkerconfig/+/296da5b1eb88a3527ee76352c2d987f82f3252eb.
# This requires copying the vendor vulkan implementation under
# /vendor/lib[64]/hw/vulkan.*.so to /data/local/tmp/libvulkan.so.
adb shell "cd /data/local/tmp && LD_LIBRARY_PATH=/data/local/tmp $tmpdir $*"

if [ -n "$TEST_TMPDIR" ]; then
  adb shell "rm -rf $TEST_TMPDIR"
fi
