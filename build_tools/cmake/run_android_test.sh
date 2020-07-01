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

# Wrapper script to push build artifacts and run tests on an Android device.
#
# This script expects the following arguments:
#   <test-binary> [<test-args>]..
# Where <test-binary> should be a path relative to /data/local/tmp/ on device.
#
# This script reads the following environment variables:
# - TEST_ANDROID_ABS_DIR: the absolute path on Android device for the build
#   artifact.
# - TEST_ARTIFACT: the build artifact to push to the Android device.
# - TEST_ARTIFACT_NAME: the build artifact's filename.
# - TEST_TMPDIR: optional; temporary directory on the Android device for
#   running tests.
#
# This script pushes $TEST_ARTIFACT onto the device as
# $TEST_ANDROID_ABS_DIR/$TEST_ARTIFACT_NAME before running
# <test-binary> with all <test-args> under /data/local/tmp.

set -x
set -e

adb push $TEST_ARTIFACT $TEST_ANDROID_ABS_DIR/$TEST_ARTIFACT_NAME

if [ -n "$TEST_TMPDIR" ]; then
  adb shell "mkdir -p $TEST_TMPDIR"
  tmpdir="TEST_TMPDIR=$TEST_TMPDIR"
else
  tmpdir=""
fi

adb shell "cd /data/local/tmp && $tmpdir $*"

if [ -n "$TEST_TMPDIR" ]; then
  adb shell "rm -rf $TEST_TMPDIR"
fi
