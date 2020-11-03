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

# Reads text from stdin containing stack frames in the format
# `#3 0x5b09835a14  (/data/local/tmp/iree-benchmark-module+0x18aa14)`
# and symbolizes such lines and prints them to stdout.
#
# The approach is to pull the files out of the device and use the host
# build of llvm-symbolizer provided in the NDK.
#
# Other lines are just echo'd to stdout.
# This is meant to be used like this:
# ANDROID_NDK=~/android-ndk-r21d ~/android-symbolize.sh < ~/a
# Where one has previously stored e.g. some ASan report to a file, here ~/a.
#
# Discussion of alternatives: https://github.com/android/ndk/issues/753

# Provide the location of your Android NDK in the ANDROID_NDK env var.
: ${ANDROID_NDK:=""}

if [ -z "${ANDROID_NDK}" ]
then
  echo "Please define ANDROID_NDK to point to your Android NDK."
  exit 1
fi

tmpdir="$(mktemp -d)"

while read line
do
  header="$(echo "$line" | grep -o '^\s*#[0-9]\+')"
  if [ -z "$header" ]
  then
    echo "$line"
    continue
  fi

  location_with_parentheses="$(echo "$line" | grep -o '(/[^)]*)$')"
  location="$(echo "$location_with_parentheses" | tr -d '()' )"
  file="$(echo "$location" | cut -d '+' -f 1)"
  address="$(echo "$location" | cut -d '+' -f 2)"
  if [[ -z "$file" || -z "$address" ]]
  then
    echo "$line"
    continue
  fi

  file_basename="$(basename "$file")"
  pulled_file="$tmpdir/$file_basename"
  if [ ! -f "$pulled_file" ]
  then
    adb pull "$file" "$pulled_file" 1>/dev/null 2>/dev/null
  fi
  llvm_symbolizer_output="$($ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-symbolizer -obj "$pulled_file" "$address")"
  if [ -z "$llvm_symbolizer_output" ]
  then
    echo "$line"
    continue
  fi

  function="$(echo "$llvm_symbolizer_output" | head -n1)"
  source_location="$(echo "$llvm_symbolizer_output" | tail -n1)"
  echo "$header $source_location $function"
done

rm -rf "$tmpdir"
