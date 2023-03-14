#!/bin/bash

# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Reads text from stdin containing stack frames in the format
# `#3 0x5b09835a14  (/data/local/tmp/iree-benchmark-module+0x18aa14)`
# and symbolizes such lines and prints them to stdout.
#
# The approach is to pull the files out of the device and use the host
# build of llvm-symbolizer provided in the NDK.
#
# Other lines are just echo'd to stdout.
# This is meant to be used like this:
# ANDROID_NDK=~/android-ndk-r21d ./build_tools/scripts/android_symbolize.sh < /tmp/asan.txt
# Where one has previously stored e.g. some ASan report to a file, here /tmp/asan.txt.
#
# See docs/developing_iree/sanitizers.md.
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
  llvm_symbolizer_output="$($ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-symbolizer --obj "$pulled_file" "$address")"
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
