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

# Runs the liburing configure script to update compat headers here.
# At the time of this writing, this facility is very simple and ok to just
# snapshot (which will need to be done for cross-compilation anyway). If this
# ever changes, something more exotic than a manual update will need to be
# done.

this_dir="$(cd $(dirname $0) && pwd)"
liburing_dir="$this_dir/../../../third_party/liburing"

if ! [ -d "$liburing_dir" ]; then
  echo "ERROR: Could not find directory $liburing_dir"
  exit 1
fi

# The configure script outputs files into the current directory and a
# src/include/liburing directory, matching the source tree.
config_dir="$this_dir/default_config"
mkdir -p "$config_dir/src/include/liburing"
cd "$config_dir"

if ! bash "$liburing_dir/configure"; then
  echo "ERROR: Could not configure"
  exit 2
fi
