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

# Benchmarks modules generating by compile_modules.sh on Android and writes
# performance data to a text proto file, perfgate-SERIAL_NUMBER-GIT_HASH.
#
# The script benchmarks modules on 7-th big core, ie, run with `taskset 80`.

set -o errexit
set -o pipefail

GIT_HASH="UNKNOWN"
while [[ $# -gt 0 ]]; do
  token="$1"
  case $token in
    --model=*)
      MODEL=${1#*=}
      shift
      ;;
    --targets=*)
      TARGETS=${1#*=}
      shift
      ;;
    --benchmark_key=*)
      BENCHMARK_KEY=${1#*=}
      shift
      ;;
    --git_hash=*)
      GIT_HASH=${1#*=}
      shift
      ;;
  esac
done

TS=$(date +%s)000
SED_EXPR='s/BM_[[:alnum:][:punct:]]+ [[:space:]]+ ([0-9]+) ms[[:print:]]+/\1/p'
OUTPUT_DIR=test_output_files
DEVICE_ROOT=/data/local/tmp/benchmark_tmpdir

mkdir -p "${OUTPUT_DIR}"

if [[ -z "${MODEL}" ]]; then
  echo "Must set --model flag.";
  exit 1
fi

if [[ -z "${BENCHMARK_KEY}" ]]; then
  echo "Must set --benchmark_key flag.";
  exit 1
fi

function perfgate_metadata {
  local perfgate_log=$1
  cat >> "${perfgate_log}" << ---EOF---
metadata: {
  git_hash: "${GIT_HASH}"
  timestamp_ms: ${TS}
  benchmark_key: "${BENCHMARK_KEY}"
}
---EOF---
}

function perfgate_sample {
  local perfgate_log=$1
  local value=$2
  local tag=$3
  cat >> "${perfgate_log}" << ---EOF---
samples: {
  time: ${value}
  target: "${tag}"
}
---EOF---
}

function run_and_log {
  local perfgate_log=$1
  local target=$2
  case "${target}" in
    "vulkan")
      tag='g'
      ;;
    "vmla")
      tag='v'
      ;;
    "dylib")
      tag='c'
      ;;
    *)
      echo "Not supported target"
      exit 1
      ;;
  esac

  test_out="${OUTPUT_DIR}/${MODEL}-${target}_output.txt"
  adb shell LD_LIBRARY_PATH=/data/local/tmp taskset 80 \
    ${DEVICE_ROOT}/iree-benchmark-module \
    --flagfile=${DEVICE_ROOT}/flagfile \
    --module_file=${DEVICE_ROOT}/${MODEL}-${target}.vmfb \
    --driver=${target} | tee ${test_out}
  for ms in $(sed -En -e "${SED_EXPR}" "${test_out}"); do
    perfgate_sample "${perfgate_log}" "${ms}" "${tag}"
  done
}

phone_serial_number=$(adb get-serialno)
perfgate_log="perfgate-${phone_serial_number}-${GIT_HASH}.log"
: > "${perfgate_log}" # Empty the log file

IFS=',' read -ra targets_array <<< "$TARGETS"
for target in "${targets_array[@]}"
do
  echo "Executing ${target}..."
  run_and_log "${perfgate_log}" "${target}"
done

perfgate_metadata "${perfgate_log}"
