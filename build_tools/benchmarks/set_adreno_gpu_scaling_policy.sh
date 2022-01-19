#!/bin/sh

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Runs on a phone with Adreno GPU to set the GPU frequency scaling policy.

################################### WARNING ####################################
# This will overheat the phone if it's not on a cooling plate, resulting in    #
# thermal throttling. To prevent anything catching on fire, the actual GPU     #
# frequencies will be throttled to below the maximum, skewing your results.    #
################################################################################

set -euo pipefail

POLICY="${1:-performance}"

readonly ADRENO_GPU_PATH="/sys/class/kgsl/kgsl-3d0"

# Available frequencies are sorted, either in ascending or descending order.
readonly ADRENO_MIN_FREQ=$(cat "${ADRENO_GPU_PATH}/devfreq/available_frequencies" | tr " " "\n" | sort -u -n | head -1)
readonly ADRENO_MAX_FREQ=$(cat "${ADRENO_GPU_PATH}/devfreq/available_frequencies" | tr " " "\n" | sort -u -n | tail -1)

# Power levels match available freqencies.
readonly ADRENO_MAX_PWRLEVEL=0
(( ADRENO_MIN_PWRLEVEL = $(cat "${ADRENO_GPU_PATH}/num_pwrlevels") - 1 ))
readonly ADRENO_MIN_PWRLEVEL

# Idle timers affect governor change and frequncy reset.
readonly ADRENO_DEFAULT_IDLE_TIMER=80    # ms
readonly ADRENO_1HOUR_IDLE_TIMER=3600000 # ms

echo "GPU info (before changing frequency scaling policy):"
echo 'model\t\tcur\t\tmin\t\tmax'
echo "---------------------------------------------------------"
paste \
  "${ADRENO_GPU_PATH}/gpu_model" \
  "${ADRENO_GPU_PATH}/devfreq/cur_freq" \
  "${ADRENO_GPU_PATH}/devfreq/min_freq" \
  "${ADRENO_GPU_PATH}/devfreq/max_freq"

echo "Setting GPU frequency scaling policy to ${POLICY}"

case "$POLICY" in
  performance)
    echo 1 > "${ADRENO_GPU_PATH}/force_clk_on"
    echo ${ADRENO_1HOUR_IDLE_TIMER} > "${ADRENO_GPU_PATH}/idle_timer"

    # Some devices only expose the msm-adreno-tz governor, so allow the
    # following to fail.
    echo performance > "${ADRENO_GPU_PATH}/devfreq/governor" || true

    echo ${ADRENO_MAX_FREQ} > "${ADRENO_GPU_PATH}/gpuclk"
    echo ${ADRENO_MAX_FREQ} > "${ADRENO_GPU_PATH}/devfreq/max_freq"
    echo ${ADRENO_MAX_FREQ} > "${ADRENO_GPU_PATH}/devfreq/min_freq"

    echo ${ADRENO_MAX_PWRLEVEL} > "${ADRENO_GPU_PATH}/max_pwrlevel"
    echo ${ADRENO_MAX_PWRLEVEL} > "${ADRENO_GPU_PATH}/min_pwrlevel"
    ;;
  default)
    echo 0 > "${ADRENO_GPU_PATH}/force_clk_on"
    echo ${ADRENO_DEFAULT_IDLE_TIMER} > "${ADRENO_GPU_PATH}/idle_timer"

    # msm-adreno-tz is the default governor for Adreno GPUs.
    echo msm-adreno-tz > "${ADRENO_GPU_PATH}/devfreq/governor"

    echo ${ADRENO_MAX_FREQ} > "${ADRENO_GPU_PATH}/devfreq/max_freq"
    echo ${ADRENO_MIN_FREQ} > "${ADRENO_GPU_PATH}/devfreq/min_freq"

    echo ${ADRENO_MAX_PWRLEVEL} > "${ADRENO_GPU_PATH}/max_pwrlevel"
    echo ${ADRENO_MIN_PWRLEVEL} > "${ADRENO_GPU_PATH}/min_pwrlevel"
    ;;
  *)
    echo "Unknown frequency scaling policy: ${POLICY}"
    exit 1
    ;;
esac

echo "GPU info (after changing frequency scaling policy):"
echo 'model\t\tcur\t\tmin\t\tmax'
echo "---------------------------------------------------------"
paste \
  "${ADRENO_GPU_PATH}/gpu_model" \
  "${ADRENO_GPU_PATH}/devfreq/cur_freq" \
  "${ADRENO_GPU_PATH}/devfreq/min_freq" \
  "${ADRENO_GPU_PATH}/devfreq/max_freq"
