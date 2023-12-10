#!/bin/sh

# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Runs on a Pixel device itself to set the GPU frequency scaling policy.

################################### WARNING ####################################
# This will overheat the phone if it's not on a cooling plate, resulting in    #
# thermal throttling. To prevent anything catching on fire, the actual GPU     #
# frequencies will be throttled to below the maximum, skewing your results.    #
################################################################################

set -euo pipefail

POLICY="${1:-performance}"

MALI_GPU_PATTERN="/sys/devices/platform/*.mali"
MALI_GPU_PATHS=($MALI_GPU_PATTERN)
# Assume there is only 1 GPU.
if (( "${#MALI_GPU_PATHS[@]}" != 1 )); then
  echo "Too many GPUs: ${MALI_GPU_PATHS[@]}"
  exit 1
fi
MALI_GPU_PATH="${MALI_GPU_PATHS[0]}"

echo "GPU info (before changing frequency scaling policy):"
echo 'policy\t\t\t\t\tcur\tmin\tmax'
echo "--------------------------------------------------------------"
paste \
  "${MALI_GPU_PATH}/power_policy" \
  "${MALI_GPU_PATH}/cur_freq" \
  "${MALI_GPU_PATH}/min_freq" \
  "${MALI_GPU_PATH}/max_freq"

echo "Setting GPU frequency scaling policy to ${POLICY}"

case "$POLICY" in
  performance)
    echo "always_on" > "${MALI_GPU_PATH}/power_policy"
    cat "${MALI_GPU_PATH}/max_freq" > "${MALI_GPU_PATH}/scaling_max_freq"
    cat "${MALI_GPU_PATH}/max_freq" > "${MALI_GPU_PATH}/scaling_min_freq"
    ;;
  default)
    echo "coarse_demand" > "${MALI_GPU_PATH}/power_policy"
    cat "${MALI_GPU_PATH}/max_freq" > "${MALI_GPU_PATH}/scaling_max_freq"
    cat "${MALI_GPU_PATH}/min_freq" > "${MALI_GPU_PATH}/scaling_min_freq"
    ;;
  *)
    echo "Unknown frequency scaling policy: ${POLICY}"
    exit 1
    ;;
esac

echo "GPU info (after changing frequency scaling policy):"
echo 'policy\t\t\t\t\tcur\tmin\tmax'
echo "--------------------------------------------------------------"
paste \
  "${MALI_GPU_PATH}/power_policy" \
  "${MALI_GPU_PATH}/cur_freq" \
  "${MALI_GPU_PATH}/min_freq" \
  "${MALI_GPU_PATH}/max_freq"
