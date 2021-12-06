#!/bin/sh

# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Runs on a Pixel 6 device itself to set the GPU frequency scaling policy.

################################### WARNING ####################################
# This will overheat the phone if it's not on a cooling plate, resulting in    #
# thermal throttling. To prevent anything catching on fire, the actual GPU     #
# frequencies will be throttled to below the maximum, skewing your results.    #
################################################################################

set -euo pipefail

POLICY="${1:-always_on}"

readonly MALI_GPU_PATH="/sys/devices/platform/1c500000.mali"

echo "GPU info (before changing power policy):"
echo 'policy\t\t\t\t\tcur\tmin\tmax'
echo "--------------------------------------------------------------"
paste \
  "${MALI_GPU_PATH}/power_policy" \
  "${MALI_GPU_PATH}/cur_freq" \
  "${MALI_GPU_PATH}/min_freq" \
  "${MALI_GPU_PATH}/max_freq"

echo "Setting GPU power policy to ${POLICY}"

if [[ "$POLICY" == "always_on" ]]; then
  echo "always_on" > "${MALI_GPU_PATH}/power_policy"
  cat "${MALI_GPU_PATH}/max_freq" > "${MALI_GPU_PATH}/scaling_max_freq"
  cat "${MALI_GPU_PATH}/max_freq" > "${MALI_GPU_PATH}/scaling_min_freq"
elif [[ "$POLICY" == "coarse_demand" ]]; then
  echo "coarse_demand" > "${MALI_GPU_PATH}/power_policy"
  cat "${MALI_GPU_PATH}/max_freq" > "${MALI_GPU_PATH}/scaling_max_freq"
  cat "${MALI_GPU_PATH}/min_freq" > "${MALI_GPU_PATH}/scaling_min_freq"
else
  echo "Unknown power policy: ${POLICY}"
  exit 1
fi

echo "GPU info (after changing power policy):"
echo 'policy\t\t\t\t\tcur\tmin\tmax'
echo "--------------------------------------------------------------"
paste \
  "${MALI_GPU_PATH}/power_policy" \
  "${MALI_GPU_PATH}/cur_freq" \
  "${MALI_GPU_PATH}/min_freq" \
  "${MALI_GPU_PATH}/max_freq"
