#!/bin/sh

# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Runs on an android device itself to set the frequency scaling governor for all
# CPUs (default performance).

################################### WARNING ####################################
# This will overheat the phone if it's not on a cooling plate, resulting in    #
# thermal throttling. To prevent anything catching on fire, the actual CPU     #
# frequencies will be throttled to below the maximum, skewing your results.    #
################################################################################

set -euo pipefail

GOVERNOR="${1:-performance}"

echo "CPU info (before changing governor):"
echo 'cpu\tgovernor\tcur\tmin\tmax'
echo "------------------------------------------------"
for i in `cat /sys/devices/system/cpu/present | tr '-' ' ' | xargs seq`; do \
    echo "cpu${i}" | paste \
      - \
      "/sys/devices/system/cpu/cpu${i}/cpufreq/scaling_governor" \
      "/sys/devices/system/cpu/cpu${i}/cpufreq/cpuinfo_cur_freq" \
      "/sys/devices/system/cpu/cpu${i}/cpufreq/cpuinfo_min_freq" \
      "/sys/devices/system/cpu/cpu${i}/cpufreq/cpuinfo_max_freq"; \
done

echo "Setting CPU frequency governor to ${GOVERNOR}"

for i in `cat /sys/devices/system/cpu/present | tr '-' ' ' | xargs seq`; do \
  echo "${GOVERNOR}" > \
    "/sys/devices/system/cpu/cpu${i?}/cpufreq/scaling_governor"; \
done

echo "CPU info (after changing governor):"
echo 'cpu\tgovernor\tcur\tmin\tmax'
echo "------------------------------------------------"
for i in `cat /sys/devices/system/cpu/present | tr '-' ' ' | xargs seq`; do \
    echo "cpu${i}" | paste \
      - \
      "/sys/devices/system/cpu/cpu${i}/cpufreq/scaling_governor" \
      "/sys/devices/system/cpu/cpu${i}/cpufreq/cpuinfo_cur_freq" \
      "/sys/devices/system/cpu/cpu${i}/cpufreq/cpuinfo_min_freq" \
      "/sys/devices/system/cpu/cpu${i}/cpufreq/cpuinfo_max_freq"; \
done
