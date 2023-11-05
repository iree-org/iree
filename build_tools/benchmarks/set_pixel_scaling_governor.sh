#!/bin/sh

# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Runs on an android device itself to set the frequency scaling governor for Pixel devices.

set -euo pipefail

GOVERNOR="${1:-performance}"

DEV_FREQ_GOVERNOR_PATHS=(
  "/sys/devices/platform/17000010.devfreq_mif/devfreq/17000010.devfreq_mif/governor"
  "/sys/devices/platform/17000020.devfreq_int/devfreq/17000020.devfreq_int/governor"
  "/sys/devices/platform/17000090.devfreq_dsu/devfreq/17000090.devfreq_dsu/governor"
  "/sys/devices/platform/170000a0.devfreq_bci/devfreq/170000a0.devfreq_bci/governor"
)

print_status() {
  for governor_path in ${DEV_FREQ_GOVERNOR_PATHS[@]}; do
    if test -f "${governor_path}"; then
      status=$(cat "${governor_path}")
      echo "${governor_path}: ${status}"
    fi
  done
}

echo "Current settings:"
print_status

echo "Setting device frequency governor to ${GOVERNOR}"
for governor_path in ${DEV_FREQ_GOVERNOR_PATHS[@]}; do
  if test -f "${governor_path}"; then
    echo "${GOVERNOR}" > "${governor_path}"
  fi
done

echo "Settings now:"
print_status
