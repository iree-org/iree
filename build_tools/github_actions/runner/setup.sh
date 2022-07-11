#!/bin/bash

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Sets up a GitHub actions runner VM. Requires custom attributes to indicate
# configuration settings and that OS inventory management be set up to enable
# querying host properties (see
# https://cloud.google.com/compute/docs/instances/os-inventory-management).
# It also still requires being passed a runner token (which will go away with
# the introduction

set -euo pipefail

token="$1"

cd actions-runner

get_metadata() {
  local url="http://metadata.google.internal/computeMetadata/v1/${1}"
  ret=0
  curl "${url}" \
    --silent --fail --show-error \
    --header "Metadata-Flavor: Google" || ret=$?
  if [[ $ret != 0 ]]; then
    echo "Failed fetching ${url}" >&2
    return ${ret}
  fi
}

get_os_info() {
  get_metadata "instance/guest-attributes/guestInventory/${1}"
}

get_attribute() {
  get_metadata "instance/attributes/${1}"
}

OS_ID="$(get_os_info ShortName)"
OS_VERSION="$(get_os_info Version)"
KERNEL_RELEASE="$(get_os_info KernelRelease)"
HOSTNAME="$(get_os_info Hostname)"
ARCH="$(get_os_info Architecture)"
ARCH="${ARCH^^}"

kernel="$(uname -s)"

# Matches default nomenclature used by GitHub
case "${kernel^^}" in
    LINUX*)
      OS_FAMILY="Linux";;
    DARWIN*)
      OS_FAMILY="macOS";;
    CYGWIN*|MINGW*)
      OS_FAMILY="Windows";;
    *)
      echo "Did not recognize output of 'uname -s': ${kernel}"
      exit 1
      ;;
esac

if [[ "${ARCH}" == "X86_64" ]]; then
  ARCH="X64" # This is the nomenclature GitHub uses
fi

ZONE="$(get_metadata instance/zone | awk -F/ '{print $NF}')"
CPU_PLATFORM="$(get_metadata instance/cpu-platform)"
MACHINE_TYPE="$(get_metadata instance/machine-type | awk -F/ '{print $NF}')"

RUNNER_GROUP="$(get_attribute github-runner-group)"
RUNNER_TRUST="$(get_attribute github-runner-trust)"
# Things like "Linux" need to be in here because we don't have
RUNNER_CUSTOM_LABELS="$(get_attribute github-runner-labels)"

declare -a RUNNER_LABELS_ARRAY=(
  "os-family=${OS_FAMILY}"
  "arch=${ARCH}"
  # Also as just raw labels, to match GitHub default behavior
  "${OS_FAMILY}"
  "${ARCH}"
  "hostname=${HOSTNAME}"
  "runner-group=${RUNNER_GROUP}"
  "trust=${RUNNER_TRUST}"
  "zone=${ZONE}"
  "cpu-platform=${CPU_PLATFORM}"
  "machine-type=${MACHINE_TYPE}"
  "os=${OS_ID}"
  "os-version=${OS_VERSION}"
  "kernel-release=${KERNEL_RELEASE}"
)

RUNNER_LABELS="$(IFS="," ; echo "${RUNNER_LABELS_ARRAY[*]}")"
# Append custom labels, taking care to only add a comma if there are any
RUNNER_LABELS="${RUNNER_LABELS}${RUNNER_CUSTOM_LABELS:+,${RUNNER_CUSTOM_LABELS}}"

set -x
./config.sh \
  --url https://github.com/iree-org \
  --token "${token}" \
  --labels "${RUNNER_LABELS}" \
  --unattended \
  --runnergroup "${RUNNER_GROUP}"
sudo ./svc.sh install
sudo ./svc.sh start
