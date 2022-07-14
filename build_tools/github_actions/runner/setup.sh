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
# the introduction of a proxy service for obtaining registration tokens).

set -euo pipefail

SCRIPT_DIR="$(dirname -- "$( readlink -f -- "$0"; )")";

source "${SCRIPT_DIR}/functions.sh"

cd actions-runner

GITHUB_TOKEN_PROXY_URL="$(get_attribute github-token-proxy-url)"
GITHUB_RUNNER_SCOPE="$(get_attribute github-runner-scope)"

GOOGLE_CLOUD_RUN_ID_TOKEN=$(get_metadata "instance/service-accounts/default/identity?audience=${GITHUB_TOKEN_PROXY_URL}")
GITHUB_REGISTRATION_TOKEN="$(curl -sSfL "${GITHUB_TOKEN_PROXY_URL}/register" \
  --header "Authorization: Bearer ${GOOGLE_CLOUD_RUN_ID_TOKEN}" \
  --data-binary "{\"scope\": \"${GITHUB_RUNNER_SCOPE}\"}" \
  | jq -r ".token"
)"

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

declare -a args=(
  --url "https://github.com/${GITHUB_RUNNER_SCOPE}"
  --labels "${RUNNER_LABELS}"
  --unattended
  --runnergroup "${RUNNER_GROUP}"
  --replace
)
# I would love to discover another way to print an array while preserving quote
# escaping. We're not just using `set -x` on the command itself because we don't
# want to leak the token (even if it's immediately invalidated, still best not
# to). `:` is the bash noop command that is equivalent to `true`.
(set -x; : Running configuration with additional args: "${args[@]}")

./config.sh --token "${GITHUB_REGISTRATION_TOKEN}" "${args[@]}"
sudo ./svc.sh install
sudo ./svc.sh start
