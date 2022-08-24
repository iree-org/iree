#!/bin/bash

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Registers GitHub actions runner VM. Requires custom attributes to indicate
# configuration settings and uses a proxy service to obtain runner registration
# tokens (https://github.com/google-github-actions/github-runner-token-proxy).

set -euo pipefail

SCRIPT_DIR="$(dirname -- "$( readlink -f -- "$0"; )")";
source "${SCRIPT_DIR}/functions.sh"

# These use OS inventory management to fetch information about the VM operating
# system (https://cloud.google.com/compute/docs/instances/os-inventory-management).
# For some reason, querying these at startup is unreliable. It seems like the
# guestInventory attributes take a really long time to be populated. For now,
# anything in here we care about needs to be injected via the
# `github-runner-labels` custom metadata attribute.
# OS_ID="$(get_os_info ShortName)"
# OS_VERSION="$(get_os_info Version)"
# KERNEL_RELEASE="$(get_os_info KernelRelease)"
# HOSTNAME="$(get_os_info Hostname)"
# ARCH="$(get_os_info Architecture)"
# ARCH="${ARCH^^}"
# if [[ "${ARCH}" == "X86_64" ]]; then
#   ARCH="X64" # This is the nomenclature GitHub uses
# fi

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

HOSTNAME="$(get_metadata instance/name)"
ZONE="$(get_metadata instance/zone | awk -F/ '{print $NF}')"
CPU_PLATFORM="$(get_metadata instance/cpu-platform)"
MACHINE_TYPE="$(get_metadata instance/machine-type | awk -F/ '{print $NF}')"

RUNNER_CUSTOM_LABELS="$(get_attribute github-runner-labels)"
RUNNER_GROUP="$(get_attribute github-runner-group)"
RUNNER_SCOPE="$(get_attribute github-runner-scope)"
RUNNER_TRUST="$(get_attribute github-runner-trust)"
RUNNER_VERSION="$(get_attribute github-runner-version)"
TOKEN_PROXY_URL="$(get_attribute github-token-proxy-url)"
CONFIG_REF="$(get_attribute github-runner-config-ref)"

declare -a RUNNER_LABELS_ARRAY=(
  "os-family=${OS_FAMILY}"
  # Also as just a raw label, to match GitHub default behavior
  "${OS_FAMILY}"
  "hostname=${HOSTNAME}"
  "runner-group=${RUNNER_GROUP}"
  "runner-version=${RUNNER_VERSION}"
  "trust=${RUNNER_TRUST}"
  "zone=${ZONE}"
  "cpu-platform=${CPU_PLATFORM}"
  "machine-type=${MACHINE_TYPE}"
  "config-ref=${CONFIG_REF}"
  # These labels require guest attributes. See note above.
  # "arch=${ARCH}"
  # "${ARCH}"
  # "os=${OS_ID}"
  # "os-version=${OS_VERSION}"
  # "kernel-release=${KERNEL_RELEASE}"
)

# Yes this is kind of a hack, but if we inject this with any metadata, then it's
# part of the instance template and we can't use the same instance template for
# prod. The name is the only thing that's controlled by the instance group.
if [[ "${HOSTNAME}" == *testing* ]]; then
  RUNNER_LABELS+=(testing)
else
  RUNNER_LABELS+=(prod)
fi

RUNNER_LABELS="$(IFS="," ; echo "${RUNNER_LABELS_ARRAY[*]}")"
# Append custom labels, taking care to only add a comma if there are any
RUNNER_LABELS="${RUNNER_LABELS}${RUNNER_CUSTOM_LABELS:+,${RUNNER_CUSTOM_LABELS}}"

INSTANCE_ID="$(get_metadata instance/id)"
GOOGLE_CLOUD_PROJECT="$(get_metadata project/project-id)"

GOOGLE_CLOUD_RUN_ID_TOKEN="$(get_metadata "instance/service-accounts/default/identity?audience=${TOKEN_PROXY_URL}")"

REGISTER_TOKEN="$(get_token register ${RUNNER_SCOPE})"

if [ -z "${REGISTER_TOKEN}" ]; then
  echo "failed to get registration runner token" >&2
  exit 1
fi

declare -a args=(
  --unattended \
  # Shut down after completing a single job
  --ephemeral \
  # We don't immediately update each time we start. We handle our own
  # updates instead.
  --disableupdate \
  --url "https://github.com/${RUNNER_SCOPE}" \
  --name "${HOSTNAME}" \
  # If we end up with name conflicts, just replace the old entry.
  --replace \
  --runnergroup "${RUNNER_GROUP}" \
  --labels "${RUNNER_LABELS}"
)
# I would love to discover another way to print an array while preserving quote
# escaping. We're not just using `set -x` on the command itself because we don't
# want to leak the token (even if it's immediately invalidated, still best not
# to). `:` is the bash noop command that is equivalent to `true` so we are
# "running" a command exclusively to print it using the shell's builtin
# functionality.
(set -x; : Running configuration with additional args: "${args[@]}")

/home/runner/actions-runner/config.sh --token "${REGISTER_TOKEN}" "${args[@]}"
