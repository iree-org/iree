#!/bin/bash

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Registers GitHub actions runner VM. Requires custom attributes to indicate
# configuration settings and uses a proxy service to obtain runner registration
# tokens (https://github.com/google-github-actions/github-runner-token-proxy).

set -xeuo pipefail

source /runner-root/config/functions.sh

# These use OS inventory management to fetch information about the VM operating
# system (https://cloud.google.com/compute/docs/instances/os-inventory-management).
# For some reason, querying these at startup is unreliable. It seems like the
# guestInventory attributes take a really long time to be populated.
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

RUNNER_TYPE="$(get_attribute github-runner-type)"
RUNNER_GROUP="$(get_attribute github-runner-group)"
RUNNER_SCOPE="$(get_attribute github-runner-scope)"
RUNNER_TRUST="$(get_attribute github-runner-trust)"
RUNNER_VERSION="$(get_attribute github-runner-version)"
TOKEN_PROXY_URL="$(get_attribute github-token-proxy-url)"
CONFIG_REF="$(get_attribute github-runner-config-ref)"

# So this is a bit of a hack, but it enables us to use the same instance
# template regardless of environment. This means we can test a template by
# deploying it to the testing environment and then promote *the same template*
# to the prod environment. Otherwise it's difficult to tell the mapping between
# prod and testing templates. Ideally, this would be explicit metadata that was
# dynamic with the instance group itself, but instance groups don't have
# anything like that and all the metadata on the instances has to be specified
# in the templates.
RUNNER_ENVIRONMENT="prod"
if [[ "${HOSTNAME}" == *-testing-* ]]; then
  RUNNER_ENVIRONMENT="testing"
fi


declare -a RUNNER_LABELS_ARRAY=(
  "os-family=${OS_FAMILY}"
  # Also as just a raw label, to match GitHub default behavior
  "${OS_FAMILY}"
  "hostname=${HOSTNAME}"
  "runner-group=${RUNNER_GROUP}"
  "runner-version=${RUNNER_VERSION}"
  "trust=${RUNNER_TRUST}"
  "environment=${RUNNER_ENVIRONMENT}"
  "zone=${ZONE}"
  "cpu-platform=${CPU_PLATFORM}"
  "machine-type=${MACHINE_TYPE}"
  "config-ref=${CONFIG_REF}"
  "${RUNNER_TYPE}"
  # These labels require guest attributes. See note above.
  # "arch=${ARCH}"
  # "${ARCH}"
  # "os=${OS_ID}"
  # "os-version=${OS_VERSION}"
  # "kernel-release=${KERNEL_RELEASE}"
)

RUNNER_LABELS="$(IFS="," ; echo "${RUNNER_LABELS_ARRAY[*]}")"

INSTANCE_ID="$(get_metadata instance/id)"
GOOGLE_CLOUD_PROJECT="$(get_metadata project/project-id)"

set +x
GOOGLE_CLOUD_RUN_ID_TOKEN="$(get_metadata "instance/service-accounts/default/identity?audience=${TOKEN_PROXY_URL}")"

set +e
REGISTER_TOKEN="$(get_runner_token register ${RUNNER_SCOPE})"

if [ -z "${REGISTER_TOKEN}" ]; then
  echo "failed to get registration runner token" >&2
  exit 1
fi
set -xe

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

set +x
/runner-root/actions-runner/config.sh --token "${REGISTER_TOKEN}" "${args[@]}"
