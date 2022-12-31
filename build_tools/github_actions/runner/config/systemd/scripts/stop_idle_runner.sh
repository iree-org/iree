#!/bin/bash

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Calls the instance deletion proxy to check if the autoscaler recommends
# scaling down and if that's consistent, stop the runner.

set -euo pipefail

source /runner-root/config/functions.sh

SLEEP_MAX=30

function get_self_status() {
  nice_curl localhost:8080 || ret=$?
  if (( ret!=0 )); then
    echo "failed"
  fi
}

function should_scale_down() {
  local self_deletion_service_url="$(get_attribute instance-self-deleter-url)"
  local id_token=$(get_metadata "instance/service-accounts/default/identity?audience=${self_deletion_service_url}&format=full")

  nice_curl -X GET --header "Authorization: Bearer ${id_token}" "${self_deletion_service_url}"
}

function maybe_stop_runner() {
  if [[ "$(get_self_status)" != "idle" ]]; then
    return 0
  fi

  if [[ "$(should_scale_down)" != "true" ]]; then
    return 0
  fi

  # Sleep for a random interval. This is to avoid runners all shutting
  # themselves down at once since waiting for the MIG to be stable will
  # synchronize them.
  sleep "$((${RANDOM} % ${SLEEP_MAX}))"

  # Now repeat our queries. In reverse order because the second one is faster
  # and it's more important that it be correct before we proceed because we
  # definitely don't want to stop a runner that's in the middle of a job. We did
  # the self status check first before because it's much faster.
  if [[ "$(should_scale_down)" != "true" ]]; then
    return 0
  fi

  if [[ "$(get_self_status)" != "idle" ]]; then
    return 0
  fi

  systemctl stop gh-runner
}

maybe_stop_runner
