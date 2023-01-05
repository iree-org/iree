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

function get_runner_status() {
  nice_curl localhost:8080; ret=$?
  if (( ret!=0 )); then
    echo "failed"
  fi
}

function get_should_scale_down() {
  local self_deletion_service_url="$(get_attribute instance-self-deleter-url)"
  local id_token=$(get_metadata "instance/service-accounts/default/identity?audience=${self_deletion_service_url}&format=full")

  nice_curl -X GET --header "Authorization: Bearer ${id_token}" "${self_deletion_service_url}"
}

function maybe_stop_runner() {
  echo "Checking runner status"
  local runner_status="$(get_runner_status)"
  echo "runner_status='${runner_status}'"
  if [[ "${runner_status}" != "idle" ]]; then
    echo "Exiting"
    return 0
  fi
  echo "Proceeding"

  echo "Checking MIG autoscaling status. This could take a while as it waits" \
       "for the MIG to stabilize."
  local should_scale_down="$(get_should_scale_down)"
  echo "should_scale_down='${should_scale_down}'"
  if [[ "${should_scale_down}" != "true" ]]; then
    echo "Exiting"
    return 0
  fi
  echo "Proceeding"

  # Double check that the runner is still idle. The above call can take a while
  # as it waits random intervals for the MIG to be become stable. We definitely
  # don't want to stop a runner that's in the middle of a job. We did the runner
  # status check first before because it's much faster and an easy chance to
  # bail out.
  echo "Rechecking runner status"
  local runner_status="$(get_runner_status)"
  echo "runner_status='${runner_status}'"
  if [[ "${runner_status}" != "idle" ]]; then
    echo "Exiting"
    return 0
  fi
  echo "Stopping runner"
  systemctl stop gh-runner
}

maybe_stop_runner
