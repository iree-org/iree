#!/bin/bash

# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Perform a best-effort attempt to cancel a running workflow (started before
# this script is run) and wait for the cancellation to complete.
#
# Due to limitations in GitHub's APIs, there are race conditions cases that
# this script can't be sure if the run is cancelled. It will detect and fail
# fast in such cases.

set -euo pipefail

MAX_WAIT_IN_SECS="${IREE_MAX_WAIT_IN_SECS:-180}"
WORKFLOW_RUN_URL="${1:-${IREE_WORKFLOW_RUN_URL}}"

RUN_ATTEMPT_BEFORE="$(gh api ${WORKFLOW_RUN_URL} \
  | jq --raw-output '.run_attempt')"

# Unfortunately Github API doesn't allow us to specify which run attempt to
# cancel. So there is a chance that we cancel a newer run attempt.
CANCEL_RESPONSE="$(gh api ${WORKFLOW_RUN_URL}/cancel --method POST \
  | jq --raw-output '.message' \
)" || true

if [[ "${CANCEL_RESPONSE}" == "Cannot cancel a workflow run that is completed." ]]; then
  exit 0
elif [[ "${CANCEL_RESPONSE}" != "null" ]]; then
  echo "Failed to cancel the workflow: ${CANCEL_RESPONSE}"
  exit 1
fi

RUN_ATTEMPT_AFTER="$(gh api ${WORKFLOW_RUN_URL} \
  | jq --raw-output '.run_attempt')"
# If the run attempt is changed after the cancel request, We don't know whether
# ${RUN_ATTEMPT_BEFORE} or ${RUN_ATTEMPT_AFTER} is cancelled. Instead of waiting
# on the wrong run attempt potentially and timeout, fail fast in this case.
if [[ "${RUN_ATTEMPT_BEFORE}" != "${RUN_ATTEMPT_AFTER}" ]]; then
  echo "Uncertain on which run attempt we cancelled."
  exit 1
fi

START_TIME="$(date +%s)"
while true; do
  # Wait on the specific run attempt so we won't wait on a new attempt that was
  # not cancelled.
  WORKFLOW_STATUS="$(gh api ${WORKFLOW_RUN_URL}/attempts/${RUN_ATTEMPT_BEFORE} \
    | jq --raw-output '.status')"
  echo "Waiting for the workflow to stop: ${WORKFLOW_STATUS}."
  if [[ "${WORKFLOW_STATUS}" == "completed" ]]; then
    break
  fi
  if (( "$(date +%s)" - "${START_TIME}" > "${MAX_WAIT_IN_SECS}" )); then
    echo "Timeout on waiting for the workflow."
    exit 1
  fi
  sleep 10
done
