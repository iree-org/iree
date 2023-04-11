#!/bin/bash

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Finds the most recent ancestor (first parents only) of the given commit
# (default HEAD) for which the ci workflow passed. There are GitHub actions that
# allegedly do this, but they mostly seem to be broken, unlicensed, or both.
# This approach iterates through commits in the history and makes a separate
# call for each. The alternative is collecting all the successful runs and then
# sorting them. Empirically, the latter was much slower. Fetching them already
# sorted does not appear to be an option as GitHub does not document any known
# sort order. Fetching for multiple specific commits at once is similarly not an
# option.

set -euo pipefail

commitish="${1:-HEAD}"

declare -r REQUIRED_WORKFLOWS=(ci.yml)
declare -ar QUERY_PARAMS=(
  branch=main
  event=push
  status=success
  exclude_pull_requests=true
)

function get_latest_green() {

  while read commit; do
    local -a query_params=("${QUERY_PARAMS[@]}" "head_sha=${commit}")
    # String join on ampersand
    local query_string="$(IFS="&" ; echo "${query_params[*]}")"

    local all_passing="true"
    for workflow in "${REQUIRED_WORKFLOWS[@]}"; do
      local successful_run_count="$(\
        gh api --jq '.total_count' \
        "/repos/openxla/iree/actions/workflows/${workflow}/runs?${query_string}" \
      )"
      # Any successful run of the workflow (including reruns) is OK.
      if (( successful_run_count==0 )); then
        all_passing="false"
        break
      fi
    done
    if [[ "${all_passing}" == true ]]; then
      echo "${commit}"
      return 0
    fi
  done < <(git rev-list --first-parent "${commitish}")
  echo "Failed to find green commit" >&2
  return 1
}

get_latest_green
