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

function get_latest_green() {
  local -a query_params=(
    branch=main
    event=push
    status=success
    exclude_pull_requests=true
  )

  while read commit; do
    local -a local_query_params=("${query_params[@]}" "head_sha=${commit}")
    # String join on comma
    query_string="$(IFS="&" ; echo "${local_query_params[*]}")"

    successful_runs="$(gh api --jq '.total_count' "/repos/iree-org/iree/actions/workflows/ci.yml/runs?${query_string}")"

    if (( successful_runs!=0 )); then
      echo "${commit}"
      return 0
    fi
  done < <(git rev-list --first-parent "${commitish}")
  echo "Failed to find green commit" >&2
  return 1
}

get_latest_green
