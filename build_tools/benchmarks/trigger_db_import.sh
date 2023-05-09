#!/bin/bash

# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Triggers the cloud function which imports the benchmark results from the given
# GitHub Actions run into the database.
#
# Usage: trigger_db_import.sh <importer> <github_run_id> <github_run_attempt>
#
# - <importer> is either 'presubmit' or 'postsubmit'
# - <github_run_id> is an integer identifier
# - <github_run_attempt> is an integer identifier
#
# /<github_run_id>/<github_run_attempt>/ is the path in the bucket where the cloud function is
# looking for the benchmark results. The bucket is determined by the <importer> option.
# Note, that you can't trigger an import from an arbitrary bucket for security reasons.

set -euo pipefail

readonly SCRIPT_DIR="$(dirname -- "$( readlink -f -- "$0"; )")"
readonly REPO_ROOT="${SCRIPT_DIR}/../../"
readonly WORKFLOW="${1}"
readonly GITHUB_RUN_ID="${2}"
readonly GITHUB_RUN_ATTEMPT="${3}"
readonly CLOUD_FUNCTION_URL="https://oobi-${WORKFLOW}-importer-zbhz5clunq-uc.a.run.app"

source "${REPO_ROOT}/build_tools/github_actions/runner/config/functions.sh"

function usage {
  echo "Usage: $0 <importer> <github_run_id> <github_run_attempt>" >&2
  echo "" >&2
  echo "- <importer> is either 'presubmit' or 'postsubmit'" >&2
  echo "- <github_run_id> is an integer identifier" >&2
  echo "- <github_run_attempt> is an integer identifier" >&2
  exit -1
}

if [[ $# -ne 3 ]]; then
  echo -e "Expected 3 arguments, found $#.\n" >&2
  usage
fi

if [[ "${WORKFLOW}" != "presubmit" && "${WORKFLOW}" != "postsubmit" ]]; then
  echo -e "<importer> can be either 'presubmit' or 'postsubmit'.\n" >&2
  usage
fi

if ! [[ "${GITHUB_RUN_ID}" -gt 0 ]] 2>/dev/null; then
  echo -e "<github_run_id> must be a positive integer!\n" >&2
  usage
fi

if ! [[ "${GITHUB_RUN_ATTEMPT}" -gt 0 ]] 2>/dev/null; then
  echo -e "<github_run_attempt> must be a positive integer!\n" >&2
  usage
fi

if [[ -v GITHUB_ACTIONS ]]; then
  # This means we run as part of a GitHub Actions workflow and will get our ID token through the runner's service account
  readonly ID_TOKEN="$(get_metadata "instance/service-accounts/default/identity?audience=${CLOUD_FUNCTION_URL}&format=full")"
else
  # If we are not running on the CI, we will use gcloud to get a user-account ID token.
  readonly ID_TOKEN="$(gcloud auth print-identity-token)"
fi

exec curl -sSLX POST \
  -H "Authorization: Bearer ${ID_TOKEN}" \
  -H "Content-Type: application/json" \
  -d "{ \"github_run_id\" : \"${GITHUB_RUN_ID}\", \"github_run_attempt\" : \"${GITHUB_RUN_ATTEMPT}\"}" \
  "${CLOUD_FUNCTION_URL}"
