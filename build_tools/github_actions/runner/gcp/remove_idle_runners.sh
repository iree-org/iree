#!/bin/bash

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Due to limitations in MIGs, we quite frequently want to manually bring down
# instances that aren't currently running a job. This script automates that.
# Note that there is totally the potential for a race here: we could identify
# the runner as idle and it could pick up a job before its deletion causes it to
# deregister on shutdown. This is also a (bigger) problem with the manual
# removal that this script automates.

set -euo pipefail

GROUP="$1"
TYPE="$2"
REGION="$3"
# No the double dash isn't a typo. --lines=-0 tells `head` to print all but the
# last 0 lines, AKA all the lines. You can use this optional argument to limit
# the number of runners removed at a time.
COUNT="${4:--0}"

MIG="github-runner-${GROUP}-${TYPE}-${REGION}"

function remove_idle_runners() {
  local -a to_delete=($(gh api --paginate -H "Accept: application/vnd.github+json" \
      /orgs/openxla/actions/runners?per_page=100 \
    | jq --raw-output \
      ".runners | .[]
      | select(
          (.labels | .[] | select(.name==\"self-hosted\"))
          and .status==\"online\"
          and .busy==false
      ) | .name | select(. | startswith(\"${MIG}\"))" \
      | head --lines="${COUNT}"))

  if (( "${#to_delete[@]}" == 0 )) ; then
    echo "Found no idle runners for '${MIG}'"
    exit 0
  fi

  echo "Deleting ${#to_delete[@]} instances: "
  printf "%s\n" "${to_delete[@]}"

  local to_delete_string="$(IFS="," ; echo "${to_delete[*]}")"

  (
    set -x
    gcloud compute instance-groups managed delete-instances "${MIG}" \
        --instances="${to_delete_string}" --region="${REGION}"
  )
}

remove_idle_runners
