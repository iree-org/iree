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

DRY_RUN="${DRY_RUN:-0}"

GROUP="$1"
TYPE="$2"
REGION="$3"
# No the double dash isn't a typo. --lines=-0 tells `head` to print all but the
# last 0 lines, AKA all the lines. You can use this optional argument to limit
# the number of runners removed at a time.
COUNT="${4:--0}"

MIG="github-runner-${GROUP}-${TYPE}-${REGION}"


# Give VMs 5 minutes to startup and register with GitHub. This is pretty generous
STARTUP_DURATION="$(( 60 * 5 ))"
CREATION_CUTOFF="$(( "$(date +%s)" - STARTUP_DURATION ))"

function get_online_runners() {
  gh api --paginate -H "Accept: application/vnd.github+json" \
      /orgs/openxla/actions/runners?per_page=100 \
    | jq --raw-output \
      ".runners | .[]
      | select(
          (.labels | .[] | select(.name==\"self-hosted\"))
          and .status==\"online\"
      ) | .name | select(. | startswith(\"${MIG}\"))"
}

function get_mig_instances() {
  # The interface for listing via instance groups is way more limited, and in
  # particular doesn't give us creation_timestamp. This interface doesn't give us
  # in_use_by (the MIG that owns the VM) even though that's in the cloud console.
  # So we just use the naming conventions. This isn't quite as bad as it seems
  # because in the final delete call we will be doing it through the MIG API.
  gcloud compute instances list --format="value(name)" \
    --filter="name~${MIG}-[a-z0-9]+ AND creation_timestamp.date(+%s)<\"${CREATION_CUTOFF}\""
}

function remove_stuck_runners() {
  local -a instances="$(get_mig_instances)"

  if (( "${#instances[@]}" == 0 )) ; then
    echo "Found no instances for '${MIG}'"
    exit 0
  fi

  local -a runners="$(get_online_runners)"

  # assymetric set difference instances - runners
  # see https://stackoverflow.com/q/2312762
  local -a to_delete=($(echo "${instances[@]}" "${runners[@]}" "${runners[@]}" | tr ' ' '\n' | sort | uniq -u))


  if (( "${#to_delete[@]}" == 0 )) ; then
    echo "Found no stuck instances for '${MIG}'"
    exit 0
  fi

  echo "Deleting ${#to_delete[@]} instances: "
  printf "%s\n" "${to_delete[@]}"

  local to_delete_string="$(IFS="," ; echo "${to_delete[*]}")"
  local -a cmd=(
    gcloud compute instance-groups managed delete-instances
    "${MIG}"
    --instances="${to_delete_string}"
    --region="${REGION}"
  )

  if (( DRY_RUN != 0 )); then
    # prepend cmd with a noop
    cmd=(: "${cmd[@]}")
  fi

  (
    set -x
    "${cmd[@]}"
  )
}

remove_stuck_runners
