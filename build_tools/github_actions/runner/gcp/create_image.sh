#!/bin/bash

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail

TIME_STRING="$(date +%Y-%m-%d-%s)"

INSTANCE_NAME="${INSTANCE_NAME:-github-runner-template-cpu-${TIME_STRING}}"
IMAGE_NAME="${IMAGE_NAME:-github-runner-cpu-${TIME_STRING}}"
ZONE="${ZONE:-us-central1-a}"
PROJECT=iree-oss
BASE_IMAGE="${BASE_IMAGE:-projects/ubuntu-os-cloud/global/images/ubuntu-2204-jammy-v20220902}"
# It takes a little bit to bring up ssh on the instance. I haven't found a
# better way to wait for this than just polling.
MAX_IP_ATTEMPTS=5
MAX_SSH_ATTEMPTS=10
MAX_SCP_ATTEMPTS=5

SCRIPT_DIR="$(dirname -- "$( readlink -f -- "$0"; )")";

CREATE_INSTANCE_ARGS=(
  "${INSTANCE_NAME}"
  --project=iree-oss
  --zone="${ZONE}"
  --machine-type=e2-medium
  # `address=''` indicates an ephemeral IP. This *shouldn't* be necessary here,
  # as the gcloud docs say that this is the default, but in fact if you leave it
  # off the VM gets no external IP and is impossible to SSH into. This knowledge
  # was hard won.
  --network-interface=network=default,address='',network-tier=PREMIUM
  --maintenance-policy=MIGRATE
  --provisioning-model=STANDARD
  --no-service-account
  --no-scopes
  --create-disk="boot=yes,device-name=${INSTANCE_NAME},image=${BASE_IMAGE},mode=rw,size=10,type=projects/${PROJECT}/zones/${ZONE}/diskTypes/pd-balanced"
  --no-shielded-secure-boot
  --shielded-vtpm
  --shielded-integrity-monitoring
  --reservation-affinity=any
  --metadata-from-file=startup-script="${SCRIPT_DIR}/image_setup.sh"
)

function get_ip() {
  gcloud compute instances describe \
    "${INSTANCE_NAME}" \
    --zone="${ZONE}" \
    --format='value(networkInterfaces[0].accessConfigs[0].ip)'
}

function ssh_ping() {
  gcloud compute ssh "${INSTANCE_NAME}" \
        --zone="${ZONE}" \
        --command=":"
}

function wait_for_ip() {
  local -i max_attempts="$1"
  local -i failed_attempts=0
  while (( failed_attempts <= max_attempts )) && [[ get_ip == "" ]]; do
    echo -n '.'
    failed_attempts="$(( failed_attempts+1 ))"
    sleep 1
  done

  if (( failed_attempts > max_attempts )); then
    echo "Instance was never assigned an external IP. Aborting"
    exit 1
  fi
}

function wait_for_ssh() {
  local -i max_attempts="$1"
  local -i failed_attempts=0
  local output=""
  while (( failed_attempts <= max_attempts )) && ! ssh_output="$(ssh_ping 2>&1)"; do
    echo -n '.'
    failed_attempts="$(( failed_attempts+1 ))"
    sleep 1
  done

  if (( failed_attempts > max_attempts )); then
    echo "Failed to connect to instance via ssh. Output from ssh command:"
    echo "${ssh_output}"
    exit 1
  fi
}

function create_image() {
  echo "Creating instance for boot disk"
  (set -x; gcloud compute instances create "${CREATE_INSTANCE_ARGS[@]}")

  # We could only use the ssh check below, but it's much nicer to know why an
  # an instance isn't responsive and this is something we can check first.
  echo "Waiting for instance to start up"
  wait_for_ip "${MAX_IP_ATTEMPTS}"
  wait_for_ssh "${MAX_SSH_ATTEMPTS}"

  local log_file="$(mktemp)"
  touch "${log_file}"

  echo ""
  echo "Streaming startup logs from instance"
  tail -f "${log_file}" &
  local -i failed_scp_attempts=0
  local last_line=""
  local scp_output=""
  # Is waiting for a certain line in the logs kind of hacky? yes
  # Is there a better way to do it? probably
  # Does the better way involve a bunch of fiddling about? also probably
  while (( failed_scp_attempts < MAX_SCP_ATTEMPTS )) && [[ "${last_line}" != "Setup complete" ]]; do
    ret=0
    scp_output="$(gcloud compute scp \
      --zone="${ZONE}" \
      "${INSTANCE_NAME}:/startup.log" \
      "${log_file}" 2>&1)" || ret=$?
    if (( ret != 0 )); then
      failed_scp_attempts="$(( failed_scp_attempts+1 ))"
      sleep 1
    else
      last_line="$(tail --lines=1 "${log_file}")"
    fi
  done

  if (( failed_scp_attempts >= MAX_SCP_ATTEMPTS )); then
    echo "Was unable to copy logs from instance. Output from scp:"
    echo "${scp_output}"
    exit 1
  fi

  if [[ "${last_line}" != "Setup complete" ]]; then
    echo "Instance did not complete its setup. Please check the logs above."
    exit 1
  fi

  echo "Startup finished successfully."

  echo "Deleting log file"
  gcloud compute ssh "${INSTANCE_NAME}" --zone="${ZONE}" \
    --no-user-output-enabled \
    --command="sudo rm /startup.log"

  echo "Shutting down instance"
  # This actually does things synchronously, so we don't need our own loop to
  # wait.
  gcloud compute instances stop "${INSTANCE_NAME}" --zone="${ZONE}"

  echo "Creating disk image"
  gcloud compute images create "${IMAGE_NAME}" \
    --source-disk="${INSTANCE_NAME}" \
    --source-disk-zone="${ZONE}"

  echo "Deleting instance"
  gcloud compute instances delete "${INSTANCE_NAME}" --zone="${ZONE}" --quiet

  echo "Successfully created image: ${IMAGE_NAME}"
}

create_image
