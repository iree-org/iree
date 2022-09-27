#!/bin/bash

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -o errexit   # Exit if any command fails
set -o errtrace  # make ERR trap inherit
set -o pipefail  # return error if any part of a pipe errors
set -o nounset   # error if an undefined variable is used

TIME_STRING="$(date +%Y-%m-%d-%s)"

SUCCESS_DELETE_INSTANCE=1
FAILURE_DELETE_INSTANCE=0

INSTANCE_NAME="${INSTANCE_NAME:-github-runner-template-cpu-${TIME_STRING}}"
IMAGE_NAME="${IMAGE_NAME:-github-runner-cpu-${TIME_STRING}}"
ZONE="${ZONE:-us-central1-a}"
PROJECT=iree-oss
BASE_IMAGE="${BASE_IMAGE:-projects/ubuntu-os-cloud/global/images/ubuntu-2204-jammy-v20220902}"
# The size of the base image
IMAGE_SIZE_GB=10
# It takes a little bit to bring up ssh on the instance. I haven't found a
# better way to wait for this than just polling.
MAX_IP_ATTEMPTS=5
MAX_SSH_ATTEMPTS=10
MAX_SCP_ATTEMPTS=5

DELETE_INSTANCE_CMD=(
  gcloud
  compute
  instances
  delete
  "${INSTANCE_NAME}"
  --zone="${ZONE}"
)

function cleanup_reminder() {
  echo "Make sure to delete ${INSTANCE_NAME} when you're done debugging:"
  echo "${DELETE_INSTANCE_CMD[@]}"
}

function failure_exit() {
  local exit_code="$?"
  trap - INT ERR EXIT
  if (( exit_code != 0 )); then
    echo "Image creation was not successful."
    if (( FAILURE_DELETE_INSTANCE==1 )); then
      echo "Attempting to delete instance ${INSTANCE_NAME}"
      "${DELETE_INSTANCE_CMD[@]}" --quiet
      exit "${exit_code}"
    else
      cleanup_reminder
    fi
  fi
  exit "${exit_code}"
}

trap failure_exit INT ERR EXIT

SCRIPT_DIR="$(dirname -- "$( readlink -f -- "$0"; )")";

CREATE_INSTANCE_CMD=(
  gcloud
  compute
  instances
  create
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
  --create-disk="boot=yes,device-name=${INSTANCE_NAME},image=${BASE_IMAGE},mode=rw,size=${IMAGE_SIZE_GB},type=projects/${PROJECT}/zones/${ZONE}/diskTypes/pd-balanced,auto-delete=yes"
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
  done

  if (( failed_attempts > max_attempts )); then
    echo "Failed to connect to instance via ssh. Output from ssh command:"
    echo "${ssh_output}"
    exit 1
  fi
}

function create_image() {
  echo "Creating instance for boot disk"
  (set -x; "${CREATE_INSTANCE_CMD[@]}")

  # We could only use the ssh check below, but it's much nicer to know why an
  # an instance isn't responsive and this is something we can check first.
  echo "Waiting for instance to start up"
  wait_for_ip "${MAX_IP_ATTEMPTS}"
  wait_for_ssh "${MAX_SSH_ATTEMPTS}"


  echo ""
  local log_file="$(mktemp --tmpdir ${INSTANCE_NAME}.XXX.startup.log)"
  echo "Streaming startup logs from instance to stdout and ${log_file}"

  # Get the PID of the startup script
  local startup_pid="$(gcloud compute ssh "${INSTANCE_NAME}" --zone="${ZONE}" \
      --no-user-output-enabled \
      --command='systemctl show --property=ExecMainPID --value google-startup-scripts')"

  echo ""
  echo "*******************"

  # -t forces a pseudo-tty which allows us to run tail with a follow
  gcloud compute ssh "${INSTANCE_NAME}" --zone="${ZONE}" \
      --no-user-output-enabled \
      --ssh-flag="-t" \
      --command="tail --follow=name --retry --pid=${startup_pid} /startup.log" \
      | tee "${log_file}"

  echo "*******************"
  echo ""

  local exit_code="$(gcloud compute ssh "${INSTANCE_NAME}" --command="cat /startup-exit.txt")"

  if [[ "${exit_code}" != +([0-9]) ]]; then
    echo "Failed to retrieve exit code from startup script (got '${exit_code}')."
    exit 1
  fi

  if (( exit_code != 0 )); then
    echo "Image setup failed with code '${exit_code}'. See logs above."
    exit "${exit_code}"
  fi

  echo "Startup finished successfully."

  echo "Deleting remote log file"
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

  if (( SUCCESS_DELETE_INSTANCE == 1 )); then
    echo "Deleting instance"
    "${DELETE_INSTANCE_CMD[@]}" --quiet
  else
    echo "Not deleting instance because SUCCESS_DELETE_INSTANCE=${SUCCESS_DELETE_INSTANCE}"
    cleanup_reminder
  fi

  echo "Successfully created image: ${IMAGE_NAME}"
}

create_image
