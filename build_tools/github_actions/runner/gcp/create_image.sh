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



SUCCESS_DELETE_INSTANCE=1
FAILURE_DELETE_INSTANCE=0

RUNNER_TYPE="${RUNNER_TYPE:-cpu}"
RUNNER_TYPE="${RUNNER_TYPE,,}"

TIME_STRING="$(date +%Y-%m-%d-%s)"
INSTANCE_NAME="${INSTANCE_NAME:-github-runner-template-${RUNNER_TYPE}-${TIME_STRING}}"
IMAGE_NAME="${INSTANCE_NAME/-template/}"
ZONE="${ZONE:-us-central1-a}"
PROJECT=iree-oss

case "${RUNNER_TYPE}" in
  arm64)
    BASE_IMAGE_ARCH="-arm64"
    ;;
  *)
    BASE_IMAGE_ARCH=""
    ;;
esac
BASE_IMAGE="${BASE_IMAGE:-projects/ubuntu-os-cloud/global/images/ubuntu-2204-jammy${BASE_IMAGE_ARCH}-v20230727}"

# We create the image using n1 machines with attached T4 GPUs. This image works
# for the A100 machines as well though.
GPU_MACHINE_TYPE="n1-standard-16"
X86_64_MACHINE_TYPE="e2-medium"
ARM64_MACHINE_TYPE="t2a-standard-8"
CPU_IMAGE_SIZE_GB=10
# We need enough space to fetch Docker images that we test with
# TODO(gcmn): See if we can make the image smaller, e.g. by resizing after setup
# or using a local ssd for scratch space during setup.
GPU_IMAGE_SIZE_GB=100

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

SSH_CMD=(
  gcloud
  compute
  ssh
  "${INSTANCE_NAME}"
  --zone="${ZONE}"
  --no-user-output-enabled
)

function cleanup_reminder() {
  echo "You can ssh in to debug with the following command:"
  echo "${SSH_CMD[@]}"
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

function get_ip() {
  gcloud compute instances describe \
    "${INSTANCE_NAME}" \
    --zone="${ZONE}" \
    --format='value(networkInterfaces[0].accessConfigs[0].ip)'
}

function instance_ssh() {
  gcloud compute ssh "${INSTANCE_NAME}" --zone="${ZONE}" \
      --no-user-output-enabled \
      "$@"
}

function ssh_ping() {
  # ssh with a no-op command
  instance_ssh --command=":"
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
  if gcloud compute instances describe "${INSTANCE_NAME}" --zone="${ZONE}" > /dev/null 2>&1; then
    echo "Using existing instance '${INSTANCE_NAME}'"
  else
    echo "Creating instance '${INSTANCE_NAME}' for boot disk"
    case "${RUNNER_TYPE}" in
      cpu)
        local machine_type="${X86_64_MACHINE_TYPE}"
        local image_size_gb="${CPU_IMAGE_SIZE_GB}"
        local maintenance_policy=MIGRATE
        local -a extra_args=()
        ;;
      arm64)
        local machine_type="${ARM64_MACHINE_TYPE}"
        local image_size_gb="${CPU_IMAGE_SIZE_GB}"
        local maintenance_policy=MIGRATE
        local -a extra_args=()
        ;;
      gpu)
        local machine_type="${GPU_MACHINE_TYPE}"
        local image_size_gb="${GPU_IMAGE_SIZE_GB}"
        local maintenance_policy=TERMINATE
        local -a extra_args=("--accelerator=count=1,type=nvidia-tesla-t4")
        ;;
      *)
        echo "Unrecognized RUNNER_TYPE=${RUNNER_TYPE}"
        exit 1
        ;;
    esac

    local -a create_instance_cmd=(
      gcloud
      compute
      instances
      create
      "${INSTANCE_NAME}"
      --project=iree-oss
      --zone="${ZONE}"
      # `address=''` indicates an ephemeral IP. This *shouldn't* be necessary here,
      # as the gcloud docs say that this is the default, but in fact if you leave it
      # off the VM gets no external IP and is impossible to SSH into. This knowledge
      # was hard won.
      --network-interface=network=default,address='',network-tier=PREMIUM
      --provisioning-model=STANDARD
      --no-service-account
      --no-scopes
      --no-shielded-secure-boot
      --shielded-vtpm
      --shielded-integrity-monitoring
      --reservation-affinity=any
      --metadata-from-file=startup-script="${SCRIPT_DIR}/image_setup.sh"
      --maintenance-policy="${maintenance_policy}"
      --metadata="github-runner-type=${RUNNER_TYPE}"
      --machine-type="${machine_type}"
      --create-disk="boot=yes,device-name=${INSTANCE_NAME},image=${BASE_IMAGE},mode=rw,size=${image_size_gb},type=projects/${PROJECT}/zones/${ZONE}/diskTypes/pd-balanced,auto-delete=yes"
      "${extra_args[@]}"
    )

    (set -x; "${create_instance_cmd[@]}")
  fi

  echo "Waiting for instance to start up"
  # We could only use the ssh check below, but it's much nicer to know why an
  # an instance isn't responsive and this is something we can check first.
  wait_for_ip "${MAX_IP_ATTEMPTS}"
  wait_for_ssh "${MAX_SSH_ATTEMPTS}"


  echo ""
  local log_file="$(mktemp --tmpdir ${INSTANCE_NAME}.XXX.startup.log)"
  echo "Streaming startup logs from instance to stdout and ${log_file}"

  # Get the PID of the startup script
  local startup_pid="$(instance_ssh --command='systemctl show --property=ExecMainPID --value google-startup-scripts')"

  echo ""
  echo "*******************"

  # -t forces a pseudo-tty which allows us to run tail with a follow
  gcloud compute ssh "${INSTANCE_NAME}" --zone="${ZONE}" \
      --no-user-output-enabled --ssh-flag="-t" \
      --command="tail --follow=name --retry --lines=+1 --pid=${startup_pid} /startup.log" \
      | tee "${log_file}"

  echo "*******************"
  echo ""

  local exit_code="$(instance_ssh --command="cat /startup-exit.txt")"

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
  instance_ssh --command="sudo rm /startup.log"

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
