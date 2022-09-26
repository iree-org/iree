#!/bin/bash

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This is the series of commands run on the a VM from a fresh image in order to
# set up the disk to be used as a boot image. This script must be run as root.

set -o verbose   # Print all command lines literally as they are read
set -o xtrace    # Print all commands after they are expanded
set -o errexit   # Exit if any command fails
set -o errtrace  # make ERR trap inherit
set -o pipefail  # return error if any part of a pipe errors
set -o nounset   # error if an undefined variable is used


GCLOUD_VERSION=402.0.0
GCLOUD_ARCHIVE_DIGEST=a9902b57d4cba2ebb76d7354570813d3d8199c36b95a1111a1b7fea013beaaf9


function save_exit_code() {
  local exit_code="$?"
  echo "${exit_code}" > /startup-exit.txt
  trap - EXIT
  exit "${exit_code}"
}

trap save_exit_code EXIT INT TERM

function apt_maybe_purge() {
  # Remove and purge packages if they are installed and don't error if they're
  # not or if they're not findable in the ppa.
  local -a to_remove=()
  for pkg in "$@"; do
    ret=0
    if dpkg --status $pkg &> /dev/null; then
      to_remove+=("${pkg}")
    fi
  done
  if (( "${#to_remove[@]}" != 0 )); then
    apt-get remove --purge --autoremove "${to_remove[@]}"
  fi
}

function startup() {
  # Shut down in 5 hours. Makes sure this instance doesn't hang around forever
  # if setup fails. Someone can cancel the shutdown with `shutdown -c`.
  nohup shutdown -h +300 &
  cd /

  ########################### Create the runner user ###########################

  # GCE "helpfully" creates users for apparently any account that has ever
  # logged in on any VM. Delete it if it's there.
  userdel --force --remove runner || true
  adduser --system --group "runner"
  groupadd docker
  usermod --append --groups docker runner
  usermod --append --groups sudo runner
  groups runner # Print out the groups of runner to verify this worked

  echo "enabling passwordless sudo for runner user"
  echo "runner ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/99-runner

  #################################### Apt #####################################
  # Disable apt prompts
  export DEBIAN_FRONTEND="noninteractive"

  # Disable automatic updates and upgrades. These are ephemeral machines. We don't
  # want the latency or inconsistency of automatic updatees.
  systemctl stop apt-daily.timer
  systemctl disable apt-daily.timer
  systemctl disable apt-daily.service
  systemctl stop apt-daily-upgrade.timer
  systemctl disable apt-daily-upgrade.timer
  systemctl disable apt-daily-upgrade.service

  # Don't install documentation (except copyrights) since this is a CI system.
  cat > /etc/dpkg/dpkg.cfg.d/99-github-actions <<EOF
force-all
no-pager
# don't install docs
path-exclude /usr/share/doc/*
path-exclude /usr/share/man/*
path-exclude /usr/share/groff/*
path-exclude /usr/share/info/*
# keep copyright files for legal reasons
path-include /usr/share/doc/*/copyright
EOF

  # Provide default apt options like --assume-yes and --quiet since this is
  # designed to run on CI.
  cat > /etc/apt/apt.conf.d/99-github-actions <<EOF
APT {
  Install-Recommends "false";
  HideAutoRemove "true";
}
Aptitude {
  CmdLine {
    Assume-Yes "true";
  }
}
Acquire {
  Retries "5";
}
DPkg {
  Use-Pty "0";
  Options {
    "--force-confdef";
    "--force-confnew";
    "--force-confold";
  }
}
Quiet "2";
EOF

  # Install apt-fast for parallel apt package installation.
  add-apt-repository -y ppa:apt-fast/stable
  apt-get update
  apt-get install apt-fast
  apt-get upgrade
  apt-get dist-upgrade
  apt-get full-upgrade
  # Install common deps.
  apt-get install \
    apt-transport-https \
    aria2 \
    ca-certificates \
    curl \
    git \
    gnupg2 \
    jq \
    lsb-release \
    software-properties-common

  ############################## Fix gcloud Installation Snap ###############################

  # Snap literally won't let you disable automatic updates. The only thing
  # that's installed through snap here is the gcloud CLI, which we definitely
  # don't want automatically updating (beyond our general desire to not
  # automatically update on ephemeral machines). So we just delete snap entirely
  # and install the CLI via apt (above)
  systemctl stop snapd
  apt_maybe_purge snapd gnome-software-plugin-snap
  rm -rf /home/*/snap
  rm -rf /root/snap

  curl --silent --fail --show-error --location \
      https://packages.cloud.google.com/apt/doc/apt-key.gpg \
      | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
  echo \
      "deb [arch=amd64 signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
      > /etc/apt/sources.list.d/google-cloud-sdk.list
  apt-get update && apt-get install google-cloud-cli

  # This setting is now enabled by default. It sounds great, but unfortunately
  # doing such an upload requires *delete* permissions on the bucket, which we
  # deliberately do not give runners. For the life of me, I could not figure out
  # how to use `gcloud config set` (the "proper" way to set properties) to work
  # on the global properties.
  cat <<EOF >> /usr/lib/google-cloud-sdk/properties
[storage]
parallel_composite_upload_enabled = False
EOF

  ############################### Install Docker ###############################

  # Remove Docker stuff that may already be installed by all its various names
  apt_maybe_purge containerd docker docker-engine docker.io moby-engine moby-cli runc

  # Install the latest Docker
  curl --silent --fail --show-error --location \
    https://download.docker.com/linux/ubuntu/gpg \
    | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
  echo \
    "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
    > /etc/apt/sources.list.d/docker.list
  apt-get update
  apt-get install docker-ce docker-ce-cli containerd.io

  # Enable docker.service.
  sudo systemctl enable docker.service
  sudo systemctl start docker.service
  sudo systemctl enable containerd.service
  sudo systemctl start containerd.service

  # Docker daemon takes time to come up after installing.
  for i in $(seq 1 30); do
    if docker info; then
      break
    fi
  done

  # Make sure the runner user can use docker
  runuser --user runner -- docker ps

  ################################### Cleanup ##################################

  apt-get clean
  rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
  rm -rf /var/lib/dhcp/*

  # Delete unnecessary log files
  find /var/log -type f -regex ".*\.gz$" -delete
  find /var/log -type f -regex ".*\.[0-9]$" -delete

  # Clear all journal files
  journalctl --rotate --vacuum-time=1s

  # And clear others
  find /var/log/ -type f -exec truncate -s 0 {} \;

  echo "Disk usage after setup"
  df -h /

  echo "Setup complete"
}

startup 2>&1 | tee /startup.log
