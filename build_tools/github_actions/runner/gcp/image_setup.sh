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


function startup() {
  #################################### APT #####################################
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
  cat > /etc/dpkg/dpkg.cfg.d/github-actions <<EOF
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
  cat > /etc/apt/apt.conf.d/github-actions <<EOF
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
  echo "runner ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/runner



  ############################### Install Docker ###############################

  # Remove Docker stuff that may already be installed, proceeding if they're not.
  apt-get remove containerd docker docker-engine docker.io moby-engine moby-cli runc || true

  # Install the latest Docker
  curl -sfSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
  echo \
    "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
    $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list
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

  ################################### Cleanup ####################################

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

  # This specific log line is load bearing, as it's referenced in create_image.sh
  echo "Setup complete"
}

startup 2>&1 | tee /startup.log
