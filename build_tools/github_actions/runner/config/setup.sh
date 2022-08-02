#!/bin/bash

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Sets up GitHub actions runner services to start and teardown runner.
# Registers the runner followed by enabling deregister service then 
# starts the registered runner to accept self-hosted GitHub workflows.

set -euo pipefail

echo "Register the self-hoster runner."
chmod +x "${HOME}/config/register.sh"
"${HOME}/config/register.sh"

echo "Setup the deregister service."
sudo cp "${HOME}/config/github-actions-runner-deregister.service /etc/systemd/system/"
chmod +x "${HOME}/config/deregister.sh"

echo "Setup the start actions runner service."
sudo cp "${HOME}/config/github-actions-runner-start.service /etc/systemd/system/"
chmod +x "${HOME}/config/start.sh"

echo "Reload system service files to reflect changes."
sudo systemctl daemon-reload

echo "Enable deregister so it can hook onto system shutdown."
sudo systemctl enable github-actions-runner-deregister

echo "Start the runner so it can be assigned workflows."
sudo systemctl start github-actions-runner-start
