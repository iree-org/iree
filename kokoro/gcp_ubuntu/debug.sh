#!/bin/bash

# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# A script that allows accessing a Kokoro VM for debugging.

# For interactive access, set the SSH_PUBLIC_KEY variable to your SSH public key
# and push to the kokoro-test branch, which will automatically trigger a build.
# It is also possible to trigger manual runs, which will run against kokoro-test
# HEAD.
# After setup the VM will print out the external IP address to the debug logs
# and you can SSH into kbuilder@INSTANCE_EXTERNAL_IP.
# You can also substitute other debugging commands here for non-interactive
# debugging or add additional setup before the sleep.

set -e

set -x

SSH_PUBLIC_KEY=""

echo "${SSH_PUBLIC_KEY:?}" >> ~/.ssh/authorized_keys
external_ip=$(curl -s -H "Metadata-Flavor: Google" http://metadata/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip)
echo "INSTANCE_EXTERNAL_IP=${external_ip}"
sleep 10000 # Keep the VM alive for a few hours
