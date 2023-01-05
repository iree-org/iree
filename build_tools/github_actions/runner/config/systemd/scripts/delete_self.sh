#!/bin/bash

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Calls the instance deletion proxy to delete this instance through the GCE API.

set -euo pipefail

source /runner-root/config/functions.sh

# If the nice way fails, hard shutdown
function shutdown_now() {
  sudo /usr/sbin/shutdown -P now
}

trap shutdown_now ERR

function delete_self() {
  local self_deletion_service_url="$(get_attribute instance-self-deleter-url)"
  local id_token=$(get_metadata "instance/service-accounts/default/identity?audience=${self_deletion_service_url}&format=full")

  nice_curl -X DELETE --header "Authorization: Bearer ${id_token}" "${self_deletion_service_url}"
}

delete_self
