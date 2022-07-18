#!/bin/bash

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Helper functions for other scripts. Should be broken into multiple files if
# we end up with a lot.
# This file should be sourced, not executed.

############################## Instance metadata ###############################

get_metadata() {
  local url="http://metadata.google.internal/computeMetadata/v1/${1}"
  ret=0
  curl "${url}" \
    --silent --fail --show-error \
    --header "Metadata-Flavor: Google" || ret=$?
  if [[ $ret != 0 ]]; then
    echo "Failed fetching ${url}" >&2
    return ${ret}
  fi
}

get_os_info() {
  get_metadata "instance/guest-attributes/guestInventory/${1}"
}

get_attribute() {
  get_metadata "instance/attributes/${1}"
}

################################################################################


############################## Utility functions ###############################

# Tests if the first argument is contained in the array in the second argument.
# Usage `is_contained "element" "${array[@]}"`
is_contained() {
  local e;
  local match="$1"
  shift
  for e in "$@"; do
    if [[ "$e" == "$match" ]]; then
      return 0
    fi
  done
  return 1
}

################################################################################
