#!/bin/bash

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Updates the autoscaling configuration for a specific instance group.

function update_autoscaling() {
  local mig_name="$1"
  local region="$2"
  local min_size="$3"
  local max_size="$4"
  local -a autoscaling_args=(
    "${mig_name}"
    --project=iree-oss
    --region="${region}"
    --cool-down-period=60
    --min-num-replicas="${min_size}"
    --max-num-replicas="${max_size}"
    --mode=only-scale-out
    --target-cpu-utilization=0.2
  )

  (set -x; gcloud beta compute instance-groups managed set-autoscaling "${autoscaling_args[@]}")
}

update_autoscaling "$@"
