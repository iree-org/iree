#!/bin/bash
# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Given a list of wheel files, launch a suitable docker container
# and audit/repair them.
# Usage: ./audit_wheels.sh <files>
# Typically:
#   ./build_tools/python_deploy/audit_wheels.sh \
#     ./llvm-projects/iree-compiler-api/wheels/iree_compiler_api-*.whl
#
# Note that this will mount the current directory under docker, so all
# files must be relative.

set -eu -o errtrace

this_dir="$(cd $(dirname $0) && pwd)"
script_name="$(basename $0)"
dockcross_image="manylinux2014-x64"

function run_on_host() {
  echo "Running on host"
  "$this_dir/setup_dockcross.sh" $dockcross_image

  echo "Running in docker..."
  "$this_dir/$dockcross_image" \
    --args "-v $this_dir:/python_deploy -e __MANYLINUX_BUILD_WHEELS_IN_DOCKER=1" \
    -- bash /python_deploy/$script_name "$@"
}

function run_in_docker() {
  for relative_path in "$@"; do
    abs_path="/work/$relative_path"
    echo "Repairing $relative_path ($abs_path)"
    auditwheel repair -w "$(dirname $abs_path)"/audited "$abs_path"
  done
}

# Trampoline to the docker container if running on the host.
if [ -z "${__MANYLINUX_BUILD_WHEELS_IN_DOCKER-}" ]; then
  run_on_host "$@"
else
  run_in_docker "$@"
fi
