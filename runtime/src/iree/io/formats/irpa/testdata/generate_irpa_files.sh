#!/bin/bash
# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# TODO(benvanik): replace with python utils using numpy so we can get more than
# just splat data tensors. For now there's no python writer.

# To regenerate, build iree-create-parameters and run from the project root:
#  $ ./runtime/src/iree/io/formats/irpa/testdata/generate_irpa_files.sh
#
# If iree-create-parameters is not on your path you can set the env var:
#  $ IREE_CREATE_PARAMETERS=../iree-create-parameters \
#    ./runtime/src/iree/io/formats/irpa/testdata/generate_irpa_files.sh

# Uncomment to see the iree-create-parameters commands issued:
# set -x
set -e

ROOT_DIR=$(git rev-parse --show-toplevel)
TESTDATA="${ROOT_DIR}/runtime/src/iree/io/formats/irpa/testdata"
CREATE_PARAMETERS="${IREE_CREATE_PARAMETERS:-iree-create-parameters}"

CMD=(
  ${CREATE_PARAMETERS}
    --output=${TESTDATA}/empty.irpa
)
"${CMD[@]}"

CMD=(
  ${CREATE_PARAMETERS}
    --output=${TESTDATA}/single.irpa
    --data=key0=4xf32=100.1
)
"${CMD[@]}"

CMD=(
  ${CREATE_PARAMETERS}
    --output=${TESTDATA}/multiple.irpa
    --data=key0=4xf32=100.1
    --data=key1=5xi8=101
)
"${CMD[@]}"

CMD=(
  ${CREATE_PARAMETERS}
    --output=${TESTDATA}/mixed.irpa
    --data=key0=4xf32=100.1
    --data=key1=5xi8=101
    --splat=key2=i8=102
    --splat=key3=4096x1024xi64=103
)
"${CMD[@]}"
