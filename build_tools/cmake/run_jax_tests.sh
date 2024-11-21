#!/bin/bash

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -xeo pipefail

pjrt_platform=$1

if [ -z "${pjrt_platform}" ]; then
    set +x
    echo "Usage: run_jax_tests.sh <pjrt_platform>"
    echo "  <pjrt_platform> can be 'cpu', 'cuda', 'rocm' or 'vulkan'"
    exit 1
fi

# cd into the PJRT plugin dir
ROOT_DIR="${ROOT_DIR:-$(git rev-parse --show-toplevel)}"
cd "${ROOT_DIR}/integrations/pjrt"

# perform some differential testing
actual_jax_platform=iree_${pjrt_platform}
expected_jax_platform=cpu

# this function will execute the test python script in
# both cpu mode and the IREE PJRT mode,
# and then compare the difference in the output
diff_jax_test() {
    local test_py_file=$1

    echo "executing ${test_py_file} in ${expected_jax_platform}.."
    local expected_tmp_out=$(mktemp /tmp/jax_test_result_expected.XXXXXX)
    JAX_PLATFORMS=$expected_jax_platform python $test_py_file > $expected_tmp_out

    echo "executing ${test_py_file} in ${actual_jax_platform}.."
    local actual_tmp_out=$(mktemp /tmp/jax_test_result_actual.XXXXXX)
    JAX_PLATFORMS=$actual_jax_platform python $test_py_file > $actual_tmp_out

    echo "comparing ${expected_tmp_out} and ${actual_tmp_out}.."
    diff --unified $expected_tmp_out $actual_tmp_out
    echo "no difference found"
}

diff_jax_test test/test_simple.py

# FIXME: we can also utilize the native test cases from JAX,
# e.g. `tests/nn_test.py` from the JAX repo, as below,
# but currently some test cases in this file will fail.
# NOTE that `absl-py` is required to run these tests.

# local jax_nn_test_file=$(mktemp /tmp/jax_nn_test.XXXXXX.py)
# wget https://github.com/jax-ml/jax/blob/jax-v0.4.20/tests/nn_test.py -O $jax_nn_test_file
# JAX_PLATFORMS=$actual_jax_platform python $jax_nn_test_file
