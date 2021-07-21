# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(iree_test_matrix)

iree_test_matrix_gen()
# TODO: If this becomes the only type of test we glob, collapse it into the
# previous.
iree_test_matrix_glob_py_tests()
