# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

execute_process(
  COMMAND cmake -E rm -rf ${IREE_TEST_TMPDIR_ROOT}
  COMMAND cmake -E make_directory ${IREE_TEST_TMPDIRS}
)
