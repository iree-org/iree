# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

###############################################################################
# Re-export LLVM header-only targets as header-only libraries.                #
###############################################################################

add_library(MLIRLinalgStructuredOpsIncGenLib INTERFACE)
add_dependencies(MLIRLinalgStructuredOpsIncGenLib
  MLIRLinalgStructuredOpsIncGen
)
add_library(MLIRLinalgOpsIncGenLib INTERFACE)
add_dependencies(MLIRLinalgOpsIncGenLib
  MLIRLinalgOpsIncGen
)
