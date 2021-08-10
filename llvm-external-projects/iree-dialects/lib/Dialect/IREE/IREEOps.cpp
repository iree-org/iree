// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/IREE/IREEOps.h"

#include "iree-dialects/Dialect/IREE/IREEDialect.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "iree-dialects/Dialect/IREE/IREEOps.cpp.inc"
