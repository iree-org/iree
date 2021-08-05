// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialects/iree_public/IREEPublicOps.h"

#include "iree-dialects/Dialects/iree_public/IREEPublicDialect.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "iree-dialects/Dialects/iree_public/IREEPublicOps.cpp.inc"
