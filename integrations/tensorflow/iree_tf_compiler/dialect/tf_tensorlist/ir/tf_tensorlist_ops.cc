// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree_tf_compiler/dialect/tf_tensorlist/ir/tf_tensorlist_ops.h"

#include "mlir/IR/Builders.h"

#define GET_OP_CLASSES
#include "iree_tf_compiler/dialect/tf_tensorlist/ir/tf_tensorlist_ops.cc.inc"
