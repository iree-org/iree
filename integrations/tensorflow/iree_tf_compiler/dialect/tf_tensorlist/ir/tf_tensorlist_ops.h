// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_INTEGRATIONS_TENSORFLOW_IREE_TF_COMPILER_TF_TENSORLIST_OPS_H_
#define IREE_INTEGRATIONS_TENSORFLOW_IREE_TF_COMPILER_TF_TENSORLIST_OPS_H_

#include "iree_tf_compiler/dialect/tf_tensorlist/ir/tf_tensorlist_types.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"

// Declares the operations for this dialect using the generated header.
#define GET_OP_CLASSES
#include "iree_tf_compiler/dialect/tf_tensorlist/ir/tf_tensorlist_ops.h.inc"

#endif  // IREE_INTEGRATIONS_TENSORFLOW_IREE_TF_COMPILER_TF_TENSORLIST_OPS_H_
