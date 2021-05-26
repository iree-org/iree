// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_INTEGRATIONS_TENSORFLOW_TFTENSORLIST_CONVERSION_CONVERTTFTENSORLISTTOTENSORLIST_H_
#define IREE_INTEGRATIONS_TENSORFLOW_TFTENSORLIST_CONVERSION_CONVERTTFTENSORLISTTOTENSORLIST_H_

#include "iree/compiler/Dialect/Modules/TensorList/IR/TensorListDialect.h"
#include "iree/compiler/Dialect/Modules/TensorList/IR/TensorListTypes.h"
#include "iree_tf_compiler/dialect/tf_tensorlist/ir/tf_tensorlist_types.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_integrations {
namespace tf_tensorlist {

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTFTensorListToTensorListPass();

}  // namespace tf_tensorlist
}  // namespace iree_integrations
}  // namespace mlir

#endif
