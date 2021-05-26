// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_INTEGRATIONS_TENSORFLOW_CONVERTTFTOTFTENSORLIST_H_
#define IREE_INTEGRATIONS_TENSORFLOW_CONVERTTFTOTFTENSORLIST_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_integrations {
namespace tf_tensorlist {

std::unique_ptr<OperationPass<FuncOp>> createConvertTFToTFTensorListPass();

}  // namespace tf_tensorlist
}  // namespace iree_integrations
}  // namespace mlir

#endif  // IREE_INTEGRATIONS_TENSORFLOW_CONVERTTFTOTFTENSORLIST_H_
