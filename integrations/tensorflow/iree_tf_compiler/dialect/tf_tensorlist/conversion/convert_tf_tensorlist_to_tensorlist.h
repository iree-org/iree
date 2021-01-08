// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef IREE_INTEGRATIONS_TENSORFLOW_TFTENSORLIST_CONVERSION_CONVERTTFTENSORLISTTOTENSORLIST_H_
#define IREE_INTEGRATIONS_TENSORFLOW_TFTENSORLIST_CONVERSION_CONVERTTFTENSORLISTTOTENSORLIST_H_

#include "iree_tf_compiler/dialect/tf_tensorlist/ir/tf_tensorlist_types.h"
#include "iree/compiler/Dialect/Modules/TensorList/IR/TensorListDialect.h"
#include "iree/compiler/Dialect/Modules/TensorList/IR/TensorListTypes.h"
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
