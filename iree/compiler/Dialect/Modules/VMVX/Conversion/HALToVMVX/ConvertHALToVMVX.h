// Copyright 2021 Google LLC
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

#ifndef IREE_COMPILER_DIALECT_MODULES_VMVX_CONVERSION_HALTOVMVX_CONVERTHALTOVMVX_H_
#define IREE_COMPILER_DIALECT_MODULES_VMVX_CONVERSION_HALTOVMVX_CONVERTHALTOVMVX_H_

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

// Converts a `() -> ()` function to the calling convention used by VMVX for
// passing in bindings, constants, and workgroup parameters.
LogicalResult updateHALToVMVXEntryFuncOp(FuncOp funcOp,
                                         TypeConverter &typeConverter);

// Populates conversion patterns from the IREE HAL dialect interface to the
// VMVX dialect interface.
void populateHALToVMVXPatterns(MLIRContext *context,
                               OwningRewritePatternList &patterns,
                               TypeConverter &typeConverter);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_MODULES_VMVX_CONVERSION_HALTOVMVX_CONVERTHALTOVMVX_H_
