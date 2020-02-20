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

#ifndef IREE_COMPILER_TRANSLATION_CODEGENUTILS_CODEGENUTILS_H
#define IREE_COMPILER_TRANSLATION_CODEGENUTILS_CODEGENUTILS_H

#include "mlir/IR/Function.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace iree_compiler {

/// Drop trailing ones.
ArrayRef<int64_t> dropTrailingOnes(ArrayRef<int64_t> vector);

/// Checks that a given function is a dispatch function.
bool isDispatchFunction(FuncOp funcOp);

/// The launch size is the size of the outputs of the kernel. For now all
/// outputs have to be the same shape and static shaped.
// TODO(ravishankarm) : Modify this to return the Values to support dynamic
// shapes.
LogicalResult getLaunchSize(FuncOp funcOp,
                            SmallVectorImpl<int64_t> &launchSize);

/// Gets the workgroup size. Has to be a static constant.
template <typename intType>
LogicalResult getWorkGroupSize(FuncOp funcOp,
                               SmallVectorImpl<intType> &workGroupSize);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_TRANSLATION_CODEGENUTILS_CODEGENUTILS_H
