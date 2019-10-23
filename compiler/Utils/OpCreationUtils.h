// Copyright 2019 Google LLC
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

// Utility functions related to the creation of new operations. Where possible,
// use custom builders. These helpers are for situations where a custom builder
// is not appropriate.

#ifndef IREE_COMPILER_UTILS_OPCREATIONUTILS_H_
#define IREE_COMPILER_UTILS_OPCREATIONUTILS_H_

#include <cstdint>

#include "compiler/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace iree_compiler {

IREE::ConstantOp createArrayConstant(OpBuilder &builder, Location loc,
                                     llvm::ArrayRef<int64_t> elements);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_UTILS_OPCREATIONUTILS_H_
